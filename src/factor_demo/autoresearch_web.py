"""将核心研究 Agent 接入现有 Web 会话、事件和产物接口。"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from pathlib import Path

from .autoresearch_agent_v2.agent import Session, SessionState
from .autoresearch_agent_v2.agent.model import Model
from .autoresearch_agent_v2.config import HistoryConfig, ModelConfig
from .autoresearch_agent_v2.feature_verifier import FeatureAnalyzeVerifier
from .autoresearch_agent_v2.python_tools import PythonTools
from .autoresearch_agent_v2.sandboxing import SandboxConfig
from .autoresearch_agent_v2.tools import FileTools, SandboxTools, TaskTools
from .case_lease import case_lease
from .datasets import DatasetSession
from .io_utils import write_json
from .live_events import MarkdownReplies
from .providers import load_provider
from .runtime import RuntimeSettings


def restore_state(path: Path, run: Path) -> SessionState:
    """恢复消息并补齐中断调用；Notebook 内存由新进程重新建立。"""
    if path.exists():
        state = SessionState.model_validate_json(path.read_text())
        answered = {m.get("tool_call_id") for m in state.messages if m["role"] == "tool"}
        missing = [
            c for m in state.messages for c in m.get("tool_calls", []) if c["id"] not in answered
        ]
        for call in missing:
            state.messages.append({
                "role": "tool", "tool_call_id": call["id"],
                "content": "上次执行中断，调用结果未确认；先检查已有产物，不要假定成功。",
            })
        return state
    return SessionState(
        session_id=uuid.uuid4().hex, agent_id="main", task_id=run.name,
        workdir=run / "workspace",
    )


class WebModel(Model):
    """保存模型请求边界，并将实际响应转换成前端已有事件。"""

    def __init__(self, config, state, checkpoint, events):
        """绑定会话、持久化位置和当前请求的事件通道。"""
        super().__init__(config, usage_callback=self.record_usage)
        self.state = state
        self.checkpoint = checkpoint
        self.events = events

    def record_usage(self, usage):
        """保存服务端报告的用量，不将其当作旧版预算余额。"""
        with (self.checkpoint.parent / "usage.jsonl").open("a") as out:
            out.write(json.dumps({"task_id": self.events.task_id, "usage": usage}) + "\n")

    async def completion(self, messages, *, tools=None):
        """请求前保存历史，返回后发布真实模型内容及推理。"""
        for item in self.events.take_guidance_records():
            guidance = {"role": "user", "content": item["message"]}
            self.state.messages.append(guidance)
            messages = [*messages, guidance]
            self.events.emit("steering.applied", message=item["message"], source=item["source"])
        write_json(self.checkpoint, self.state.model_dump(mode="json"))
        call_id = uuid.uuid4().hex
        self.events.emit("model.start", call_id=call_id)
        self.events.emit("model.waiting")
        try:
            result = await super().completion(messages, tools=tools)
            for field, kind in (("reasoning_content", "reasoning"), ("content", "content")):
                if isinstance(result.get(field), str):
                    self.events.emit(f"model.{kind}.delta", call_id=call_id, text=result[field])
            return result
        finally:
            self.events.emit("model.stream.end", call_id=call_id)


async def research_turn(run, provider_name, message, mode, *, events, timeout=3600, live=True):
    """在现有后台进程中执行核心 Agent，支持持久化、指导和取消。"""
    run = run.resolve()
    with case_lease(run):
        await _research_turn(run, provider_name, message, mode, events, timeout)


async def _research_turn(run, provider_name, message, mode, events, timeout):
    """绑定当前数据版本和研究工具，将最终回复交给现有前端。"""
    folder = run / ".autoresearch"
    folder.mkdir(exist_ok=True)
    checkpoint = folder / "session.json"
    state = restore_state(checkpoint, run)
    provider = load_provider(provider_name)
    settings = RuntimeSettings.load(Path("configs/runtime.json"))
    thinking = {"type": "enabled" if settings.thinking else "disabled"}
    if provider.api_style == "openai" and provider.model.lower().startswith("glm-"):
        thinking["clear_thinking"] = False
    config = ModelConfig(
        model=f"{provider.api_style}/{provider.model}", api_key=provider.api_key,
        base_url=provider.base_url, timeout_s=settings.request_timeout,
        thinking=thinking,
        reasoning_effort=settings.reasoning_effort if provider.api_style == "openai" else None,
        tool_choice="auto" if provider.model.lower().startswith("deepseek") else "required",
    )
    dataset = DatasetSession(run)
    data_path = dataset.path()
    current = dataset.current()
    state.datasets.clear()
    model = WebModel(config, state, checkpoint, events)
    session = Session(model, state, history_config=HistoryConfig(
        context_window_tokens=settings.input_budget,
    ))
    session.register_dataset("DATASET", str(data_path), current.get("title", "当前会话数据"))
    SandboxTools.register(session, config=SandboxConfig(
        workdir=state.workdir, read_paths=(data_path,), timeout_s=120,
    ))
    TaskTools.register(session, storage_root=folder / "tasks")
    session.register_stateful_tool(FileTools)
    session.register_stateful_tool(PythonTools)
    session.register_stateful_tool(FeatureAnalyzeVerifier)
    write_json(run / "status.json", {
        "status": "running", "mission": events.task_id, "engine": "autoresearch_v2",
    })
    events.emit("turn.start", mode=mode, engine="autoresearch_v2")
    owner = asyncio.current_task()

    async def watch_cancel():
        """轮询持久化停止标记，通过取消等待关闭 kernel。"""
        while True:
            if (run / "requests" / f"{events.task_id}.cancel").exists():
                owner.cancel()
                return
            await asyncio.sleep(0.2)

    watcher = asyncio.create_task(watch_cancel())
    status = "interrupted"
    try:
        async with asyncio.timeout(timeout):
            async with session:
                original_call = session.loop.tools.call

                async def observed_call(name, arguments, *, call_id=None):
                    """复用现有工具事件，并对讨论模式限制写入和执行能力。"""
                    started = time.monotonic()
                    events.emit("agent.tool.start", name=name, tool_call_id=call_id,
                                arguments=arguments)
                    ok = False
                    output = ""
                    try:
                        if mode != "research" and name not in {
                            "finish", "read_file", "task_list", "task_get",
                        }:
                            raise PermissionError("讨论模式只能读取已有研究记录，不能执行或修改。")
                        result = await original_call(name, arguments, call_id=call_id)
                        output = json.dumps(result, ensure_ascii=False, default=str)
                        ok = not (isinstance(result, dict) and result.get("error"))
                        if isinstance(result, dict) and result.get("status") == "error":
                            ok = False
                        if isinstance(result, dict) and result.get("returncode", 0) != 0:
                            ok = False
                        return result
                    except BaseException as error:
                        output = type(error).__name__
                        raise
                    finally:
                        events.emit("agent.tool.end", name=name, tool_call_id=call_id, ok=ok,
                                    elapsed_s=time.monotonic() - started, output=output[:12000])

                session.loop.tools.call = observed_call
                instruction = (
                    "本次请求在新的执行进程中继续。历史 Notebook 文件仍在，但内存未恢复；"
                    "需要计算时新建 Notebook 并重新加载数据。当前 DATASET 是前端所选数据版本。\n"
                )
                if mode != "research":
                    instruction += "本次仅讨论已有结果，不启动新研究，不执行代码或写入文件。\n"
                result = await session.run(instruction + message)
        output_dir = run / "requests" / events.task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "final_response.md").write_text(result)
        replies = MarkdownReplies(events)
        replies.publication = "final"
        replies.send("final", result)
        status = "checkpoint"
    except asyncio.CancelledError:
        status = "cancelled"
        raise
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        write_json(checkpoint, state.model_dump(mode="json"))
        write_json(run / "status.json", {
            "status": status, "mission": events.task_id, "engine": "autoresearch_v2",
        })
