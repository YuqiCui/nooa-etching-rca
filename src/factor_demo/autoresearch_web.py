"""将核心研究 Agent 接入现有 Web 会话、事件和产物接口。"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

from .autoresearch_agent_v2.agent import Session, SessionState
from .autoresearch_agent_v2.agent.model import Model
from .autoresearch_agent_v2.config import HistoryConfig, ModelConfig
from .autoresearch_agent_v2.runtime import AgentRuntime, SessionFactory
from .autoresearch_agent_v2.sandboxing import Sandbox, SandboxConfig
from .autoresearch_agent_v2.skills import SkillRegistry
from .autoresearch_agent_v2.tools import SandboxTools, TaskTools
from .autoresearch_agent_v2.tools.builtin_tools.records import RecordTools
from .autoresearch_agent_v2.tools.call_context import execution_id
from .autoresearch_agent_v2.tools.registry import ToolRegistry, register_builtin_tools
from .business_tools import register_business_tools
from .case_lease import case_lease
from .datasets import DatasetSession
from .io_utils import write_json
from .live_events import LiveEvents, MarkdownReplies
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
        self._event_source = events
        self._events = None

    @property
    def events(self):
        """首次需要事件时才打开子连接，未通过启动校验时不创建外部资源。"""
        if self._events is None:
            self._events = (
                self._event_source() if callable(self._event_source) else self._event_source
            )
        return self._events

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
    with (
        case_lease(run),
        contextlib.closing(
            SkillRegistry([Path(__file__).resolve().parent / "knowledge" / "v2"])
        ) as skill_registry,
    ):
        await _research_turn(
            run, provider_name, message, mode, events, timeout, skill_registry=skill_registry,
        )


def observe_tool_calls(session, events):
    """为获准工具复用现有事件，不另行定义业务权限。"""
    original_call = session.loop.tools.call

    async def observed_call(name, arguments, *, call_id=None):
        """记录实际工具调用；工具策略在统一入口执行。"""
        started = time.monotonic()
        events.emit("agent.tool.start", name=name, tool_call_id=call_id,
                    arguments=arguments)
        ok = False
        output = ""
        try:
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


def register_progress_tool(session, event_source):
    """为当前会话挂载进展工具，事件连接由宿主提供或延迟取得。

    Args:
        session: 尚未启动的 Agent 会话。
        event_source: 当前会话的事件连接，或返回该连接的无参数函数。
    """
    def report_progress(content: str) -> dict:
        """发布阶段性 Markdown 消息，保留当前研究循环。

        Args:
            content: 向用户展示的非空进展正文，可包含 Markdown 表格和产物链接。

        Returns:
            已发布的消息标识与投递状态。

        Raises:
            ValueError: 正文为空，或当前事件连接已经关闭。
        """
        if not content.strip():
            raise ValueError("阶段性进展不能为空")
        events = event_source() if callable(event_source) else event_source
        if events.closed:
            raise ValueError("当前进展通道已经关闭")
        identity = execution_id.get() or uuid.uuid4().hex
        message_id = "progress-" + uuid.uuid5(
            uuid.NAMESPACE_OID, f"{session.state.session_id}:{identity}"
        ).hex
        return MarkdownReplies(events).send(message_id, content)

    session.loop.tools.register_callable(
        "report_progress", report_progress,
        description="向用户发布阶段性进展，输入为 Markdown 正文字符串；不会结束当前工作。",
    )
    session.loop.tools.add_instructions(
        "使用 report_progress(content) 向用户汇报阶段性进展，说明已有发现、"
        "必要的不确定性或下一步。调用后继续执行任务；最终结果使用 finish(content) 提交。",
        names=("report_progress",),
    )


def create_runtime(run, folder, config, events):
    """装配通用运行时，隔离子会话的状态、事件与宿主文档。"""
    def child_model(parent, request):
        """为子会话沿用事件协议，并分开记录真实模型用量。"""
        def child_events():
            """会话启动通过后再创建当前子任务的事件连接。"""
            return LiveEvents(
                run, events.task_id, root=events.root, agent_id=request.state.agent_id,
                parent_agent_id=parent.state.agent_id,
            )
        child_checkpoint = folder / "children" / request.state.agent_id / "session.json"
        child_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        return WebModel(config, request.state, child_checkpoint, child_events)

    def configure_child(child):
        """仅装配通用状态和事件资源，不改变任务正文或研究方法。"""
        TaskTools.register(child, storage_root=folder / "tasks")
        RecordTools.register(child, storage_root=folder / "records")
        register_progress_tool(child, lambda: child.loop.model.events)

        @asynccontextmanager
        async def event_scope(current):
            """随子会话清理独立事件连接。"""
            child_events = current.loop.model.events
            try:
                yield child_events
            finally:
                child_events.close()

        child.register_resource(LiveEvents, event_scope)
        previous = child.loop.before_run_hook

        async def before_run(content):
            """启动资源后给子工具安装相同的事件记录。"""
            observe_tool_calls(child, child.loop.model.events)
            return await previous(content)

        child.loop.before_run_hook = before_run

    runtime = AgentRuntime(
        folder / "runtime", SessionFactory(configure=configure_child, model_factory=child_model)
    )
    return runtime


async def _research_turn(run, provider_name, message, mode, events, timeout, *, skill_registry):
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
    registry = ToolRegistry()
    register_builtin_tools(registry)
    register_business_tools(registry)
    allowed = "all" if mode == "research" else [
        "finish", "report_progress", "read_file", "task_list", "task_get", "record__get",
    ]
    session = Session(
        model, state, tool_registry=registry, allow_tools=allowed,
        history_config=HistoryConfig(context_window_tokens=settings.input_budget),
        skill_registry=skill_registry,
        skills=("autoresearch",) if mode == "research" else (),
    )
    session.register_dataset("DATASET", str(data_path), current.get("title", "当前会话数据"))
    session.register_resource(
        Sandbox,
        partial(SandboxTools.create_sandbox, config=SandboxConfig(
            workdir=state.workdir, read_paths=(data_path,), timeout_s=120,
        )),
        eager=False,
    )
    TaskTools.register(session, storage_root=folder / "tasks")
    RecordTools.register(session, storage_root=folder / "records")
    register_progress_tool(session, events)

    runtime = create_runtime(run, folder, config, events) if mode == "research" else None
    if runtime is not None:
        runtime.attach(session)
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
                observe_tool_calls(session, events)
                instruction = (
                    "本次请求在新的执行进程中继续。历史 Notebook 文件仍在，但内存未恢复；"
                    "需要计算时新建 Notebook 并重新加载数据。当前 DATASET 是前端所选数据版本。\n"
                )
                if mode != "research":
                    instruction += "本次仅讨论已有结果，不启动新研究，不执行代码或写入文件。\n"
                result = (
                    await runtime.run(session, instruction + message)
                    if runtime is not None
                    else await session.run(instruction + message)
                )
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
        if runtime is not None:
            await runtime.aclose()
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        write_json(checkpoint, state.model_dump(mode="json"))
        write_json(run / "status.json", {
            "status": status, "mission": events.task_id, "engine": "autoresearch_v2",
        })
