"""使用已准备的无 Skill 配对实验目录运行单个公开案例，不限制代码请求次数。"""

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--suite", type=Path, required=True, help="已准备 code、contexts 和 results 输入的目录"
)
parser.add_argument(
    "--case",
    required=True,
    choices=[
        "01-ai4i",
        "02-hydraulic",
        "03-secom",
        "04-steel",
        "05-concrete",
        "06-airfoil",
        "07-bike",
        "08-bank",
        "09-wine",
        "10-energy",
    ],
)
parser.add_argument("--arm", required=True, choices=["a", "b"])
parser.add_argument("--provider", default="custom")
ARGS = parser.parse_args()
SUITE = ARGS.suite.resolve(strict=True)
CASE_ID, ARM = ARGS.case, ARGS.arm
BASE = SUITE / "results" / CASE_ID / ARM
CODE = SUITE / "code"
CONTEXT = SUITE / "contexts" / ARM
RUN = BASE / "case"
AUDIT = BASE / "audit"
if (BASE / "status.json").exists():
    raise FileExistsError("拒绝覆盖已有实验；请准备新的实验目录")
AUDIT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE / "src"))
load_dotenv(PROJECT / ".env", override=False)
os.environ["LITELLM_MODE"] = "PRODUCTION"
os.chdir(CONTEXT)

from factor_demo import autoresearch_web as web
from factor_demo.autoresearch_agent_v2.agent import (
    Session,
    history,
)
from factor_demo.autoresearch_agent_v2.agent import loop as loop_module
from factor_demo.autoresearch_agent_v2.agent import session as session_module
from factor_demo.autoresearch_agent_v2.agent.history import COMPACT_PROMPT, get_context_messages
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox, SandboxConfig
from factor_demo.autoresearch_agent_v2.tools.role_policy import ROUTER_TOOLS, WORKER_TOOLS
from factor_demo.live_events import LiveEvents
from factor_demo.providers import load_provider
from factor_demo.runtime import RuntimeSettings

USED = 0
REQUESTS = 0
SCHEMAS = {}
ROUTER_TOOLS = tuple(name for name in ROUTER_TOOLS if not name.startswith("skill__"))


def now():
    """保存明确时区的事件时间。"""
    return datetime.now(UTC).isoformat()


def write(path, value):
    """原子保存宿主状态，避免观察者读到半份 JSON。"""
    try:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as exc:
        print(f"audit write failed: {path.name}: {type(exc).__name__}", file=sys.stderr, flush=True)


def append(path, value):
    """保存逐行审计记录，不记录凭据或连接配置。"""
    try:
        with path.open("a") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"audit append failed: {path.name}: {type(exc).__name__}", file=sys.stderr, flush=True
        )


ORIGINAL_COMPLETION = Model.completion


async def traced_completion(self, messages, *, tools=None, tool_choice=None):
    """保留真实模型输入和返回消息，敏感连接配置不进入记录。"""
    global REQUESTS
    REQUESTS += 1
    identity = self.state.agent_id
    number = REQUESTS
    started = time.monotonic()
    stem = f"{number:04d}-{identity}"
    write(
        AUDIT / f"{stem}-request.json",
        {"messages": messages, "tools": tools, "tool_choice": tool_choice},
    )
    append(
        AUDIT / "model_events.jsonl",
        {
            "at": now(),
            "event": "start",
            "request": number,
            "agent_id": identity,
            "parent_agent_id": self.state.parent_agent_id,
            "assignment_id": self.state.assignment_id,
            "message_count": len(messages),
            "tool_count": len(tools or []),
            "purpose": "compaction" if messages[-1].get("content") == COMPACT_PROMPT else "task",
        },
    )
    original_callback = self.usage_callback

    def record_usage(usage):
        """关联服务端用量与当前请求，同时保留原始持久化。"""
        append(
            AUDIT / "request_usage.jsonl",
            {
                "at": now(),
                "request": number,
                "agent_id": identity,
                "parent_agent_id": self.state.parent_agent_id,
                "assignment_id": self.state.assignment_id,
                "purpose": "compaction"
                if messages[-1].get("content") == COMPACT_PROMPT
                else "task",
                "usage": usage,
            },
        )
        if original_callback is not None:
            original_callback(usage)

    self.usage_callback = record_usage
    try:
        result = await ORIGINAL_COMPLETION(
            self,
            messages,
            tools=tools,
            **({} if tool_choice is None else {"tool_choice": tool_choice}),
        )
        write(AUDIT / f"{stem}-response.json", result)
        append(
            AUDIT / "model_events.jsonl",
            {
                "at": now(),
                "event": "end",
                "request": number,
                "agent_id": identity,
                "elapsed_s": time.monotonic() - started,
                "tool_names": [c["function"]["name"] for c in (result.get("tool_calls") or [])],
            },
        )
        return result
    except BaseException as exc:
        append(
            AUDIT / "model_events.jsonl",
            {
                "at": now(),
                "event": "error",
                "request": number,
                "agent_id": identity,
                "elapsed_s": time.monotonic() - started,
                "error_type": type(exc).__name__,
            },
        )
        raise
    finally:
        self.usage_callback = original_callback


ORIGINAL_COMPACT = history.compact_history


async def traced_compact(state, model, **options):
    """记录压缩前后有效上下文，审计不参与任务决策。"""

    def size():
        return model.count_tokens(
            [
                {"role": "system", "content": options["system_prompt"]},
                *get_context_messages(state),
            ],
            tools=options.get("tools"),
        )

    started = time.monotonic()
    record = {
        "at": now(),
        "agent_id": state.agent_id,
        "assignment_id": state.assignment_id,
        "reason": "worker_reassignment" if options.get("keep_recent_turns") == 0 else "pressure",
        "before_input_estimate": size(),
        "source_messages": len(state.messages),
    }
    append(AUDIT / "compaction_events.jsonl", {**record, "event": "start"})
    try:
        changed = await ORIGINAL_COMPACT(state, model, **options)
    except BaseException as exc:
        append(
            AUDIT / "compaction_events.jsonl",
            {
                **record,
                "at": now(),
                "event": "error",
                "error_type": type(exc).__name__,
                "elapsed_s": time.monotonic() - started,
            },
        )
        raise
    append(
        AUDIT / "compaction_events.jsonl",
        {
            **record,
            "at": now(),
            "event": "end",
            "changed": changed,
            "after_input_estimate": size(),
            "elapsed_s": time.monotonic() - started,
            "end_index": state.compaction.end_index if state.compaction else None,
            "summary_chars": len(state.compaction.summary) if state.compaction else 0,
        },
    )
    return changed


ORIGINAL_START = Session.start


async def checked_start(session):
    """核对实际权限，并记录各会话是否持有执行沙箱。"""
    result = await ORIGINAL_START(session)
    identity = session.state.agent_id
    role = "worker" if session.state.parent_agent_id else "router"
    expected = set(WORKER_TOOLS if role == "worker" else ROUTER_TOOLS)
    actual = set(session.tool_policy.allowed_names)
    assert session.skill_registry is None
    assert not session.initial_skills and not session.state.skill_snapshots
    assert not any(name.startswith("skill__") for name in actual)
    prompt = session.loop.build_system_prompt()
    assert "# 可用技能" not in prompt
    assert "# 已加载技能" not in prompt
    if actual != expected:
        raise RuntimeError(f"{role} 权限集合与确认表不一致")
    try:
        sandbox = session.resource(Sandbox).config.model_dump(mode="json")
    except RuntimeError:
        sandbox = None
    SCHEMAS[identity] = {
        "role": role,
        "parent_agent_id": session.state.parent_agent_id,
        "allowed_names": sorted(actual),
        "sandbox": sandbox,
        "native_schemas": session.schemas(),
    }
    write(AUDIT / "permissions.json", SCHEMAS)
    return result


ORIGINAL_OBSERVE = web.observe_tool_calls


def audited_tools(session, events):
    """复用网页工具通道并记录调用，不限制代码请求次数。"""
    ORIGINAL_OBSERVE(session, events)
    if getattr(session, "_case_audit_factory", None) is session.loop.tools:
        return
    session._case_audit_factory = session.loop.tools
    original = session.loop.tools.call

    async def invoke(name, arguments, *, call_id=None):
        """只记录真实调用和执行计数，不替模型实现研究算法。"""
        global USED
        count = (
            len(arguments.get("cells", []))
            if name == "notebook__run_cells"
            else (1 if name in {"notebook__exec_cell", "sandbox__run_shell"} else 0)
        )
        event = {
            "at": now(),
            "agent_id": session.state.agent_id,
            "assignment_id": session.state.assignment_id,
            "name": name,
            "call_id": call_id,
            "arguments": arguments,
        }
        USED += count
        write(BASE / "execution_count.json", {"used": USED, "limit": None})
        append(AUDIT / "tool_events.jsonl", {**event, "event": "start", "used": USED})
        started = time.monotonic()
        try:
            result = await original(name, arguments, call_id=call_id)
        except BaseException as exc:
            append(
                AUDIT / "tool_events.jsonl",
                {
                    **event,
                    "at": now(),
                    "started_at": event["at"],
                    "event": "error",
                    "error_type": type(exc).__name__,
                },
            )
            raise
        append(
            AUDIT / "tool_events.jsonl",
            {
                **event,
                "at": now(),
                "started_at": event["at"],
                "event": "end",
                "elapsed_s": time.monotonic() - started,
                "result": result,
            },
        )
        return result

    session.loop.tools.call = invoke


async def preflight():
    """真实执行前核对本机隔离、只读输入与凭据过滤。"""
    canary = BASE / "private_canary.txt"
    canary.write_text("host-only isolation probe")
    code = (
        """import os,json,socket
from pathlib import Path
result={}
result['credential_keys_present']=any(k in os.environ for k in ['FACTOR_API_KEY','FACTOR_API_BASE','OPENAI_API_KEY'])
result['input_read_allowed']=bool(Path(INPUT).read_text().splitlines()[0])
try:
 Path(PRIVATE).read_text()
 result['private_read_blocked']=False
except PermissionError:
 result['private_read_blocked']=True
try:
 with Path(INPUT).open('a'): pass
 result['input_write_blocked']=False
except PermissionError:
 result['input_write_blocked']=True
try:
 s=socket.socket()
 result['network_outbound_blocked']=s.connect_ex(('127.0.0.1',9)) in (1,13)
 s.close()
except PermissionError:
 result['network_outbound_blocked']=True
try:
 Path(SKILL).read_text()
 result['skill_files_read_blocked']=False
except PermissionError:
 result['skill_files_read_blocked']=True
print(json.dumps(result))
""".replace("INPUT", repr(str(RUN / "input/train.csv")))
        .replace("PRIVATE", repr(str(canary)))
        .replace("SKILL", repr(str(SUITE / "retired-skills/autoresearch/SKILL.md")))
    )
    sandbox = Sandbox(SandboxConfig(workdir=RUN / "workspace", read_paths=(RUN / "input",)))
    try:
        result = await sandbox.run_shell(f"{shlex.quote(sys.executable)} -I -c {shlex.quote(code)}")
    finally:
        await sandbox.aclose()
    result = result.model_dump(mode="json")
    write(BASE / "preflight.json", result)
    if not result.get("ok"):
        raise RuntimeError("本机内部沙箱预检查未通过")
    checks = json.loads(result["stdout"])
    if checks.pop("credential_keys_present") or not all(checks.values()):
        raise RuntimeError("隔离或凭据过滤预检查未通过")


async def main():
    """运行公开原任务，保存原始结果，不在 Agent 运行中读取评分答案。"""
    started = time.monotonic()
    status = {"case": CASE_ID, "arm": ARM, "status": "running", "started_at": now()}
    write(BASE / "status.json", status)
    provider = load_provider(ARGS.provider)
    settings = RuntimeSettings.load(CONTEXT / "configs/runtime.json")
    write(
        BASE / "run_config.json",
        {
            "case": CASE_ID,
            "arm": ARM,
            "provider": ARGS.provider,
            "model": provider.model,
            "reasoning_effort": settings.reasoning_effort,
            "thinking": settings.thinking,
            "task_timeout_s": settings.task_timeout,
            "tool_timeout_s": 120,
            "code_execution_limit": None,
            "outer_runtime": "native",
            "inner_sandbox": "macos",
            "network": False,
            "source": str(CODE),
            "settings": settings.model_dump(),
            "max_tokens_sent": None,
            "skills_enabled": False,
            "comparison_suite": str(SUITE),
        },
    )
    assert settings.compact_on_worker_reassignment == (ARM == "a")
    events = LiveEvents(RUN, "noskills-paired-check")
    try:
        await preflight()
        web.SkillRegistry = lambda *args, **kwargs: None
        web.ROUTER_TOOLS = ROUTER_TOOLS
        session_module.compact_history = traced_compact
        loop_module.compact_history = traced_compact
        Model.completion = traced_completion
        Session.start = checked_start
        web.observe_tool_calls = audited_tools
        raw = (RUN / "input/user_prompt.txt").read_text()
        prompt = raw + (
            "\n\n运行约定：TASK.md、task.json、data_dictionary.json 已复制到当前工作目录。"
            "已登记的 DATASET 包含全部输入文件，读取 CSV 时使用 DATASET 的绝对路径。"
            "最终产物统一放在 artifacts/final/，文件名按任务要求；"
            "features_train.csv、features_validation.csv、features_test.csv 也放在该目录。"
            f"总时限{settings.task_timeout / 60:g}分钟，单次代码执行最多120秒。"
            "接近截止时间时整理已有材料并交付，不能把未完成工作说成完成。"
        )
        (BASE / "effective_user_prompt.txt").write_text(prompt)
        await web.research_turn(
            RUN,
            ARGS.provider,
            prompt,
            "research",
            events=events,
            timeout=settings.task_timeout,
        )
        status["status"] = "completed"
    except BaseException as exc:
        status.update(
            status=(
                "timeout"
                if isinstance(exc, TimeoutError)
                else "cancelled"
                if isinstance(exc, asyncio.CancelledError)
                else "failed"
            ),
            error_type=type(exc).__name__,
        )
        (BASE / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        events.close()
        status.update(
            finished_at=now(),
            elapsed_s=time.monotonic() - started,
            model_requests=REQUESTS,
            code_executions=USED,
        )
        write(BASE / "status.json", status)
        manifest = {
            str(p.relative_to(RUN / "workspace")): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((RUN / "workspace").rglob("*"))
            if p.is_file()
        }
        write(BASE / "artifact_hashes.json", manifest)
        print(json.dumps(status), flush=True)
    if status["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
