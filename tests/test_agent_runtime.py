"""用真实调度与临时存储验证五工具 Worker 协议，模型使用本地回调。"""

import asyncio
import json
from copy import deepcopy

import pytest

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.errors import UnknownToolError
from factor_demo.autoresearch_agent_v2.runtime import AgentRuntime, RuntimeLimits, SessionFactory
from factor_demo.autoresearch_agent_v2.tools.builtin_tools.workers import WorkerTools


def call(name, arguments, identity):
    """构造模型的原生工具调用。"""
    return {"id": identity, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def response(*calls):
    """构造一批工具调用。"""
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def delivery(content="result", ids=("D1",), outcome="completed"):
    """生成完整交付，负面或不确定内容仍可构成有效交付。"""
    return json.dumps({"outcome": outcome, "summary": content, "deliverables": [
        {"id": identity, "status": "provided", "content": content, "artifact_refs": []}
        for identity in ids
    ], "limitations": []})


class FixtureModel:
    """只调用本地回调，保留实际循环和工具消息协议。"""
    config = ModelConfig(model="openai/fixture", api_key="fixture")

    def __init__(self, complete):
        """保存回调及实际请求。"""
        self.complete = complete
        self.requests = []
        self.request_options = []

    def count_tokens(self, messages, *, tools=None):
        """让调度测试不依赖远程 tokenizer。"""
        return 100

    async def completion(self, messages, *, tools=None, tool_choice=None):
        """保留请求后执行测试模型。"""
        self.requests.append(deepcopy(messages))
        self.request_options.append({"tools": deepcopy(tools), "tool_choice": tool_choice})
        return await self.complete(messages)


def root_session(tmp_path, model=None, **options):
    """创建位于模型可写目录之外的宿主运行范围。"""
    workdir = tmp_path / "workspace"
    workdir.mkdir(exist_ok=True)

    async def unused(messages):
        raise AssertionError("本场景不应调用入口模型")

    return Session(model or FixtureModel(unused), SessionState(
        session_id="root-session", agent_id="main", task_id="project", workdir=workdir,
    ), **options)


async def finished(messages):
    """直接提交有效交付。"""
    return response(call("finish", {"content": delivery()}, "finish"))


async def new_worker(session):
    """通过新工具初始化并取得稳定 Worker 标识。"""
    result = await session.call("StartWorker", {"worker_type": "general"})
    assert result["ok"], result
    return result["worker_id"]


async def assign(session, worker_id, task="check", *, call_id=None):
    """通过新工具分配有清单的任务。"""
    return await session.call("AssignTask", {
        "worker_id": worker_id, "task": task, "deliver_list": ["检查结果与证据"],
    }, call_id=call_id)


async def collect(session, worker_id, *, call_id=None):
    """等待结果后显式收集。"""
    await session.call("WaitWorkers", {})
    return await session.call("CollectWorkerResult", {"worker_id": worker_id}, call_id=call_id)


def test_parallel_workers_and_explicit_collection(tmp_path):
    async def scenario():
        entered = set()
        both = asyncio.Event()
        prompts = []

        def model(parent, request):
            async def complete(messages):
                prompts.append(messages[0]["content"])
                entered.add(request.state.agent_id)
                if len(entered) == 2:
                    both.set()
                await asyncio.wait_for(both.wait(), 2)
                return await finished(messages)
            return FixtureModel(complete)

        session = root_session(tmp_path)
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(model_factory=model)) as runtime:
            runtime.attach(session)
            async with session:
                names = {t["function"]["name"] for t in session.schemas()}
                assert set(WorkerTools.tool_names) <= names
                assert not any(name.startswith("agent__") for name in names)
                assert "You are a Router." in session.loop.build_system_prompt()
                first, second = await new_worker(session), await new_worker(session)
                assert entered == set()
                assert all(w["status"] == "idle" for w in (
                    await session.call("CheckWorkerStatus", {})
                )["workers"])
                assert (await assign(session, first))["status"] == "busy"
                assert (await assign(session, second))["phase"] == "queued"
                while len(entered) < 2 or any(w["status"] == "busy" for w in (
                    await session.call("CheckWorkerStatus", {})
                )["workers"]):
                    await session.call("WaitWorkers", {})
                    await asyncio.sleep(0)
                await runtime.deliver(session)
                notices = [m["content"] for m in session.state.messages
                           if str(m.get("content", "")).startswith("Worker runtime notification")]
                assert all('"deliverables"' not in notice for notice in notices)
                assert "children_pending" in await runtime.finish(session, "early")
                for identity in (first, second):
                    result = await collect(session, identity)
                    assert result["status"] == "idle"
                    assert result["result"]["deliverables"][0]["content"] == "result"
                assert await runtime.finish(session, "done") is True
                assert all("You are a Worker." in prompt for prompt in prompts)
                assert all("You are a Router." not in prompt for prompt in prompts)
                assert {s.state.workdir for s in runtime._sessions.values()} == {session.state.workdir}
    asyncio.run(scenario())


def test_idle_busy_completed_reuse_preserves_context(tmp_path):
    async def scenario():
        session = root_session(tmp_path)
        runtime = AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(finished)))
        runtime.attach(session)
        try:
            async with session:
                identity = await new_worker(session)
                state = runtime._sessions[identity].state
                session_id = state.session_id
                first = await assign(session, identity, "first task")
                await session.call("WaitWorkers", {})
                assert (await assign(session, identity))["error"]["code"] == "result_not_collected"
                assert (await session.call("CheckWorkerStatus", {}))["workers"][0]["status"] == "completed"
                assert (await session.call("WaitWorkers", {}))["ready_workers"]
                assert (await collect(session, identity))["result"]["outcome"] == "completed"
                second = await assign(session, identity, "correct previous work")
                runtime._complete(identity, first["assignment_id"], json.loads(delivery("late")))
                assert runtime.store.get("worker", identity)["status"] == "busy"
                result = await collect(session, identity)
                assert first["assignment_id"] != second["assignment_id"]
                assert result["assignment_id"] == second["assignment_id"]
                assert state.session_id == session_id
                assert "first task" in json.dumps(state.messages)
                assert "correct previous work" in json.dumps(state.messages)
                assert (await session.call("WaitWorkers", {}))["reason"] == "no_pending_work"
        finally:
            await runtime.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("call_fields", [{}, {"tool_calls": None}, {"tool_calls": []}])
def test_repair_history_accepts_no_tool_calls(tmp_path, call_fields):
    """无工具调用的消息不妨碍补齐中断回执，也不改变已有的有效回执。"""
    state = root_session(tmp_path).state
    interrupted = response(call("read_file", {"path": "result.txt"}, "interrupted"))
    text = {"role": "assistant", "content": "继续核对已有结果。", **call_fields}
    completed = response(call("read_file", {"path": "result.txt"}, "completed"))
    receipt = {"role": "tool", "tool_call_id": "completed", "content": "result"}
    state.messages = [interrupted, text, completed, receipt]

    AgentRuntime._repair_history(state)

    assert [message["tool_call_id"] for message in state.messages if message["role"] == "tool"] == [
        "interrupted", "completed",
    ]
    assert state.messages[2:] == [text, completed, receipt]
    assert "结果未确认" in state.messages[1]["content"]


def test_worker_reuse_after_text_only_response(tmp_path):
    """Worker 返回合法的 null 工具列表后，仍能收集结果并接续下一项任务。"""
    async def scenario():
        calls = 0

        async def complete(messages):
            nonlocal calls
            calls += 1
            if calls % 2:
                return {"role": "assistant", "content": "检查完成，准备交付。", "tool_calls": None}
            return await finished(messages)

        session = root_session(tmp_path)
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(complete))) as runtime:
            runtime.attach(session)
            async with session:
                identity = await new_worker(session)
                first = await assign(session, identity, "first task")
                assert (await collect(session, identity))["result"]["outcome"] == "completed"
                second = await assign(session, identity, "continue previous work")
                assert second["ok"], second
                assert second["assignment_id"] != first["assignment_id"]
                result = await collect(session, identity)
                assert result["assignment_id"] == second["assignment_id"]
                assert result["result"]["outcome"] == "completed"
                assert result["status"] == "idle"
                assert calls == 4
    asyncio.run(scenario())


def test_validation_repair_and_failure_are_collectible(tmp_path):
    async def scenario():
        calls = 0

        async def repair(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return response(call("finish", {"content": delivery(ids=("wrong",))}, "bad"))
            assert "invalid_delivery" in messages[-1]["content"]
            return await finished(messages)

        session = root_session(tmp_path)
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(repair))) as runtime:
            runtime.attach(session)
            async with session:
                identity = await new_worker(session)
                await assign(session, identity)
                assert (await collect(session, identity))["result"]["outcome"] == "completed"
                assert calls == 2
    asyncio.run(scenario())


def test_finish_repair_limit_terminates_as_failed(tmp_path):
    async def bad(messages):
        return response(call("finish", {"content": "not JSON"}, "bad"))

    async def scenario():
        session = root_session(tmp_path)
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(bad))) as runtime:
            runtime.attach(session)
            async with session:
                identity = await new_worker(session)
                await assign(session, identity)
                result = (await collect(session, identity))["result"]
                assert result["outcome"] == "failed"
                assert result["deliverables"][0]["status"] == "unavailable"
    asyncio.run(scenario())


def test_busy_conflicts_timeout_and_host_cancellation(tmp_path):
    async def blocked(messages):
        await asyncio.Event().wait()

    async def scenario():
        session = root_session(tmp_path)
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(blocked)),
            limits=RuntimeLimits(wait_timeout_s=0.02)) as runtime:
            runtime.attach(session)
            async with session:
                identity = await new_worker(session)
                assert (await session.call("CollectWorkerResult", {"worker_id": identity}))[
                    "error"]["code"] == "no_result"
                await assign(session, identity)
                assert (await assign(session, identity))["error"]["code"] == "worker_busy"
                assert (await session.call("CollectWorkerResult", {"worker_id": identity}))[
                    "error"]["code"] == "not_ready"
                assert (await session.call("WaitWorkers", {}))["reason"] == "timeout"
                await runtime.cancel_children("main")
                assert (await collect(session, identity))["result"]["outcome"] == "cancelled"
    asyncio.run(scenario())


def test_tool_receipts_survive_reassignment(tmp_path):
    async def scenario():
        session = root_session(tmp_path)
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(finished))) as runtime:
            runtime.attach(session)
            async with session:
                arguments = {"worker_type": "general"}
                first = await session.call("StartWorker", arguments, call_id="start")
                assert first == await session.call("StartWorker", arguments, call_id="start")
                identity = first["worker_id"]
                assigned = await assign(session, identity, call_id="assign")
                result = await collect(session, identity, call_id="collect")
                await assign(session, identity, "new task")
                await session.call("WaitWorkers", {})
                assert result == await session.call("CollectWorkerResult", {
                    "worker_id": identity}, call_id="collect")
                assert assigned == await assign(session, identity, call_id="assign")
                assert (await session.call("CheckWorkerStatus", {}))["workers"][0]["status"] == "completed"
                assert (await assign(session, identity, "different", call_id="assign"))[
                    "error"]["code"] == "execution_conflict"
                await collect(session, identity)
    asyncio.run(scenario())


def test_ownership_and_worker_limits(tmp_path):
    async def scenario():
        session = root_session(tmp_path)
        other = root_session(tmp_path)
        other.state.agent_id = "other"
        other.state.session_id = "other-session"
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(finished)),
            limits=RuntimeLimits(max_workers=1)) as runtime:
            runtime.attach(session)
            runtime.attach(other)
            async with session, other:
                identity = await new_worker(session)
                assert (await session.call("StartWorker", {"worker_type": "unknown"}))[
                    "error"]["code"] == "unknown_worker_type"
                assert (await session.call("StartWorker", {"worker_type": "general"}))[
                    "error"]["code"] == "worker_limit"
                with pytest.raises(PermissionError):
                    await assign(other, identity)
                with pytest.raises(PermissionError):
                    await other.call("CollectWorkerResult", {"worker_id": identity})
                assert (await other.call("CheckWorkerStatus", {}))["workers"] == []
    asyncio.run(scenario())


def test_nested_wait_releases_single_execution_slot(tmp_path):
    async def scenario():
        session = root_session(tmp_path)

        def model(parent, request):
            calls = 0
            nested = None

            async def complete(messages):
                nonlocal calls, nested
                calls += 1
                if parent.state.parent_agent_id is not None:
                    return await finished(messages)
                if calls == 1:
                    return response(call("StartWorker", {"worker_type": "general"}, "start"))
                if calls == 2:
                    nested = json.loads(messages[-1]["content"])["worker_id"]
                    return response(call("AssignTask", {"worker_id": nested, "task": "nested",
                                                          "deliver_list": ["result"]}, "assign"))
                if calls == 3:
                    return response(call("WaitWorkers", {}, "wait"))
                if calls == 4:
                    return response(call("CollectWorkerResult", {"worker_id": nested}, "collect"))
                return await finished(messages)
            return FixtureModel(complete)

        async with AgentRuntime(tmp_path / "runtime", SessionFactory(model_factory=model),
                                limits=RuntimeLimits(max_concurrent=1)) as runtime:
            runtime.attach(session)
            async with session:
                identity = await new_worker(session)
                await assign(session, identity)
                result = await asyncio.wait_for(collect(session, identity), 3)
                assert result["result"]["outcome"] == "completed"
    asyncio.run(scenario())


def test_completed_result_restores_without_reexecution(tmp_path):
    async def scenario():
        session = root_session(tmp_path)
        runtime = AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(finished)))
        runtime.attach(session)
        await session.start()
        identity = await new_worker(session)
        await assign(session, identity)
        await session.call("WaitWorkers", {})
        state = session.state.model_copy(deep=True)
        await runtime.aclose()
        await session.aclose()

        def forbidden(parent, request):
            raise AssertionError("收集已有结果不能重启 Worker 模型")

        restored = root_session(tmp_path)
        restored.state = state
        restored.loop.session = state
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(model_factory=forbidden)) as runtime:
            runtime.attach(restored)
            async with restored:
                result = await collect(restored, identity)
                assert result["result"]["outcome"] == "completed"
                assert (await collect(restored, identity))["replayed"] is True
    asyncio.run(scenario())


def test_closed_tool_references_are_revoked(tmp_path):
    async def scenario():
        session = root_session(tmp_path)
        runtime = AgentRuntime(tmp_path / "runtime")
        runtime.attach(session)
        await session.start()
        tools = session.resource(WorkerTools)
        await session.aclose()
        with pytest.raises(RuntimeError, match="关闭"):
            tools.status()
        await runtime.aclose()
    asyncio.run(scenario())


def test_invalid_factory_cannot_reuse_parent(tmp_path):
    async def scenario():
        session = root_session(tmp_path)
        async with AgentRuntime(tmp_path / "runtime", lambda parent, request: parent) as runtime:
            runtime.attach(session)
            async with session:
                result = await session.call("StartWorker", {"worker_type": "general"})
                assert result["error"]["code"] == "initialization_failed"
                assert (await session.call("CheckWorkerStatus", {}))["workers"] == []
                assert session.schemas()
    asyncio.run(scenario())


def test_wait_batch_executes_nothing(tmp_path):
    async def scenario():
        calls = 0

        async def complete(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return response(call("StartWorker", {"worker_type": "general"}, "start"),
                                call("WaitWorkers", {}, "wait"))
            assert all("本批次所有工具均未执行" in m["content"] for m in messages[-2:])
            return response(call("finish", {"content": "done"}, "finish"))

        session = root_session(tmp_path, FixtureModel(complete))
        async with AgentRuntime(tmp_path / "runtime") as runtime:
            assert await runtime.run(session, "test") == "done"
            assert runtime.check_worker_status(session)["workers"] == []
        await session.aclose()
    asyncio.run(scenario())


def test_user_message_interrupts_wait_without_cancelling_worker(tmp_path):
    async def blocked(messages):
        await asyncio.Event().wait()

    async def scenario():
        session = root_session(tmp_path)
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(blocked))) as runtime:
            runtime.attach(session)
            async with session:
                identity = await new_worker(session)
                await assign(session, identity)
                waiter = asyncio.create_task(session.call("WaitWorkers", {}))
                await asyncio.sleep(0)
                runtime.interrupt_wait(session)
                result = await asyncio.wait_for(waiter, 1)
                assert result["reason"] == "interrupted"
                assert result["cause"] == "user_message"
                assert result["busy_workers"] == [identity]
                await runtime.cancel_children("main")
                await collect(session, identity)
    asyncio.run(scenario())


def test_interrupted_execution_fails_without_repeating_side_effects(tmp_path):
    async def scenario():
        entered = asyncio.Event()

        async def blocked(messages):
            entered.set()
            await asyncio.Event().wait()

        session = root_session(tmp_path)
        runtime = AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(blocked)))
        runtime.attach(session)
        await session.start()
        identity = await new_worker(session)
        await assign(session, identity)
        await entered.wait()
        state = session.state.model_copy(deep=True)
        await runtime.aclose()
        await session.aclose()
        calls = 0

        async def complete(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return response(call("CollectWorkerResult", {"worker_id": identity}, "collect"))
            result = json.loads(messages[-1]["content"])
            assert result["result"]["outcome"] == "failed"
            assert "不自动重跑" in result["result"]["summary"]
            return response(call("finish", {"content": "interrupted"}, "finish"))

        def forbidden(parent, request):
            raise AssertionError("不确定的执行不能自动重跑")

        restored = Session(FixtureModel(complete), state)
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(model_factory=forbidden)) as runtime:
            assert await runtime.run(restored, "continue") == "interrupted"
        await restored.aclose()
    asyncio.run(scenario())


def test_worker_file_tools_share_workspace_and_validate_artifacts(tmp_path):
    async def scenario():
        from factor_demo.autoresearch_agent_v2.tools import FileTools

        session = root_session(tmp_path)
        session.register_stateful_tool(FileTools)
        calls = 0

        async def complete(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return response(call("write_file", {"path": "artifacts/verified_summary.txt",
                                                    "content": "shared result"}, "write"))
            result = json.loads(delivery())
            result["deliverables"][0]["artifact_refs"] = ["artifacts/verified_summary.txt"]
            return response(call("finish", {"content": json.dumps(result)}, "finish"))

        async with AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(complete))) as runtime:
            runtime.attach(session)
            async with session:
                identity = await new_worker(session)
                await assign(session, identity)
                result = await collect(session, identity)
                assert result["result"]["outcome"] == "completed"
                assert "shared result" in await session.call("read_file", {
                    "path": result["result"]["deliverables"][0]["artifact_refs"][0],
                })
    asyncio.run(scenario())


def test_native_sandbox_resources_are_independent_and_cancellation_cleans_up(tmp_path):
    async def scenario():
        from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox
        from factor_demo.autoresearch_agent_v2.tools import SandboxTools

        entered = asyncio.Event()
        calls = 0
        child_sandbox = None

        def model(parent, request):
            async def complete(messages):
                nonlocal calls, child_sandbox
                calls += 1
                if calls == 1:
                    child_sandbox = runtime._sessions[request.state.agent_id].resource(Sandbox)
                    return response(call("sandbox__run_shell", {
                        "command": "cp parent.txt child.txt",
                    }, "shell"))
                assert json.loads(messages[-1]["content"])["ok"]
                entered.set()
                await asyncio.Event().wait()
            return FixtureModel(complete)

        session = root_session(tmp_path)
        SandboxTools.register(session)
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(model_factory=model)) as runtime:
            runtime.attach(session)
            async with session:
                result = await session.call("sandbox__run_shell", {"command": "echo shared > parent.txt"})
                assert result["ok"], result
                identity = await new_worker(session)
                await assign(session, identity)
                await asyncio.wait_for(entered.wait(), 5)
                parent_sandbox = session.resource(Sandbox)
                assert child_sandbox is not parent_sandbox
                assert child_sandbox.config == parent_sandbox.config
                await runtime.cancel_children("main")
                await collect(session, identity)
                assert (session.state.workdir / "child.txt").read_text().strip() == "shared"
                # Worker 可复用，单次取消不关闭会话沙箱；根运行结束后统一释放。
        with pytest.raises(RuntimeError, match="closed"):
            await child_sandbox.run_shell("echo closed")
    asyncio.run(scenario())


def test_worker_capabilities_do_not_require_router_execution_permissions(tmp_path):
    async def scenario():
        from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox, SandboxConfig
        from factor_demo.autoresearch_agent_v2.tools import SandboxTools

        allowed = ["finish", *WorkerTools.tool_names]
        session = root_session(tmp_path, allow_tools=allowed)
        approved_input = tmp_path / "approved-input"
        approved_input.mkdir()
        config = SandboxConfig(workdir=session.state.workdir, read_paths=(approved_input,))
        SandboxTools.register(session, config=config)
        async with AgentRuntime(tmp_path / "runtime", SessionFactory(
            model_factory=lambda parent, request: FixtureModel(finished))) as runtime:
            runtime.attach(session)
            async with session:
                with pytest.raises(RuntimeError, match="not ready"):
                    session.resource(Sandbox)
                with pytest.raises((PermissionError, UnknownToolError)):
                    await session.call("sandbox__run_shell", {"command": "echo forbidden"})
                identity = await new_worker(session)
                child = runtime._sessions[identity]
                assert "sandbox__run_shell" in {t["function"]["name"] for t in child.schemas()}
                assert child.resource(Sandbox).config == config
                await assign(session, identity)
                assert (await collect(session, identity))["result"]["outcome"] == "completed"
    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", ["duplicate", "unavailable", "missing", "outside"])
def test_delivery_rejects_invalid_items_and_artifact_paths(tmp_path, invalid):
    """交付清单与真实产物引用均受检查，不能把私有目录文件作为交付。"""
    from factor_demo.autoresearch_agent_v2.runtime.delivery import validate_delivery

    workdir = tmp_path / "workspace"
    workdir.mkdir()
    private = tmp_path / "private.txt"
    private.write_text("private")
    result = json.loads(delivery())
    item = result["deliverables"][0]
    if invalid == "duplicate":
        result["deliverables"].append(dict(item))
    elif invalid == "unavailable":
        item["status"] = "unavailable"
    elif invalid == "missing":
        item["artifact_refs"] = ["missing.txt"]
    else:
        item["artifact_refs"] = [str(private)]
    with pytest.raises((ValueError, OSError)):
        validate_delivery(json.dumps(result), {"deliver_list": [{"id": "D1"}]}, workdir)
