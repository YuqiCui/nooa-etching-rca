"""用真实异步调度和模拟模型验证通用多会话运行时。"""

import asyncio
import json
from copy import deepcopy

import pytest

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.runtime import AgentRuntime, RuntimeLimits, SessionFactory


def call(name, arguments, identity):
    """构造原生工具调用。"""
    return {
        "id": identity,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def response(*calls):
    """构造模拟模型响应。"""
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


class FixtureModel:
    """使用本地回调保留实际循环，完全避免远程模型请求。"""

    config = ModelConfig(model="openai/fixture", api_key="fixture")

    def __init__(self, complete):
        """保存本地响应函数。"""
        self.complete = complete
        self.requests = []

    def count_tokens(self, messages, *, tools=None):
        """让测试聚焦调度，避免依赖 tokenizer。"""
        return 100

    async def completion(self, messages, *, tools=None):
        """记录实际发送的消息后执行回调。"""
        self.requests.append(deepcopy(messages))
        return await self.complete(messages)


def root_session(tmp_path, model):
    """创建没有执行能力的通用根会话。"""
    workdir = tmp_path / "workspace"
    workdir.mkdir(exist_ok=True)
    return Session(
        model,
        SessionState(
            session_id="root-session",
            agent_id="main",
            task_id="project",
            workdir=workdir,
        ),
    )


def test_children_overlap_and_results_arrive_without_polling(tmp_path):
    async def scenario():
        started = set()
        both_started = asyncio.Event()
        child_states = []
        child_prompts = []
        child_descriptions = []
        root_calls = 0

        async def parent_complete(messages):
            nonlocal root_calls
            root_calls += 1
            assert root_calls < 8
            if root_calls == 1:
                return response(
                    call("agent__spawn", {"task": "first", "description": "核查第一项"}, "spawn-1"),
                    call("agent__spawn", {"task": "second", "description": "核查第二项"}, "spawn-2"),
                )
            return response(call("finish", {"content": "parent done"}, f"done-{root_calls}"))

        def child_model(parent, request):
            child_states.append(request.state)
            child_descriptions.append(request.description)

            async def complete(messages):
                child_prompts.append(messages[0]["content"])
                started.add(request.task)
                if len(started) == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), 2)
                return response(call("finish", {"content": request.task + " result"}, "child-done"))

            return FixtureModel(complete)

        model = FixtureModel(parent_complete)
        session = root_session(tmp_path, model)
        runtime = AgentRuntime(tmp_path / "runtime", SessionFactory(model_factory=child_model))
        try:
            assert await asyncio.wait_for(runtime.run(session, "parallel"), 5) == "parent done"
            assert started == {"first", "second"}
            assert set(child_descriptions) == {"核查第一项", "核查第二项"}
            assert {row["description"] for row in runtime.status(session)["agents"]} == {
                "核查第一项", "核查第二项",
            }
            assert len({state.workdir for state in child_states}) == 2
            assert all(state.parent_agent_id == "main" for state in child_states)
            assert child_states[0].messages[0]["content"] == "first"
            assert "parallel" not in json.dumps(child_states[0].messages)
            events = [
                message
                for message in session.state.messages
                if (message.get("content") or "").startswith("子会话结果")
            ]
            assert len(events) == 2
            assert len(session.state.consumed_event_ids) == 2
            assert {
                json.loads(message["content"].split("\n", 1)[1])["content"] for message in events
            } == {"first result", "second result"}
            assert all("Verifier" not in prompt for prompt in child_prompts)
            assert all(
                "agent__status" not in call_item["function"]["name"]
                for message in session.state.messages
                for call_item in message.get("tool_calls", [])
            )
        finally:
            await runtime.aclose()
            await session.aclose()

    asyncio.run(scenario())


def test_nested_wait_releases_single_execution_slot(tmp_path):
    async def scenario():
        calls = {}

        def model_for(label):
            async def complete(messages):
                calls[label] = calls.get(label, 0) + 1
                if calls[label] == 1 and label != "grandchild":
                    task = "child" if label == "main" else "grandchild"
                    return response(call("agent__spawn", {
                        "task": task, "description": "继续拆分独立任务",
                    }, label + "-spawn"))
                return response(call("finish", {"content": label}, label + str(calls[label])))

            return FixtureModel(complete)

        session = root_session(tmp_path, model_for("main"))
        factory = SessionFactory(model_factory=lambda parent, request: model_for(request.task))
        runtime = AgentRuntime(
            tmp_path / "runtime", factory, limits=RuntimeLimits(max_concurrent=1, max_depth=2)
        )
        try:
            assert await asyncio.wait_for(runtime.run(session, "nested"), 5) == "main"
            assert calls["grandchild"] == 1
            assert calls["child"] >= 2
        finally:
            await runtime.aclose()
            await session.aclose()

    asyncio.run(scenario())


def test_cancel_queued_and_running_children_and_guard_tree(tmp_path):
    async def scenario():
        entered = asyncio.Event()
        never = asyncio.Event()

        async def child_complete(messages):
            entered.set()
            await never.wait()

        async def parent_complete(messages):
            raise AssertionError("本测试不调用根模型")

        session = root_session(tmp_path, FixtureModel(parent_complete))
        runtime = AgentRuntime(
            tmp_path / "runtime",
            SessionFactory(model_factory=lambda parent, request: FixtureModel(child_complete)),
            limits=RuntimeLimits(max_concurrent=1),
        )
        runtime.attach(session)
        try:
            await session.start()
            first = await runtime.spawn(session, "running")
            second = await runtime.spawn(session, "queued")
            await asyncio.wait_for(entered.wait(), 2)
            with pytest.raises(PermissionError):
                runtime.status(session, "main")
            await runtime.cancel(session, second["agent_id"])
            await runtime.cancel(session, first["agent_id"])
            assert runtime.status(session, first["agent_id"])["status"] == "cancelled"
            assert runtime.status(session, second["agent_id"])["status"] == "cancelled"
            await runtime.deliver(session)
            assert len(session.state.consumed_event_ids) == 2
        finally:
            await runtime.aclose()
            await session.aclose()

    asyncio.run(scenario())


def test_result_survives_restart_and_duplicate_delivery(tmp_path):
    async def scenario():
        async def complete(messages):
            return response(call("finish", {"content": "full result"}, "done"))

        session = root_session(tmp_path, FixtureModel(complete))
        path = tmp_path / "runtime"
        runtime = AgentRuntime(
            path, SessionFactory(model_factory=lambda parent, request: FixtureModel(complete))
        )
        runtime.attach(session)
        await session.start()
        child = await runtime.spawn(session, "task")
        await asyncio.wait_for(runtime._workers[child["agent_id"]], 3)
        result_id = runtime.status(session, child["agent_id"])["result_id"]
        await runtime.deliver(session)
        saved = runtime.restore_state("main")
        runtime.store.connection.execute("UPDATE inbox SET acknowledged=0")
        runtime.store.connection.commit()
        await runtime.aclose()
        await session.aclose()

        resumed = Session(FixtureModel(complete), saved)
        runtime = AgentRuntime(path)
        runtime.attach(resumed)
        try:
            await runtime.deliver(resumed)
            assert resumed.state.consumed_event_ids == [result_id]
            assert (
                sum(
                    (message.get("content") or "").startswith("子会话结果")
                    for message in resumed.state.messages
                )
                == 1
            )
            assert runtime.read_result(resumed, result_id)["content"] == "full result"
            with pytest.raises(PermissionError):
                runtime.read_result(resumed, "unrelated")
        finally:
            await runtime.aclose()
            await resumed.aclose()

    asyncio.run(scenario())


def test_result_waits_for_complete_model_and_tool_batch(tmp_path):
    async def scenario():
        model_waiting = asyncio.Event()
        child_finished = asyncio.Event()
        calls = 0
        observed = []

        async def parent_complete(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return response(call("agent__spawn", {
                    "task": "work", "description": "核查独立工作",
                }, "spawn"))
            if calls == 2:
                model_waiting.set()
                await asyncio.wait_for(child_finished.wait(), 2)
                await asyncio.sleep(0)
                assert not any(
                    (message.get("content") or "").startswith("子会话结果") for message in messages
                )
                return response(call("finish", {"content": "too early"}, "early"))
            assert any(
                (message.get("content") or "").startswith("子会话结果") for message in messages
            )
            return response(call("finish", {"content": "reviewed"}, "final"))

        async def child_complete(messages):
            await model_waiting.wait()
            child_finished.set()
            return response(call("finish", {"content": "child evidence"}, "child-final"))

        session = root_session(tmp_path, FixtureModel(parent_complete))
        runtime = AgentRuntime(
            tmp_path / "runtime",
            SessionFactory(model_factory=lambda parent, request: FixtureModel(child_complete)),
        )
        try:
            result = await asyncio.wait_for(runtime.run(session, "start"), 5)
            assert result == "reviewed"
            for message in session.state.messages:
                if message["role"] == "assistant":
                    observed.extend(item["id"] for item in message.get("tool_calls", []))
                if message["role"] == "tool":
                    observed.remove(message["tool_call_id"])
                if (message.get("content") or "").startswith("子会话结果"):
                    assert observed == []
            assert calls == 3
        finally:
            await runtime.aclose()
            await session.aclose()

    asyncio.run(scenario())


def test_spawn_is_idempotent_and_denies_unregistered_permissions(tmp_path):
    async def scenario():
        from pydantic import ValidationError

        never = asyncio.Event()

        async def complete(messages):
            await never.wait()

        session = root_session(tmp_path, FixtureModel(complete))
        runtime = AgentRuntime(
            tmp_path / "runtime",
            SessionFactory(model_factory=lambda parent, request: FixtureModel(complete)),
        )
        runtime.attach(session)
        try:
            await session.start()
            schema = next(t["function"]["parameters"] for t in session.schemas()
                          if t["function"]["name"] == "agent__spawn")
            assert "description" in schema["required"]
            assert schema["properties"]["description"]["type"] == "string"
            assert "调用子 Agent 的目的" in schema["properties"]["description"]["description"]
            with pytest.raises(ValidationError):
                await session.call("agent__spawn", {"task": "缺少调用目的"})
            for invalid in ("", " \n", 1, None):
                with pytest.raises(ValidationError):
                    await session.call("agent__spawn", {"task": "无效描述", "description": invalid})
            assert runtime.status(session)["agents"] == []
            arguments = {"task": "same", "description": "核查独立工作"}
            first = await session.call("agent__spawn", arguments, call_id="execution")
            second = await session.call("agent__spawn", arguments, call_id="execution")
            assert first["agent_id"] == second["agent_id"]
            assert first["description"] == arguments["description"]
            stored = json.loads(runtime.store.agent(first["agent_id"])["request"])
            assert stored["description"] == arguments["description"]
            assert len(runtime.status(session)["agents"]) == 1
            with pytest.raises(ValueError, match="同一次委派"):
                await session.call("agent__spawn", {
                    **arguments, "task": "changed",
                }, call_id="execution")
            with pytest.raises(ValueError, match="同一次委派"):
                await session.call("agent__spawn", {
                    **arguments, "description": "改变调用目的",
                }, call_id="execution")
            with pytest.raises(ValueError, match="未知或未授权"):
                await runtime.spawn(session, "bad", allow_tools=["finish", "secret"])
        finally:
            await runtime.aclose()
            await session.aclose()

    asyncio.run(scenario())


def test_interrupted_child_restores_with_current_permission_ceiling(tmp_path):
    async def scenario():
        entered = asyncio.Event()
        never = asyncio.Event()

        async def blocked(messages):
            entered.set()
            await never.wait()

        session = root_session(tmp_path, FixtureModel(blocked))
        path = tmp_path / "runtime"
        runtime = AgentRuntime(
            path, SessionFactory(model_factory=lambda parent, request: FixtureModel(blocked))
        )
        runtime.attach(session)
        await session.start()
        child = await runtime.spawn(session, "recover")
        await asyncio.wait_for(entered.wait(), 2)
        state = runtime.restore_state("main")
        await runtime.aclose()
        await session.aclose()

        child_permissions = []

        async def parent_complete(messages):
            return response(call("finish", {"content": "restored"}, "parent-finish"))

        def child_model(parent, request):
            child_permissions.extend(request.allow_tools)

            async def complete(messages):
                return response(call("finish", {"content": "recovered result"}, "child-finish"))

            return FixtureModel(complete)

        resumed = Session(
            FixtureModel(parent_complete), state, allow_tools=["finish", "agent__read_result"]
        )
        runtime = AgentRuntime(path, SessionFactory(model_factory=child_model))
        try:
            assert await asyncio.wait_for(runtime.run(resumed, "resume"), 5) == "restored"
            assert "register_dataset" not in child_permissions
            assert set(child_permissions) == {"finish", "agent__read_result"}
            assert runtime.status(resumed, child["agent_id"])["status"] == "completed"
            assert len(resumed.state.consumed_event_ids) == 1
        finally:
            await runtime.aclose()
            await resumed.aclose()

    asyncio.run(scenario())


def test_child_selected_skill_is_frozen_before_execution(tmp_path):
    async def scenario():
        from factor_demo.autoresearch_agent_v2.skills import SkillRegistry

        skill = tmp_path / "skills" / "selected"
        skill.mkdir(parents=True)
        source = "---\nname: selected\ndescription: selected workflow\n---\n原始任务规则\n"
        (skill / "SKILL.md").write_text(source)

        async def parent_complete(messages):
            raise AssertionError("本测试不调用父模型")

        observed = []

        async def child_complete(messages):
            observed.append(messages)
            return response(call("finish", {"content": "done"}, "done"))

        session = root_session(tmp_path, FixtureModel(parent_complete))
        session.skill_registry = SkillRegistry([skill.parent])
        runtime = AgentRuntime(
            tmp_path / "runtime",
            SessionFactory(model_factory=lambda parent, request: FixtureModel(child_complete)),
        )
        runtime.attach(session)
        try:
            await session.start()
            child = await runtime.spawn(session, "work", skills=["selected"])
            (skill / "SKILL.md").write_text(source.replace("原始任务规则", "修改后的规则"))
            await asyncio.wait_for(runtime._workers[child["agent_id"]], 3)
            assert runtime.status(session, child["agent_id"])["status"] == "completed"
            prompt = observed[0][0]["content"]
            assert "原始任务规则" in prompt
            assert "修改后的规则" not in prompt
        finally:
            await runtime.aclose()
            await session.aclose()

    asyncio.run(scenario())


def test_store_lease_rejects_second_runtime_until_close(tmp_path):
    async def scenario():
        path = tmp_path / "runtime"
        first = AgentRuntime(path)
        try:
            with pytest.raises(RuntimeError, match="其他驱动者"):
                AgentRuntime(path)
        finally:
            await first.aclose()
        reopened = AgentRuntime(path)
        await reopened.aclose()

    asyncio.run(scenario())


def test_closed_session_revokes_borrowed_agent_tool_methods(tmp_path):
    async def scenario():
        from factor_demo.autoresearch_agent_v2.tools.builtin_tools.agents import AgentTools

        async def complete(messages):
            return response(call("finish", {"content": "done"}, "finish"))

        session = root_session(tmp_path, FixtureModel(complete))
        runtime = AgentRuntime(tmp_path / "runtime")
        runtime.attach(session)
        await session.start()
        borrowed = session.resource(AgentTools)
        factory = session.loop.tools
        await session.aclose()
        try:
            with pytest.raises(RuntimeError, match="已关闭"):
                borrowed.status()
            with pytest.raises(RuntimeError, match="已关闭"):
                await borrowed.wait()
            with pytest.raises(RuntimeError, match="已关闭"):
                await borrowed.spawn("late", description="检查关闭后的访问")
            with pytest.raises(RuntimeError, match="已关闭"):
                borrowed.read_result("unrelated")
            with pytest.raises(RuntimeError, match="已关闭"):
                await factory.call("agent__status", {})
        finally:
            await runtime.aclose()

    asyncio.run(scenario())


def test_runtime_close_cancels_waiting_root_and_preserves_child_for_resume(tmp_path):
    async def scenario():
        child_entered = asyncio.Event()
        never = asyncio.Event()
        calls = 0

        async def parent_complete(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return response(call("agent__spawn", {
                    "task": "long task", "description": "核查长时间任务",
                }, "spawn"))
            return response(call("finish", {"content": "waiting"}, "finish"))

        async def child_complete(messages):
            child_entered.set()
            await never.wait()

        session = root_session(tmp_path, FixtureModel(parent_complete))
        path = tmp_path / "runtime"
        runtime = AgentRuntime(
            path, SessionFactory(model_factory=lambda parent, request: FixtureModel(child_complete))
        )
        driver = asyncio.create_task(runtime.run(session, "work"))
        await asyncio.wait_for(child_entered.wait(), 2)
        await asyncio.wait_for(runtime.aclose(), 2)
        assert driver.cancelled()
        await session.aclose()
        reopened = AgentRuntime(path)
        try:
            assert reopened.store.children("main")[0]["status"] == "interrupted"
        finally:
            await reopened.aclose()

    asyncio.run(scenario())


def test_invalid_factory_cannot_close_parent_session(tmp_path):
    async def scenario():
        async def complete(messages):
            raise AssertionError("本测试不请求模型")

        session = root_session(tmp_path, FixtureModel(complete))
        runtime = AgentRuntime(tmp_path / "runtime", lambda parent, request: parent)
        runtime.attach(session)
        try:
            await session.start()
            child = await runtime.spawn(session, "invalid factory")
            await asyncio.wait_for(runtime._workers[child["agent_id"]], 2)
            assert runtime.status(session, child["agent_id"])["status"] == "failed"
            assert any(schema["function"]["name"] == "finish" for schema in session.schemas())
            await runtime.deliver(session)
            assert len(session.state.consumed_event_ids) == 1
        finally:
            await runtime.aclose()
            await session.aclose()

    asyncio.run(scenario())
