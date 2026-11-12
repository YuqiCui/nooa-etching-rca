"""通过模拟模型响应验证基础循环和工具注册。"""

import asyncio
from copy import deepcopy

import litellm
import pytest

from factor_demo.autoresearch_agent_v2.agent.loop import AgentLoop
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.agent.prompts import SYSTEM_PROMPT
from factor_demo.autoresearch_agent_v2.agent.typing import SessionState
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.tools import ToolFactory, create_tool_factory


def tool_call(name, content, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": '{"content": "' + content + '"}'},
    }


def make_loop(tmp_path):
    return AgentLoop(
        Model(ModelConfig(model="openai/fixture-model", api_key="fixture-key")),
        SessionState(session_id="s1", agent_id="a1", task_id="t1", workdir=tmp_path),
    )


def test_tools_continue_and_finish_returns_then_stops(monkeypatch, tmp_path, capsys):
    requests = []
    responses = iter(
        [
            [tool_call("inspect", "Checking data", "r1")],
            [tool_call("inspect", "Found result", "r2"), tool_call("finish", "Done", "f1")],
        ]
    )

    async def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "Keep reasoning",
                        "tool_calls": next(responses),
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    loop = make_loop(tmp_path)
    loop.tools.register_callable("inspect", lambda content: content, description="检查数据")
    assert asyncio.run(loop.run("Start")) == "Done"
    assert capsys.readouterr().out == ""
    assert len(requests) == 2
    for request in requests:
        assert request["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert sum(message["role"] == "system" for message in request["messages"]) == 1
    assert loop.session.messages[0] == {"role": "user", "content": "Start"}
    assert requests[0]["tool_choice"] == "required"
    schemas = requests[0]["tools"]
    assert [item["function"]["name"] for item in schemas] == ["finish", "inspect"]
    assert schemas[0]["function"]["parameters"]["required"] == ["content"]
    replayed = requests[1]["messages"]
    assert replayed[2]["reasoning_content"] == "Keep reasoning"
    assert replayed[3] == {"role": "tool", "tool_call_id": "r1", "content": "Checking data"}
    assert loop.session.messages[-1] == {"role": "tool", "tool_call_id": "f1", "content": "Done"}


def test_finish_skips_later_calls_in_same_response(monkeypatch, tmp_path, capsys):
    async def complete(**kwargs):
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            tool_call("finish", "Done", "f1"),
                            tool_call("inspect", "Late", "r1"),
                        ],
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    assert asyncio.run(make_loop(tmp_path).run("Start")) == "Done"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "error_type", [None, ValueError, TypeError, KeyError, RuntimeError, OSError, TimeoutError]
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_tool_errors_return_results_and_continue_same_batch(
    monkeypatch, tmp_path, error_type, asynchronous
):
    requests = []
    replies = iter(
        [
            [
                tool_call("broken" if error_type else "notebook__write_cell_dummy", "x", "bad"),
                tool_call("inspect", "后续调用完成", "next"),
            ],
            [tool_call("finish", "Done", "done")],
        ]
    )

    async def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "tool_calls": next(replies)}}]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    loop = make_loop(tmp_path)
    executions = []

    def broken(content: str) -> str:
        executions.append(content)
        raise error_type("模拟工具执行失败")

    async def async_broken(content: str) -> str:
        await asyncio.sleep(0)
        return broken(content)

    loop.tools.register_callable(
        "broken", async_broken if asynchronous else broken, description="模拟执行错误"
    )
    loop.tools.register_callable("inspect", lambda content: content, description="继续执行")
    assert asyncio.run(loop.run("Start")) == "Done"
    assert len(requests) == 2
    returned = requests[1]["messages"][-2:]
    assert returned[0]["role"] == "tool"
    assert returned[0]["tool_call_id"] == "bad"
    assert (error_type.__name__ if error_type else "UnknownToolError") in returned[0]["content"]
    assert returned[1] == {"role": "tool", "tool_call_id": "next", "content": "后续调用完成"}
    assert executions == (["x"] if error_type else [])


@pytest.mark.parametrize("signal_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit])
def test_native_tool_control_signals_propagate_and_cleanup_runs(monkeypatch, tmp_path, signal_type):
    loop = make_loop(tmp_path)
    cleaned = []

    async def complete(messages, *, tools=None):
        return {"role": "assistant", "tool_calls": [tool_call("stop", "x", "stop")]}

    def stop(content: str) -> str:
        raise signal_type()

    async def after_run():
        cleaned.append(True)

    monkeypatch.setattr(loop.model, "completion", complete)
    loop.after_run_hook = after_run
    loop.tools.register_callable("stop", stop, description="模拟控制信号")
    with pytest.raises(signal_type):
        asyncio.run(loop.run("Start"))
    assert cleaned == [True]
    assert not any(message["role"] == "tool" for message in loop.session.messages)


@pytest.mark.parametrize("arguments,error_name", [("{", "JSONDecodeError"), ("{}", "TypeError")])
def test_invalid_finish_arguments_return_error_instead_of_ending_turn(
    monkeypatch, tmp_path, arguments, error_name
):
    """结束工具解析或执行失败时，模型仍有机会修正调用。"""
    loop = make_loop(tmp_path)
    failed_call = tool_call("finish", "ignored", "bad-finish")
    failed_call["function"]["arguments"] = arguments
    replies = iter(
        [
            {"role": "assistant", "tool_calls": [failed_call]},
            {"role": "assistant", "tool_calls": [tool_call("finish", "完成", "done")]},
        ]
    )
    requests = []

    async def complete(messages, *, tools=None):
        requests.append(messages)
        return next(replies)

    monkeypatch.setattr(loop.model, "completion", complete)
    assert asyncio.run(loop.run("Start")) == "完成"
    assert len(requests) == 2
    assert requests[1][-1]["tool_call_id"] == "bad-finish"
    assert error_name in requests[1][-1]["content"]


def test_registered_async_tool_is_executed_and_returned_to_model(monkeypatch, tmp_path):
    requests = []

    async def double(value):
        return value * 2

    async def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        calls = (
            [
                {
                    "id": "d1",
                    "type": "function",
                    "function": {"name": "double", "arguments": '{"value": 3}'},
                }
            ]
            if len(requests) == 1
            else [tool_call("finish", "Done", "f1")]
        )
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": calls,
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    loop = make_loop(tmp_path)
    loop.tools.register(
        "double",
        double,
        description="数值乘二",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    )
    assert asyncio.run(loop.run("Start")) == "Done"
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "d1",
        "content": "6",
    }
    assert requests[0]["tools"][-1]["function"]["name"] == "double"


def test_duplicate_registration_keeps_original_function():
    registry = ToolFactory()
    registry.register("value", lambda: 1, description="原始工具", parameters={})
    with pytest.raises(ValueError, match="already registered"):
        registry.register("value", lambda: 2, description="重复工具", parameters={})
    assert asyncio.run(registry.call("value", {})) == 1


@pytest.mark.parametrize("response_fields", [{}, {"tool_calls": None}, {"tool_calls": []}])
def test_response_without_tool_injects_reminder_and_continues(
    monkeypatch, tmp_path, capsys, response_fields
):
    requests = []
    responses = iter(
        [
            {"role": "assistant", "content": "Plain response", **response_fields},
            {"role": "assistant", "content": "Another response", **response_fields},
            {"role": "assistant", "tool_calls": [tool_call("finish", "Done", "f1")]},
        ]
    )

    async def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        return litellm.ModelResponse(choices=[{"message": next(responses)}])

    monkeypatch.setattr(litellm, "acompletion", complete)
    loop = make_loop(tmp_path)
    assert asyncio.run(loop.run("Start")) == "Done"
    assert capsys.readouterr().out == ""
    assert len(requests) == 3
    reminder = {
        "role": "user",
        "content": "请调用工具继续；结束本轮请调用 finish。",
    }
    for request, content in zip(requests[1:], ["Plain response", "Another response"]):
        assert request["messages"][-2]["role"] == "assistant"
        assert request["messages"][-2]["content"] == content
        assert request["messages"][-1] == reminder
    assert loop.session.messages[2] == reminder
    assert loop.session.messages[4] == reminder
    assert loop.session.messages[-1] == {"role": "tool", "tool_call_id": "f1", "content": "Done"}


def test_create_tool_factory_registers_builtins_and_keeps_instances_independent(capsys):
    factory = create_tool_factory()
    other = create_tool_factory()

    assert isinstance(factory, ToolFactory)
    assert [schema["function"]["name"] for schema in factory.schemas()] == ["finish"]
    with pytest.raises(KeyError):
        asyncio.run(factory.call("report", {"content": "进展"}))
    assert asyncio.run(factory.call("finish", {"content": "完成"})) == "完成"
    assert capsys.readouterr().out == ""

    factory.register("extra", lambda: 1, description="扩展工具", parameters={})
    assert [schema["function"]["name"] for schema in other.schemas()] == ["finish"]


@pytest.mark.parametrize("outcome", ["finish", "model_error", "cancel", "hook_error"])
def test_run_hooks_transform_content_and_always_run_on_exit(monkeypatch, tmp_path, outcome):
    events = []

    class HookedLoop(AgentLoop):
        async def before_run_hook(self, content):
            events.append(("before", content))
            assert self.session.messages == []
            if outcome == "hook_error":
                raise RuntimeError("hook failed")
            return "提示词：" + content

        async def after_run_hook(self):
            events.append(("after", None))

    async def complete(**kwargs):
        events.append(("model", kwargs["messages"][1]["content"]))
        if outcome == "model_error":
            raise RuntimeError("model failed")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [tool_call("finish", "Done", "c1")],
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    base = make_loop(tmp_path)
    loop = HookedLoop(base.model, base.session)

    if outcome == "finish":
        assert asyncio.run(loop.run("Start")) == "Done"
    else:
        error_type = asyncio.CancelledError if outcome == "cancel" else RuntimeError
        with pytest.raises(error_type):
            asyncio.run(loop.run("Start"))

    expected = [("before", "Start")]
    if outcome != "hook_error":
        expected.append(("model", "提示词：Start"))
        assert loop.session.messages[0] == {"role": "user", "content": "提示词：Start"}
    else:
        assert loop.session.messages == []
    assert events == [*expected, ("after", None)]


def test_custom_system_prompt_is_sent_to_model(monkeypatch, tmp_path):
    async def complete(**kwargs):
        assert kwargs["messages"] == [
            {"role": "system", "content": "自定义系统提示词"},
            {"role": "user", "content": "Start"},
        ]
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [tool_call("finish", "Done", "f1")],
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    base = make_loop(tmp_path)
    loop = AgentLoop(base.model, base.session, system_prompt="自定义系统提示词")
    assert asyncio.run(loop.run("Start")) == "Done"


def test_default_prompt_leaves_business_contracts_to_tasks_and_skills():
    """共用基础提示词不固定研究合同或专用验证角色。"""
    assert "skill" in SYSTEM_PROMPT
    for business_rule in ("report.md", "五折", "ResearchResult", "Verifier", "acceptance_criteria"):
        assert business_rule not in SYSTEM_PROMPT
    assert "finish(content)" in SYSTEM_PROMPT
    assert "report(content)" not in SYSTEM_PROMPT


@pytest.mark.parametrize("with_call", [False, True])
def test_body_json_never_executes_as_a_tool_call(monkeypatch, tmp_path, with_call):
    """正文伪调用不能触发工具，只有独立 tool_calls 字段可以执行。"""
    body = '{"content":[{"type":"tool_use","name":"inspect","input":{"content":"伪调用"}}]}'
    requests = []
    executed = []

    async def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        calls = (
            ([tool_call("inspect", "真实调用", "inspect-1")] if with_call else [])
            if len(requests) == 1
            else [tool_call("finish", "完成", "finish-1")]
        )
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": body,
                        "tool_calls": calls,
                    }
                }
            ]
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    loop = make_loop(tmp_path)
    loop.tools.register_callable(
        "inspect", lambda content: executed.append(content), description="记录调用"
    )
    assert asyncio.run(loop.run("检查")) == "完成"
    assert executed == (["真实调用"] if with_call else [])
    assert loop.session.messages[1]["content"] == body
    assert len(requests) == 2
