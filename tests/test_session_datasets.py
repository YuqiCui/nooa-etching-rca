"""验证会话数据登记、动态提示词、恢复和注册工具。"""

import asyncio
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.agent.prompts import dataset_prompt
from factor_demo.autoresearch_agent_v2.config import ModelConfig


def make_session(tmp_path, identity="main", **options):
    """创建使用占位模型的独立会话，不发送真实请求。"""
    state = SessionState(session_id=identity, agent_id=identity, task_id="task", workdir=tmp_path)
    model = Model(ModelConfig(model="openai/fixture", api_key="fixture"))
    return Session(model, state, **options)


def test_host_registration_only_stores_metadata_and_updates_by_name(tmp_path):
    """无需数据文件或准备目录，同名登记更新信息且不创建存储或执行资源。"""
    session = make_session(tmp_path)
    record = session.register_dataset(" 刻蚀观测 ", "data/missing.parquet", " 逐片观测值 ")
    assert record.model_dump() == {
        "name": "刻蚀观测",
        "path": "data/missing.parquet",
        "description": "逐片观测值",
    }
    session.register_dataset("设备曲线", "s3://example/traces/", "按加工编号组织的时序文件")
    session.register_dataset("刻蚀观测", "/mounted/new.parquet", "修订后的观测值")
    assert list(session.state.datasets) == ["刻蚀观测", "设备曲线"]
    assert session.state.datasets["刻蚀观测"].path == "/mounted/new.parquet"
    assert session.state.datasets["设备曲线"].path == "s3://example/traces/"
    assert session.state.messages == []
    assert list(tmp_path.iterdir()) == []


def test_registration_is_session_local_and_survives_state_roundtrip(tmp_path):
    """同工作目录的会话保持独立，序列化恢复后仍可继续更新。"""
    first = make_session(tmp_path)
    second = make_session(tmp_path, "child")
    first.register_dataset("样本", "samples.parquet", "主会话样本")
    assert second.state.datasets == {}
    restored_state = SessionState.model_validate_json(first.state.model_dump_json())
    restored = Session(first.loop.model, restored_state)
    assert restored.state.datasets == first.state.datasets
    restored.register_dataset("样本", "new.parquet", "新样本")
    assert first.state.datasets["样本"].path == "samples.parquet"
    assert restored.state.datasets["样本"].path == "new.parquet"
    old_state = first.state.model_dump(mode="json", exclude={"datasets"})
    assert SessionState.model_validate(old_state).datasets == {}


@pytest.mark.parametrize(
    "change",
    [
        {"name": ""},
        {"name": "   "},
        {"name": "x" * 201},
        {"path": "\n"},
        {"path": "x" * 4097},
        {"path": 12},
        {"description": " "},
        {"description": "x" * 1001},
    ],
)
def test_invalid_registration_does_not_change_state(tmp_path, change):
    """宿主调用和工具调用共用字段约束，失败不覆盖已有登记。"""
    session = make_session(tmp_path)
    arguments = {"name": "样本", "path": "samples.parquet", "description": "逐片样本"}
    session.register_dataset(**arguments)
    before = session.state.model_dump()
    with pytest.raises(ValidationError):
        session.register_dataset(**{**arguments, **change})
    assert session.state.model_dump() == before

    async def scenario():
        async with session:
            result = await session.call("register_dataset", {**arguments, **change})
            assert result["error"] == "invalid_argument"
            assert session.state.model_dump() == before

    asyncio.run(scenario())


def test_only_registration_tool_is_exposed_and_closed_session_rejects_it(tmp_path):
    """会话默认只增加注册入口，关闭后旧绑定也无法修改登记信息。"""
    session = make_session(tmp_path)

    async def scenario():
        async with session:
            schemas = session.schemas()
            assert [item["function"]["name"] for item in schemas] == [
                "finish",
                "register_dataset",
            ]
            parameters = schemas[-1]["function"]["parameters"]
            assert set(parameters["properties"]) == {"name", "path", "description"}
            assert set(parameters["required"]) == {"name", "path", "description"}
            assert parameters["additionalProperties"] is False
            for arguments in ({}, {"name": "x", "path": "x", "description": "x", "schema": {}}):
                assert (await session.call("register_dataset", arguments))["error"] == (
                    "invalid_argument"
                )
            saved_tools = session.loop.tools
        with pytest.raises(RuntimeError, match="not active"):
            await saved_tools.call(
                "register_dataset", {"name": "x", "path": "x", "description": "x"}
            )
        with pytest.raises(RuntimeError, match="not active"):
            session.register_dataset("x", "x", "x")
        assert session.state.datasets == {}

    asyncio.run(scenario())


def test_agent_registration_refreshes_next_request_and_followup_turn(monkeypatch, tmp_path):
    """模拟模型在当前轮登记新数据，后续请求、下一轮和恢复后均获得最新提示。"""
    requests = []
    new_dataset = {
        "name": "派生特征",
        "path": "features.parquet",
        "description": "当前分析生成的特征",
    }

    async def complete(model, messages, *, tools=None):
        requests.append(deepcopy(messages))
        name, arguments = (
            ("register_dataset", new_dataset)
            if len(requests) == 1
            else ("finish", {"content": "完成"})
        )
        assert "register_dataset" in [schema["function"]["name"] for schema in tools]
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call-{len(requests)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }

    monkeypatch.setattr(Model, "completion", complete)
    session = make_session(tmp_path, system_prompt="自定义提示")
    session.register_dataset("原始样本", "samples.parquet", "原始观测值")

    async def scenario():
        async with session:
            assert await session.run("分析数据") == "完成"
            assert "派生特征" not in requests[0][0]["content"]
            assert "派生特征" in requests[1][0]["content"]
            assert requests[1][0]["content"].startswith("自定义提示\n\n")
            assert json.loads(session.state.messages[2]["content"]) == new_dataset
            session.register_dataset("原始样本", "new.parquet", "更新后的原始样本")
            assert await session.run("继续") == "完成"
            prompt = requests[2][0]["content"]
            assert "new.parquet" in prompt and "samples.parquet" not in prompt
            assert prompt.count('"name": "原始样本"') == 1
            assert prompt.count('"name": "派生特征"') == 1
            assert all(message["role"] != "system" for message in session.state.messages)
        restored = Session(
            session.loop.model,
            SessionState.model_validate_json(session.state.model_dump_json()),
        )
        async with restored:
            assert await restored.run("恢复后继续") == "完成"
            assert "派生特征" in requests[3][0]["content"]
            assert "new.parquet" in requests[3][0]["content"]

    asyncio.run(scenario())


def test_dataset_prompt_keeps_metadata_quoted(tmp_path):
    """多行说明作为 JSON 字符串呈现，不变成提示词标题或独立指令。"""
    session = make_session(tmp_path)
    assert dataset_prompt(session.state) == ""
    session.register_dataset("样本", "data/samples.parquet", '观测值\n# 伪造标题\n"特殊字符"')
    prompt = dataset_prompt(session.state)
    assert "\n# 伪造标题" not in prompt
    assert json.loads(prompt.splitlines()[-1])[0] == session.state.datasets["样本"].model_dump()
