"""验证默认子会话工厂兼容独立内置工具并保持文件隔离。"""

import asyncio

from test_agent_runtime import FixtureModel, call, response, root_session

from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.runtime import AgentRuntime
from factor_demo.autoresearch_agent_v2.tools import FileTools


def test_default_factory_copies_manual_builtin_without_parent_history(tmp_path, monkeypatch):
    """旧式手动登记可用于子会话，但历史与写目录不随定义复制。"""
    child_requests = []

    async def child_completion(self, messages, *, tools=None):
        child_requests.append(messages)
        names = {item["function"]["name"] for item in tools}
        assert names == {"finish", "write_file"}
        if not any(message["role"] == "tool" for message in messages):
            return response(call("write_file", {"path": "result.txt", "content": "独立结果"}, "w"))
        return response(call("finish", {"content": "已保存"}, "f"))

    monkeypatch.setattr(Model, "completion", child_completion)

    async def scenario():
        calls = 0

        async def parent_completion(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return response(call("agent__spawn", {
                    "task": "子任务正文", "description": "独立保存研究结果",
                    "allow_tools": ["finish", "write_file"],
                }, "spawn"))
            return response(call("finish", {"content": "主任务结果"}, f"root-{calls}"))

        parent = root_session(tmp_path, FixtureModel(parent_completion))
        parent.register_stateful_tool(FileTools)
        async with AgentRuntime(tmp_path / "runtime") as runtime:
            try:
                assert await runtime.run(parent, "父任务私有说明") == "主任务结果"
                child = runtime.store.children("main")[0]
                child_state = runtime.restore_state(child["agent_id"])
                assert (child_state.workdir / "result.txt").read_text() == "独立结果"
                assert not (parent.state.workdir / "result.txt").exists()
            finally:
                await parent.aclose()
        initial_input = child_requests[0][-1]
        assert initial_input == {"role": "user", "content": "子任务正文"}
        assert all("父任务私有说明" not in str(message) for message in child_requests[0])

    asyncio.run(scenario())
