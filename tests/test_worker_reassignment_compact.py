"""验证 Worker 换任务时的可配置压缩、持久化及失败处理。"""

import asyncio
from copy import deepcopy

import pytest
from test_agent_runtime import FixtureModel, assign, collect, finished, new_worker, root_session

from factor_demo.autoresearch_agent_v2.agent.history import (
    COMPACT_PROMPT,
    SUMMARY_PREFIX,
    SUMMARY_SUFFIX,
)
from factor_demo.autoresearch_agent_v2.config import HistoryConfig
from factor_demo.autoresearch_agent_v2.runtime import AgentRuntime, SessionFactory


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("restore", [False, True])
def test_reassignment_compacts_before_new_task_and_preserves_archive(tmp_path, enabled, restore):
    """首任务不压缩；复用任务先压缩全部旧历史，恢复会话后也采用相同策略。"""
    models = []
    summaries = []
    task_requests = []

    async def complete(messages):
        if messages[-1].get("content") == COMPACT_PROMPT:
            summaries.append(deepcopy(messages))
            return {
                "role": "assistant",
                "content": "Previous assignments are complete; reuse artifacts.",
            }
        task_requests.append(deepcopy(messages))
        return await finished(messages)

    def model_factory(parent, request):
        """恢复时创建独立模型，保留所有实际请求以检查前缀和工具策略。"""
        model = FixtureModel(complete)
        models.append(model)
        return model

    async def scenario():
        session = root_session(
            tmp_path,
            history_config=HistoryConfig(
                compact_on_worker_reassignment=enabled,
            ),
        )
        async with AgentRuntime(
            tmp_path / "runtime",
            SessionFactory(
                model_factory=model_factory,
            ),
        ) as runtime:
            runtime.attach(session)
            async with session:
                identity = await new_worker(session)
                await assign(session, identity, "first assignment")
                assert (await collect(session, identity))["result"]["outcome"] == "completed"
                assert len(task_requests) == 1 and not summaries
                original = runtime._sessions[identity].state.get_messages()
                session_id = runtime._sessions[identity].state.session_id
                if restore:
                    await runtime._discard_session(identity)
                for title in ("second assignment", "third assignment"):
                    before = runtime.restore_state(identity).get_messages()
                    await assign(session, identity, title)
                    assert (await collect(session, identity))["result"]["outcome"] == "completed"
                    child = runtime._sessions[identity]
                    assert child.state.session_id == session_id
                    assert child.state.messages[: len(original)] == original
                    request = task_requests[-1]
                    assert title in request[-1]["content"]
                    if enabled:
                        assert child.state.compaction.end_index == len(before)
                        assert request[1]["content"].startswith(SUMMARY_PREFIX)
                        assert request[1]["content"].endswith(SUMMARY_SUFFIX)
                        assert len(request) == 3
                        assert title not in str(summaries[-1][1:-1])
                        assert runtime.restore_state(identity).compaction == child.state.compaction
                    else:
                        assert child.state.compaction is None
                        assert "first assignment" in str(request)
                assert len(summaries) == (2 if enabled else 0)
                if enabled:
                    assert summaries[0][1:-1] == original
                    assert summaries[1][1]["content"].startswith(SUMMARY_PREFIX)
                    assert "first assignment" not in str(summaries[1][1:-1])
                    for model in models:
                        for messages, options in zip(
                            model.requests,
                            model.request_options,
                            strict=True,
                        ):
                            if messages[-1].get("content") == COMPACT_PROMPT:
                                assert options["tools"] == child.schemas()
                                assert options["tool_choice"] == "auto"

    asyncio.run(asyncio.wait_for(scenario(), 5))


def test_pending_compaction_does_not_block_router_or_other_worker_and_replay_is_idempotent(
    tmp_path,
):
    """压缩在后台等待时仍可查询和派发其他 Worker，重复请求不重复压缩。"""
    entered = asyncio.Event()
    release = asyncio.Event()
    summaries = []

    async def complete(messages):
        if messages[-1].get("content") == COMPACT_PROMPT:
            summaries.append(messages)
            entered.set()
            await release.wait()
            return {"role": "assistant", "content": "Earlier assignment is complete."}
        return await finished(messages)

    async def scenario():
        session = root_session(
            tmp_path,
            history_config=HistoryConfig(
                compact_on_worker_reassignment=True,
            ),
        )
        async with AgentRuntime(
            tmp_path / "runtime",
            SessionFactory(
                model_factory=lambda parent, request: FixtureModel(complete),
            ),
        ) as runtime:
            runtime.attach(session)
            async with session:
                reused = await new_worker(session)
                await assign(session, reused, "first task")
                await collect(session, reused)
                result = await assign(session, reused, "second task", call_id="reassign")
                await entered.wait()
                assert result == await assign(session, reused, "second task", call_id="reassign")
                assert len(summaries) == 1
                independent = await new_worker(session)
                await assign(session, independent, "independent task")
                assert (await collect(session, independent))["result"]["outcome"] == "completed"
                assert runtime.store.get("worker", reused)["status"] == "busy"
                release.set()
                assert (await collect(session, reused))["result"]["outcome"] == "completed"

    asyncio.run(asyncio.wait_for(scenario(), 5))


@pytest.mark.parametrize("failure", ["error", "empty", "tool", "cancel"])
def test_failed_reassignment_compaction_preserves_checkpoint_and_stops_new_task(tmp_path, failure):
    """压缩失败或取消后不执行新任务，也不覆盖最后一次成功的摘要。"""
    fail_compaction = False
    task_requests = []
    entered = asyncio.Event()

    async def complete(messages):
        if messages[-1].get("content") != COMPACT_PROMPT:
            task_requests.append(messages)
            return await finished(messages)
        if not fail_compaction:
            return {
                "role": "assistant",
                "content": "Previous work completed; artifacts are available.",
            }
        if failure == "error":
            raise RuntimeError("摘要服务失败")
        if failure == "cancel":
            entered.set()
            await asyncio.Event().wait()
        if failure == "tool":
            return await finished(messages)
        return {"role": "assistant", "content": " "}

    async def scenario():
        nonlocal fail_compaction
        session = root_session(
            tmp_path,
            history_config=HistoryConfig(
                compact_on_worker_reassignment=True,
            ),
        )
        async with AgentRuntime(
            tmp_path / "runtime",
            SessionFactory(
                model_factory=lambda parent, request: FixtureModel(complete),
            ),
        ) as runtime:
            runtime.attach(session)
            async with session:
                identity = await new_worker(session)
                for task in ("first assignment", "second assignment"):
                    await assign(session, identity, task)
                    assert (await collect(session, identity))["result"]["outcome"] == "completed"
                child = runtime._sessions[identity]
                before = child.state.get_messages()
                checkpoint = child.state.compaction
                fail_compaction = True
                await assign(session, identity, "must not start")
                if failure == "cancel":
                    await entered.wait()
                    await runtime.cancel_children("main")
                result = await collect(session, identity)
                expected = "cancelled" if failure == "cancel" else "failed"
                assert result["result"]["outcome"] == expected
                assert len(task_requests) == 2
                assert child.state.get_messages() == before
                assert child.state.compaction == checkpoint
                assert runtime.restore_state(identity).compaction == checkpoint

    asyncio.run(asyncio.wait_for(scenario(), 5))


def test_restored_worker_discards_retired_skills_without_parent_registry(tmp_path):
    """无技能父会话恢复旧 Worker 时清除预加载列表和正文快照。"""
    import json

    async def scenario():
        session = root_session(tmp_path)
        async with AgentRuntime(
            tmp_path / "runtime",
            SessionFactory(model_factory=lambda parent, request: FixtureModel(finished)),
        ) as runtime:
            runtime.attach(session)
            async with session:
                identity = await new_worker(session)
                await assign(session, identity, "first assignment")
                await collect(session, identity)
                await runtime._discard_session(identity)
                worker = runtime.store.get("worker", identity)
                checkpoint = json.loads(worker["checkpoint"])
                checkpoint["skill_snapshots"] = [{"retired": "旧流程不能重新加载"}]
                worker["checkpoint"] = json.dumps(checkpoint)
                worker["skills"] = ["research-executor"]
                runtime.store.put("worker", identity, worker)
                await assign(session, identity, "second assignment")
                result = await collect(session, identity)
                assert result["result"]["outcome"] == "completed"
                child = runtime._sessions[identity]
                assert child.skill_registry is None
                assert not child.initial_skills
                assert not child.state.skill_snapshots
                assert runtime.store.get("worker", identity)["skills"] == []
                assert runtime.restore_state(identity).skill_snapshots == []

    asyncio.run(asyncio.wait_for(scenario(), 5))
