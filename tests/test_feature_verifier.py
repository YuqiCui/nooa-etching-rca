"""核对历史评估程序以及持续验证会话、非阻塞交接和故障恢复。"""

import asyncio
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.feature_verifier import FeatureAnalyzeVerifier
from factor_demo.autoresearch_agent_v2.feature_verifier.agent import VerifierModel
from factor_demo.autoresearch_agent_v2.feature_verifier.evaluation import evaluate, save_results
from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox
from factor_demo.autoresearch_agent_v2.tools import NotebookTools


def candidate_table():
    """生成按独立组划分的强信号，仅用于验证协议。"""
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(40), 2)
    target = groups % 2
    return pd.DataFrame({
        "sample": np.arange(len(groups)), "lot": groups, "label": target,
        "split": np.where(groups < 30, "train", "dev"),
        "sensor_mean": target * 10 + rng.normal(0, 0.2, len(groups)),
    })


def feature_spec(**changes):
    """提供固定候选集和真实分组字段。"""
    return {"features": ["sensor_mean"], "target": "label", "group": "lot",
            "split": "split", "sample_id": "sample",
            "selection_scope": "train_discovered", "dev_usage": "untouched", **changes}


def test_grouped_five_fold_scores_and_standalone_artifacts(tmp_path):
    result, metrics, predictions = evaluate(candidate_table(), feature_spec())
    assert result["status"] == "accepted"
    assert len(metrics[metrics.partition.eq("cv")]) == 15
    assert len(metrics[metrics.partition.eq("dev")]) == 3
    for _, rows in predictions[predictions.partition.eq("cv")].groupby("model"):
        assert not rows.sample_id.duplicated().any()
        assert rows.groupby("group").fold.nunique().max() == 1
    save_results(result, metrics, predictions, tmp_path)
    for name in ("validation.json", "fold_metrics.csv", "oof_predictions.csv",
                 "cv_scores.png", "confusion_matrix.png"):
        assert (tmp_path / name).stat().st_size > 0


@pytest.mark.parametrize("problem", ["group_overlap", "test_rows", "target_feature",
                                      "too_few_groups", "duplicate_sample"])
def test_leakage_and_invalid_folds_are_rejected(problem):
    table, spec = candidate_table(), feature_spec()
    if problem == "group_overlap":
        table.loc[table.split.eq("dev"), "lot"] = 0
    elif problem == "test_rows":
        table.loc[0, "split"] = "test"
    elif problem == "target_feature":
        spec["features"] = ["label"]
    elif problem == "too_few_groups":
        table = table[table.lot.lt(8)]
    else:
        table.loc[1, "sample"] = table.loc[0, "sample"]
    with pytest.raises(ValueError):
        evaluate(table, spec)


def test_discovered_features_without_independent_dev_are_inconclusive():
    table = candidate_table()
    table = table[table.split.eq("train")]
    result, _, _ = evaluate(table, feature_spec())
    assert result["status"] == "inconclusive"


def test_constant_features_fail_against_dummy():
    table = candidate_table()
    table["sensor_mean"] = 1.0
    result, _, _ = evaluate(table, feature_spec())
    assert result["status"] == "rejected"


def test_dev_already_used_in_research_cannot_certify_discovered_features():
    result, _, _ = evaluate(candidate_table(), feature_spec(dev_usage="used_in_research"))
    assert result["status"] == "inconclusive"


def tool_call(name, arguments, identity):
    """构造模拟模型工具调用；统计和沙箱仍使用实际实现。"""
    return {"id": identity, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


@pytest.mark.parametrize("effort", ["medium", "high"])
def test_persistent_verifier_nonblocking_submission_and_report(tmp_path, monkeypatch, effort):
    """模拟模型验证连续上下文、排队隔离、反馈注入及原轮修订。"""
    requests = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def complete(self, messages, *, tools=None):
        requests.append(json.loads(json.dumps(messages)))
        names = {item["function"]["name"] for item in tools}
        assert "report" in names and "finish" not in names
        assert "validate_features" not in names
        assert "研究端秘密历史" not in str(messages)
        assert self.config.reasoning_effort == effort
        entered.set()
        await release.wait()
        return {"role": "assistant", "tool_calls": [tool_call(
            "report", {"status": "regressed", "feedback": "持平；sensor_mean 重要性 1；证据 tmp/cv.csv"},
            f"report-{len(requests)}")]}

    monkeypatch.setattr(VerifierModel, "completion", complete)

    async def scenario():
        model = Model(ModelConfig(model="openai/fixture", api_key="fixture", reasoning_effort=effort))
        state = SessionState(session_id="research", agent_id="research", task_id="t",
                             workdir=tmp_path,
                             messages=[{"role": "user", "content": "研究端秘密历史"}])
        session = Session(model, state)
        session.register_stateful_tool(NotebookTools)
        session.register_stateful_tool(FeatureAnalyzeVerifier)
        async with session:
            notebooks = session.resource(NotebookTools)
            verifier = session.resource(FeatureAnalyzeVerifier)
            notebooks._kernels["round1"] = None
            notebooks._save_document("round1", nbformat.v4.new_notebook(), None)
            try:
                first = await session.call("feature_analyze_verifier", {
                    "notebook_name": "round1", "report": "第一版", "round_id": "r1"})
                assert first["status"] == "submitted"
                await asyncio.wait_for(entered.wait(), 5)
                assert not verifier.task.done()
                # 父会话继续操作，新提交入队，但不提前进入 verifier 消息。
                result = await session.call("register_dataset", {
                    "name": "continue", "path": "input.csv", "description": "继续研究"})
                assert result["name"] == "continue"
                queued = await session.call("feature_analyze_verifier", {
                    "notebook_name": "round1", "report": "排队研究", "round_id": "r2"})
                assert queued["status"] == "submitted"
                assert (await verifier.status())["queue_length"] == 1
                assert len(requests) == 1
                assert queued["verification_path"] not in str(verifier.child.state.messages)
                assert verifier.current["submission_id"] == first["submission_id"]
                assert len(list((tmp_path / "verifier_runs").glob("*/submission.json"))) == 2
                release.set()
                status = await verifier.status(wait=True)
                assert status["status"] == "idle"
                assert status["latest_report"]["accepted"] is True
                identity = verifier.child.state.session_id
                assert verifier.child.resource(Sandbox).config.read_paths == ()
                await session.loop.before_completion_hook()
                assert "sensor_mean" in state.messages[-1]["content"]
                second = await session.call("feature_analyze_verifier", {
                    "notebook_name": "round1", "report": "原轮修订", "round_id": "r1"})
                assert second["submission_id"] != first["submission_id"]
                await verifier.status(wait=True)
                assert verifier.child.state.session_id == identity
                assert len(requests) == 3
                assert "第一版" not in str(requests[-1])  # 报告通过冻结文件传递。
                assert first["verification_path"] in str(requests[-1])
                assert "sensor_mean" in str(requests[-1])
                assert second["verification_path"] in str(requests[-1])
            finally:
                notebooks._kernels.clear()
        # 重新绑定仍恢复同一个 verifier 会话及历史。
        new_session = Session(model, state)
        new_session.register_stateful_tool(NotebookTools)
        new_session.register_stateful_tool(FeatureAnalyzeVerifier)
        async with new_session:
            verifier = new_session.resource(FeatureAnalyzeVerifier)
            await verifier._ensure_child()
            assert verifier.child.state.session_id == identity
            assert "sensor_mean" in str(verifier.child.state.messages)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["error", "cancelled", "invalid_report"])
def test_verifier_failure_preserves_queued_submissions(tmp_path, monkeypatch, mode):
    """执行故障、取消和无效状态不能冒充 report 或投递排队提交。"""
    async def complete(self, messages, *, tools=None):
        if mode == "cancelled":
            raise asyncio.CancelledError
        raise RuntimeError("验证未完成")

    monkeypatch.setattr(VerifierModel, "completion", complete)

    async def scenario():
        model = Model(ModelConfig(model="openai/fixture", api_key="fixture"))
        session = Session(model, SessionState(session_id="r", agent_id="r", task_id="t",
                                              workdir=tmp_path))
        session.register_stateful_tool(NotebookTools)
        session.register_stateful_tool(FeatureAnalyzeVerifier)
        async with session:
            notebooks = session.resource(NotebookTools)
            notebooks._kernels["r1"] = None
            notebooks._save_document("r1", nbformat.v4.new_notebook(), None)
            verifier = session.resource(FeatureAnalyzeVerifier)
            try:
                await verifier.submit("r1", "研究")
                if mode == "cancelled":
                    with pytest.raises(asyncio.CancelledError):
                        await verifier.status(wait=True)
                else:
                    await verifier.status(wait=True)
                queued = await verifier.submit("r1", "新成果", round_id="r2")
                assert queued["status"] == "submitted"
                assert (await verifier.status())["queue_length"] == 1
                assert not list((tmp_path / "verifier_runs").glob("*/result.json"))
                if mode == "invalid_report":
                    with pytest.raises(ValidationError):
                        await verifier.child.loop.tools.call("report", {
                            "status": "accepted", "feedback": "错误状态"})
                # 恢复原提交，真实 report 后才重新开放。
                async def recovered(self, messages, *, tools=None):
                    return {"role": "assistant", "tool_calls": [tool_call(
                        "report", {"status": "blocked", "feedback": "缺少目录内数据"}, "recovered")]}
                monkeypatch.setattr(VerifierModel, "completion", recovered)
                status = await verifier.status(retry=True, wait=True)
                assert status["status"] == "idle"
                assert status["latest_report"]["status"] == "blocked"
                assert status["latest_report"]["submission_id"] == queued["submission_id"]
                assert status["queue_length"] == 0
                assert len(list((tmp_path / "verifier_runs").glob("*/result.json"))) == 2
            finally:
                notebooks._kernels.clear()

    asyncio.run(scenario())


def test_runtime_medium_is_supported_and_default():
    from factor_demo.runtime import RuntimeSettings

    assert RuntimeSettings().reasoning_effort == "medium"
    assert RuntimeSettings.load(Path("configs/runtime.json")).reasoning_effort == "medium"


def test_close_and_resume_unreported_submission(tmp_path, monkeypatch):
    """关闭时取消后台验证，恢复后原提交仍占用交接位置。"""
    entered = asyncio.Event()

    async def blocked(self, messages, *, tools=None):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(VerifierModel, "completion", blocked)

    def make_session():
        """创建同一研究任务的宿主会话。"""
        session = Session(Model(ModelConfig(model="openai/fixture", api_key="fixture")),
                          SessionState(session_id="r", agent_id="r", task_id="t",
                                       workdir=tmp_path))
        session.register_stateful_tool(NotebookTools)
        session.register_stateful_tool(FeatureAnalyzeVerifier)
        return session

    async def scenario():
        async with make_session() as session:
            notebooks = session.resource(NotebookTools)
            notebooks._kernels["r1"] = None
            notebooks._save_document("r1", nbformat.v4.new_notebook(), None)
            try:
                receipt = await session.call("feature_analyze_verifier", {
                    "notebook_name": "r1", "report": "未完成验证", "round_id": "r1"})
                await asyncio.wait_for(entered.wait(), 5)
                queued = await session.call("feature_analyze_verifier", {
                    "notebook_name": "r1", "report": "排队的下一轮", "round_id": "r2"})
            finally:
                notebooks._kernels.clear()
        async with make_session() as resumed:
            verifier = resumed.resource(FeatureAnalyzeVerifier)
            assert (await verifier.status())["status"] == "busy"
            assert (await verifier.status())["queue_length"] == 1
            assert verifier.current["submission_id"] == receipt["submission_id"]

            async def complete(self, messages, *, tools=None):
                assert receipt["verification_path"] in str(messages)
                return {"role": "assistant", "tool_calls": [tool_call(
                    "report", {"status": "needs_revision", "feedback": "原轮缺少关键计算"}, "resume")]}

            monkeypatch.setattr(VerifierModel, "completion", complete)
            status = await verifier.status(retry=True, wait=True)
            assert status["latest_report"]["submission_id"] == queued["submission_id"]
            assert status["queue_length"] == 0
            assert len(list((tmp_path / "verifier_runs").glob("*/result.json"))) == 2
            assert status["status"] == "idle"
            assert status["latest_report"]["accepted"] is False

    asyncio.run(scenario())


def test_concurrent_submissions_are_frozen_and_processed_fifo(tmp_path, monkeypatch):
    """并发提交持久化后按序投递，等待消息不进入当前验证上下文。"""
    entered = asyncio.Event()
    release = asyncio.Event()
    order = []

    async def completion(self, messages, *, tools=None):
        current = messages[-1]["content"]
        folder = current.split("增量研究提交：", 1)[1].split("/submission.json", 1)[0]
        submission = json.loads((tmp_path / folder / "submission.json").read_text())
        order.append(submission["round_id"])
        if len(order) == 1:
            entered.set()
            await release.wait()
        return {"role": "assistant", "tool_calls": [tool_call(
            "report", {"status": "regressed", "feedback": f"已验证 {submission['round_id']}"},
            f"fifo-{len(order)}")]}

    monkeypatch.setattr(VerifierModel, "completion", completion)

    async def scenario():
        session = Session(Model(ModelConfig(model="openai/fixture", api_key="fixture")),
                          SessionState(session_id="r", agent_id="r", task_id="t",
                                       workdir=tmp_path))
        session.register_stateful_tool(NotebookTools)
        session.register_stateful_tool(FeatureAnalyzeVerifier)
        async with session:
            notebooks = session.resource(NotebookTools)
            verifier = session.resource(FeatureAnalyzeVerifier)
            notebooks._kernels["n"] = None
            notebooks._save_document("n", nbformat.v4.new_notebook(), None)
            try:
                first = await verifier.submit("n", "首轮", round_id="r0")
                await entered.wait()
                receipts = await asyncio.gather(*[
                    verifier.submit("n", f"第{i}轮", round_id=f"r{i}")
                    for i in range(1, 4)])
                assert all(item["status"] == "submitted" for item in receipts)
                saved = json.loads((tmp_path / "verifier_runs/control.json").read_text())
                assert [item["round_id"] for item in saved["queue"]] == ["r1", "r2", "r3"]
                assert saved["current"]["submission_id"] == first["submission_id"]
                assert order == ["r0"]
                for item in receipts:
                    assert item["verification_path"] not in str(verifier.child.state.messages)
                frozen = (tmp_path / receipts[0]["verification_path"] / "notebook.ipynb").read_text()
                notebooks._save_document("n", nbformat.v4.new_notebook(
                    cells=[nbformat.v4.new_markdown_cell("后续修改")]),
                    notebooks._read_document("n"))
                assert (tmp_path / receipts[0]["verification_path"] / "notebook.ipynb").read_text() == frozen
                release.set()
                status = await verifier.status(wait=True)
                assert status["status"] == "idle"
                assert order == ["r0", "r1", "r2", "r3"]
                assert [item["round_id"] for item in verifier.pending] == order
                assert status["queue_length"] == 0
            finally:
                notebooks._kernels.clear()

    asyncio.run(scenario())
