# ruff: noqa: F811
"""Regression cases from the code review, with real NOOA and worker boundaries."""

import asyncio
import json
from collections.abc import Mapping

import polars as pl
import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from test_capabilities import public_case
from test_orchestration import endpoint, settings  # noqa: F401 - pytest fixtures

from factor_demo.autoresearch import research_turn
from factor_demo.autoresearch_agent import AutoresearchAgent
from factor_demo.capabilities import Capabilities
from factor_demo.live_events import LiveEvents
from factor_demo.reply_writer import StreamingReplyWriter
from factor_demo.research_context import install_research_context
from factor_demo.research_workspace import PythonWorkspace, prepare_workspace


def test_model_sees_working_skill_directory_and_essential_rules(tmp_path):
    root = public_case(tmp_path / "case")
    response = LLMResponse(
        parts=(
            ToolCall(
                id="skills",
                name="python_cell",
                arguments=json.dumps(
                    {
                        "code": "rules = self.statistics.read_file('SKILL.md')\nassert 'fold' in rules\nprint(rules)\nreturn_result({'summary':'检查技能输入'})"
                    }
                ),
            ),
        ),
        finish_reason="tool_calls",
    )

    async def exercise():
        client = FakeLLMClient(scripted_responses=[response], strict_exhaustion=True)
        agent = AutoresearchAgent(PythonWorkspace(root), llm=client)
        try:
            install_research_context(agent, "检查技能")
            await agent.research("检查技能")
            text = "\n".join(
                str(m.get("content", ""))
                for m in client.calls[0].messages
                if isinstance(m, Mapping)
            )
            assert "no attribute 'skills'" not in text
            assert "Active Skills" in text and "self.statistics" in text
            assert "学得的特征和预处理须在训练fold内拟合" in text
        finally:
            await agent.aclose()

    asyncio.run(exercise())


def test_no_reference_input_is_optional_but_declared_reference_is_required(tmp_path):
    from test_formula_lab import fixture_case

    lab = fixture_case(tmp_path / "source")
    source = lab.run / "input"
    runs = pl.read_parquet(source / "runs.parquet")
    for name in ["outcomes", "features"]:
        runs.select("run_id").write_parquet(source / (name + ".parquet"))
    for name in ["sensors", "catalog", "recipe", "case"]:
        (source / (name + ".json")).write_text("{}")
    prepare_workspace(source, tmp_path / "export")
    assert pl.read_parquet(tmp_path / "export/input/control_features.parquet").height == 0
    runs.with_columns(pl.lit(True).alias("is_control")).write_parquet(source / "runs.parquet")
    with pytest.raises(FileNotFoundError):
        prepare_workspace(source, tmp_path / "broken")


def test_writer_failure_delivers_saved_result_without_repeating_research(
    endpoint, tmp_path, monkeypatch
):
    fake = endpoint()
    root = public_case(tmp_path / "case")

    async def failure(*args, **kwargs):
        raise ValueError("incomplete writer response")

    monkeypatch.setattr(StreamingReplyWriter, "send", failure)
    events = LiveEvents(root, "delivery")
    try:
        reply = asyncio.run(
            research_turn(
                root,
                "custom",
                "fixture",
                settings_path=settings(tmp_path),
                events=events,
                live=True,
                policy=Capabilities(review_enabled=False),
            )
        )
        assert reply["summary"]
        assert json.loads((root / "status.json").read_text())["status"] == "checkpoint"
        rows = events.since(0, 10000)
        final = [r for r in rows if r["type"] == "reply.commit"]
        assert len(final) == 1 and "公开数据" in final[0]["data"]["markdown"]
        assert any(r["type"] == "reply.delivery_fallback" for r in rows)
        assert fake.tools["main"] == 1
    finally:
        events.close()


def test_large_python_and_subprocess_output_is_bounded_and_recovers(tmp_path):
    from factor_demo.research_workspace import PythonWorkspace

    root = public_case(tmp_path / "case")

    async def exercise():
        worker = PythonWorkspace(root)
        try:
            result = await worker.execute("print('x'*2_000_000)")
            assert result["ok"] and result["output_truncated"]
            saved = json.loads((worker.work / result["full_output"]).read_text())
            assert len(saved["output"]) <= 262144
            assert saved["output_characters"] >= 2_000_000
            assert (await worker.execute("print('recovered')"))["output"].strip() == "recovered"
            result = await worker.execute(
                "import subprocess,sys\nsubprocess.run([sys.executable,'-c',\"print('y'*2_000_000)\"],check=True)"
            )
            assert result["ok"] and result["output_truncated"]
        finally:
            await worker.close()

    asyncio.run(exercise())


def test_snapshot_files_are_immutable_and_content_deduplicated(tmp_path):
    from factor_demo.artifact_store import snapshot_tree

    source = tmp_path / "source"
    source.mkdir()
    (source / "table.csv").write_text("x\n1\n")
    store = tmp_path / "store"
    snapshot_tree(source, tmp_path / "one", store)
    snapshot_tree(source, tmp_path / "two", store)
    assert (tmp_path / "one/table.csv").stat().st_ino == (tmp_path / "two/table.csv").stat().st_ino
    (source / "table.csv").write_text("x\n2\n")
    assert (tmp_path / "one/table.csv").read_text() == "x\n1\n"


def test_revision_limit_is_distinct_from_model_budget_exhaustion(endpoint, tmp_path):
    endpoint(revision=True)
    root = public_case(tmp_path / "case")
    events = LiveEvents(root, "revision-limit")
    try:
        reply = asyncio.run(
            research_turn(
                root,
                "custom",
                "fixture",
                settings_path=settings(tmp_path),
                events=events,
                live=True,
                policy=Capabilities(max_revisions=0),
            )
        )
        assert reply["review_status"] == "revision_limit"
        assert reply["delivery_status"] == "delivered"
        review = json.loads((root / "latest_review.json").read_text())
        assert review["reason"] == "revision_limit"
    finally:
        events.close()


def test_visualization_delivers_after_one_model_call_without_reviewer_or_writer(endpoint, tmp_path):
    fake = endpoint(visualization=True)
    root = public_case(tmp_path / "case")
    events = LiveEvents(root, "lightweight")
    try:
        result = asyncio.run(
            research_turn(
                root,
                "custom",
                "只查看数据，不做根因或模型比较",
                settings_path=settings(tmp_path),
                events=events,
                live=True,
                policy=Capabilities(max_model_calls=1, max_total_tokens=300000),
            )
        )
        assert result["review_status"] == "not_required"
        assert result["delivery_status"] == "delivered"
        assert len(fake.calls) == 1 and not fake.reviews
        assert result["dataset_ref"]
        assert len([e for e in events.since(0, 10000) if e["type"] == "reply.commit"]) == 1
    finally:
        events.close()
