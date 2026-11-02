"""Other process schemas exercise the real CodeAct worker without etch defaults."""

import asyncio
import json
from collections.abc import Mapping

import polars as pl
import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from factor_demo.autoresearch_agent import AutoresearchAgent
from factor_demo.dataset_models import DatasetSchema
from factor_demo.research_context import install_research_context
from factor_demo.research_workspace import PythonWorkspace, prepare_workspace


@pytest.mark.parametrize("process", ["epi", "pvd", "cvd"])
def test_other_processes_keep_custom_roles_sensors_steps_and_variable_recipes(tmp_path, process):
    source = tmp_path / ("etch_is_only_a_path_fragment_" + process)
    source.mkdir()
    (source / "traces").mkdir()
    rows = []
    for i in range(5):
        name = f"traces/{i}.parquet"
        pl.DataFrame(
            {"elapsed": [0.0, 1.0, 2.0], "phase": [17, 17, 23], "custom_sensor": [1.0, 2.0, 3.0]}
        ).write_parquet(source / name)
        rows.append(
            {
                "wafer_key": f"{process}-{i}",
                "condition": f"condition-{i}",
                "campaign": f"lot-{i}",
                "partition": "test" if i == 4 else "train",
                "telemetry": name,
                "recipe_key": "recipe-" + str(i % 2),
            }
        )
    pl.DataFrame(rows).write_csv(source / "lots.csv")
    schema = DatasetSchema(
        modality="timeseries",
        index_file="lots.csv",
        id_column="wafer_key",
        label_column="condition",
        group_column="campaign",
        split_column="partition",
        asset_column="telemetry",
        time_column="elapsed",
        step_column="phase",
        recipe_column="recipe_key",
        fixed_recipe=False,
        units={"custom_sensor": "declared_unit"},
    )
    (source / "dataset.json").write_text(json.dumps({"schema": schema.model_dump()}))
    root = tmp_path / "case"
    prepare_workspace(source, root)
    code = "result = await self.run_code(\"import polars as pl\\nframe=pl.read_csv(DATASET/'lots.csv')\\nprint('sample_count', frame.height)\\nprint(pl.read_parquet(DATASET / frame['telemetry'][0]).columns)\")\nassert result['ok'], result['output']\nreturn_result({'summary':result['output']})"
    response = LLMResponse(
        parts=(ToolCall(id="process", name="python_cell", arguments=json.dumps({"code": code})),),
        finish_reason="tool_calls",
    )

    async def exercise():
        client = FakeLLMClient(scripted_responses=[response], strict_exhaustion=True)
        agent = AutoresearchAgent(PythonWorkspace(root), llm=client)
        try:
            install_research_context(agent, "读取当前工艺数据，不预设传感器或步骤")
            assert agent._knowledge_registry.activated() == ["core.physics", "core.statistics"]
            result = await agent.research("按实际schema检查数据")
            assert "sample_count 4" in result.summary and "custom_sensor" in result.summary
            text = "\n".join(
                str(m.get("content", ""))
                for m in client.calls[0].messages
                if isinstance(m, Mapping)
            )
            assert "backside_he_leak" not in text and "自主刻蚀研究员" not in text
            assert "不预设工艺类型" in text
            assert agent._datasets.schema().recipe_column == "recipe_key"
            assert not agent._datasets.schema().fixed_recipe
        finally:
            await agent.aclose()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "profile,fixed,domain",
    [(None, True, False), ("etch", False, True), ("future_process", True, False)],
)
def test_domain_knowledge_requires_explicit_profile_not_fixed_recipe(
    tmp_path, profile, fixed, domain
):
    from test_capabilities import public_case

    root = public_case(tmp_path / "case")
    schema = DatasetSchema(
        index_file="runs.parquet",
        id_column="run_id",
        split_column="split",
        recipe_column="recipe_id",
        fixed_recipe=fixed,
        knowledge_profile=profile,
    )
    (root / "input/dataset.json").write_text(json.dumps({"schema": schema.model_dump()}))

    async def exercise():
        agent = AutoresearchAgent(PythonWorkspace(root), llm=FakeLLMClient())
        try:
            install_research_context(agent, "根据本轮问题研究")
            assert ("domain.domain_rules" in agent._knowledge_registry.activated()) == domain
            assert "core.physics" in agent._knowledge_registry.activated()
        finally:
            await agent.aclose()

    asyncio.run(exercise())


def test_profile_cannot_load_arbitrary_paths():
    with pytest.raises(ValueError):
        DatasetSchema(knowledge_profile="../../private")


@pytest.mark.parametrize("directory_asset", [False, True])
def test_private_assessment_uses_declared_identity_and_holdout_roles(tmp_path, directory_asset):
    from factor_demo.evaluation import prepare_evaluation

    source = tmp_path / "source"
    source.mkdir()
    (source / "truth.private").mkdir()
    (source / "traces").mkdir()
    pl.DataFrame({"temperature": [300.0, 301.0]}).write_parquet(source / "traces/wafer.parquet")
    asset = "traces/wafer.parquet"
    if directory_asset:
        (source / "traces/object").mkdir()
        (source / "traces/wafer.parquet").rename(source / "traces/object/sensor.parquet")
        asset = "traces/object"
    pl.DataFrame(
        {
            "wafer_key": ["held-wafer"],
            "condition": ["stable"],
            "partition": ["sealed"],
            "telemetry": [asset],
        }
    ).write_csv(source / "lots.csv")
    schema = DatasetSchema(
        index_file="lots.csv",
        id_column="wafer_key",
        label_column="condition",
        split_column="partition",
        asset_column="telemetry",
        excluded_splits=["sealed"],
    )
    (source / "dataset.json").write_text(json.dumps({"schema": schema.model_dump()}))
    assessment = {"expected_mechanisms": ["custom_process_mechanism"], "localization": {}}
    (source / "truth.private/assessment.json").write_text(json.dumps(assessment))
    run = tmp_path / "case"
    prepare_evaluation(source, run)
    assert json.loads((run / ".evaluation/manifest.json").read_text())["test_ids"] == ["held-wafer"]
    assert json.loads((run / ".evaluation/truth.json").read_text()) == assessment
    assert pl.read_parquet(run / ".evaluation/input/runs.parquet")["wafer_key"].to_list() == [
        "held-wafer"
    ]
    assert (run / ".evaluation/input" / asset).exists()


def test_switching_dataset_refreshes_and_retires_the_explicit_domain_pack(tmp_path):
    from test_capabilities import public_case

    root = public_case(tmp_path / "case")
    schema = DatasetSchema(index_file="runs.parquet", id_column="run_id", knowledge_profile="etch")
    (root / "input/dataset.json").write_text(json.dumps({"schema": schema.model_dump()}))

    from factor_demo.datasets import DatasetSession

    session = DatasetSession(root)
    original = session.current()["id"]
    child = session.registry.register(
        root / "input",
        schema.model_copy(update={"knowledge_profile": None}),
        "generic representation",
        parents=[original],
        reason="explicit scope change",
        relation="transform",
    )
    codes = [
        f"assert 'backside_he_leak' in self.domain_rules.read_file('SKILL.md')\nawait self.select_dataset({child['id']!r})",
        "try:\n self.domain_rules.read_file('SKILL.md')\nexcept AttributeError:\n pass\nelse:\n raise AssertionError('Previous scope skill is still exposed')\nawait self.return_dataset('root')",
        "assert 'backside_he_leak' in self.domain_rules.read_file('SKILL.md')\nreturn_result({'summary':'领域包随数据范围切换'})",
    ]
    responses = [
        LLMResponse(
            parts=(
                ToolCall(
                    id=f"domain-{i}", name="python_cell", arguments=json.dumps({"code": code})
                ),
            ),
            finish_reason="tool_calls",
        )
        for i, code in enumerate(codes)
    ]

    async def exercise():
        client = FakeLLMClient(scripted_responses=responses, strict_exhaustion=True)
        agent = AutoresearchAgent(PythonWorkspace(root), llm=client)
        try:
            install_research_context(agent, "切换到其他工艺数据表示")
            result = await agent.research("切换并回退领域包")
            assert result.summary == "领域包随数据范围切换"
            assert len(client.calls) == 3
            prompts = [str(call.messages[0]["content"]) for call in client.calls]
            assert "backside_he_leak" in prompts[0]
            assert "backside_he_leak" not in prompts[1]
            assert "未加载专用工艺知识" in prompts[1]
            assert "backside_he_leak" in prompts[2]
            assert "domain.domain_rules" in agent._knowledge_registry.activated()
        finally:
            await agent.aclose()

    asyncio.run(exercise())


def test_an_untyped_hardware_file_never_implies_etch_truth(tmp_path):
    from factor_demo.evaluation import prepare_evaluation

    source = tmp_path / "source"
    (source / "truth.private").mkdir(parents=True)
    (source / "truth.private/hardware.json").write_text(
        '{"arbitrary_device":"not an etch simulator"}'
    )
    run = tmp_path / "case"
    prepare_evaluation(source, run)
    assert not (run / ".evaluation").exists()
