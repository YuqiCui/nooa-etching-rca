"""检查独立脚本的数据边界和用量统计，不调用真实模型。"""

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiments/run_sensor_similarity.py"
spec = importlib.util.spec_from_file_location("sensor_similarity_script", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_usage_summary_does_not_double_count_cached_or_reasoning_tokens():
    records = [
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "total_tokens": 130,
                "prompt_tokens_details": {"cached_tokens": 60},
                "completion_tokens_details": {"reasoning_tokens": 20},
            }
        },
        {"usage": {"input_tokens": 10, "output_tokens": 5}},
        {"usage": None},
        {"usage": {"total_tokens": 7}},
    ]
    summary = runner._summarize_usage(records)
    assert summary["received_responses"] == 4
    assert summary["responses_without_usage"] == 1
    assert summary["responses_with_incomplete_usage"] == 1
    assert summary["input_tokens"] == 110
    assert summary["output_tokens"] == 35
    assert summary["total_tokens"] == 152
    assert summary["cached_input_tokens"] == 60
    assert summary["reasoning_tokens"] == 20


def test_dry_run_never_loads_credentials_or_creates_run(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "task.txt").write_text("Top 10", encoding="utf-8")
    run = tmp_path / "run"

    def forbidden_provider(*args):
        raise AssertionError("dry-run must not load credentials")

    monkeypatch.setattr(runner, "load_provider", forbidden_provider)
    runner.main(["--dataset", str(dataset), "--run-dir", str(run), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["task"] == "Top 10"
    assert not run.exists()


def test_discovery_import_matches_previous_agent_scope(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    (public / "traces").mkdir()
    schema = {
        "modality": "timeseries",
        "index_file": "samples.parquet",
        "id_column": "run_id",
        "split_column": "split",
        "asset_column": "trace_path",
        "steps_file": "steps.parquet",
    }
    (public / "dataset.json").write_text(json.dumps({"schema": schema}))
    pl.DataFrame(
        {
            "run_id": ["a", "b"],
            "split": ["train", "test"],
            "trace_path": ["traces/a.parquet", "traces/b.parquet"],
        }
    ).write_parquet(public / "samples.parquet")
    pl.DataFrame({"run_id": ["a", "b"]}).write_parquet(public / "steps.parquet")
    for name in ["a", "b"]:
        (public / "traces" / f"{name}.parquet").write_bytes(b"trace")
    for name in ["sensors.json", "catalog.json", "recipe.json"]:
        (public / name).write_text("{}")
    for name in ["features.parquet", "group_A.parquet", "group_B.parquet", "truth.json"]:
        (public / name).write_bytes(b"excluded")
    target = tmp_path / "input"
    runner.discovery_copy(public, target)
    assert sorted(
        str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()
    ) == [
        "catalog.json",
        "dataset.json",
        "recipe.json",
        "samples.parquet",
        "sensors.json",
        "steps.parquet",
        "traces/a.parquet",
    ]
    assert pl.read_parquet(target / "samples.parquet")["run_id"].to_list() == ["a"]
    assert pl.read_parquet(target / "steps.parquet")["run_id"].to_list() == ["a"]


def test_model_override_preserves_provider_connection(tmp_path, monkeypatch):
    from factor_demo.providers import Provider

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    provider = Provider("custom", "original", "https://example.invalid/v1", "openai", "secret")
    monkeypatch.setattr(runner, "load_provider", lambda name: provider)
    for model in ("glm-5.3-flash", "deepseek-4.1-flash"):
        args = runner._parse_args(["--dataset", str(dataset), "--model", model])
        config = runner._build_model_config(args)
        assert config.model == f"openai/{model}"
        assert config.base_url == provider.base_url
        assert config.api_key.get_secret_value() == "secret"
        assert config.max_tokens is None
        assert config.num_retries == 10
        assert config.thinking["type"] == "enabled"
        assert config.reasoning_effort == "max"
        if model.startswith("glm-"):
            assert config.thinking["clear_thinking"] is False
        assert "budget_tokens" not in config.thinking
        assert config.tool_choice == ("auto" if model.startswith("deepseek") else "required")
        assert provider.model == "original"


@pytest.mark.parametrize("model", ["glm-5.3-flash", "deepseek-v4.1-flash"])
@pytest.mark.parametrize("effort", ["high", "max"])
def test_reasoning_controls_reach_sdk_request_body(tmp_path, monkeypatch, model, effort):
    """用真实 SDK 序列化核对旧配置的推理强度，不访问模型服务。"""
    from factor_demo.providers import Provider
    from factor_demo.runtime import RuntimeSettings

    settings = RuntimeSettings(thinking=True, reasoning_effort=effort)
    settings_path = tmp_path / "runtime.json"
    settings_path.write_text(settings.model_dump_json())
    provider = Provider("custom", model, "https://reasoning.invalid/v1", "openai", "fixture")
    monkeypatch.setattr(runner, "load_provider", lambda name: provider)
    requests = []

    async def send(client, request, **kwargs):
        assert request.url.host == "reasoning.invalid"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "完成"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    args = runner._parse_args(["--model", model, "--settings", str(settings_path)])
    config = runner._build_model_config(args)
    asyncio.run(runner.Model(config).completion([{"role": "user", "content": "核对请求"}]))
    assert len(requests) == 1
    body = requests[0]
    assert body["reasoning_effort"] == settings.reasoning_effort
    assert body["thinking"] == (
        {"type": "enabled", "clear_thinking": False}
        if model.startswith("glm-")
        else {"type": "enabled"}
    )
    assert "budget_tokens" not in body["thinking"]
    assert "max_tokens" not in body and "max_completion_tokens" not in body


def test_trace_preserves_requests_responses_and_failure_checkpoint(tmp_path, monkeypatch):
    from factor_demo.autoresearch_agent_v2.agent import SessionState
    from factor_demo.autoresearch_agent_v2.config import ModelConfig

    state = SessionState(session_id="s", agent_id="a", task_id="t", workdir=tmp_path)
    config = ModelConfig(model="openai/test", api_key="secret")
    model = runner._TracingModel(config, state, tmp_path, lambda usage: None)
    reply = {"role": "assistant", "content": "result", "reasoning_content": "reason"}

    async def complete(self, messages, *, tools=None):
        if len(messages) > 1:
            raise RuntimeError("failed request")
        return reply

    monkeypatch.setattr(runner.Model, "completion", complete)

    async def exercise():
        import pytest

        state.messages.append({"role": "user", "content": "task"})
        assert await model.completion(state.messages) == reply
        state.messages.append(reply)
        with pytest.raises(RuntimeError, match="failed request"):
            await model.completion(state.messages)

    asyncio.run(exercise())
    events = [
        json.loads(line) for line in (tmp_path / "model_events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["request", "response", "request", "error"]
    assert events[1]["message"] == reply
    assert events[-1]["error_type"] == "RuntimeError"
    assert json.loads((tmp_path / "session.json").read_text())["messages"] == state.messages
    assert "secret" not in (tmp_path / "model_events.jsonl").read_text()


@pytest.mark.parametrize("change", [None, "model", "status", "symlink", "dataset"])
def test_recovery_preserves_source_and_rejects_changed_or_unsafe_inputs(tmp_path, change):
    from factor_demo.autoresearch_agent_v2.agent import SessionState
    from factor_demo.autoresearch_agent_v2.config import ModelConfig

    source = tmp_path / "source"
    run = tmp_path / "recovery"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for path in (source / "input", source / "workspace", run / "workspace"):
        path.mkdir(parents=True)
    (source / "input/data.txt").write_text("data")
    (source / "workspace/result.txt").write_text("expensive result")
    config = ModelConfig(model="openai/test", api_key="secret")
    source_config = config.model_dump(mode="json", exclude={"api_key", "base_url"})
    source_config["dataset"] = str(dataset)
    (source / "run_config.json").write_text(json.dumps(source_config))
    (source / "status.json").write_text(
        json.dumps(
            {"status": "completed" if change == "status" else "failed", "error_type": "ValueError"}
        )
    )
    (source / "token_summary.json").write_text('{"received_responses":52}')
    old_state = SessionState(
        session_id="old",
        agent_id="a",
        task_id="old",
        workdir=source / "workspace",
        messages=[
            {"role": "user", "content": "早期任务"},
            {"role": "assistant", "content": "早期结果"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "pending", "function": {"name": "sandbox__run_shell", "arguments": "{}"}}
                ],
            }
        ],
        compaction={"summary": "早期任务已有结果。", "end_index": 2, "source_message_count": 2},
    )
    (source / "session.json").write_text(old_state.model_dump_json())
    original = (source / "session.json").read_bytes()
    state = SessionState(session_id="new", agent_id="a", task_id="new", workdir=run / "workspace")
    if change == "model":
        config = config.model_copy(update={"model": "openai/other"})
    if change == "dataset":
        dataset = tmp_path / "other"
    if change == "symlink":
        (source / "workspace/link").symlink_to(tmp_path / "secret")
    if change:
        with pytest.raises(ValueError):
            runner._restore_failed_run(source, run, state, config, dataset)
        assert not (run / "input").exists()
    else:
        prompt = runner._restore_failed_run(source, run, state, config, dataset)
        assert state.compaction == old_state.compaction
        assert str(state.workdir) in prompt
        assert state.messages[-1]["tool_call_id"] == "pending"
        assert "中断" in state.messages[-1]["content"]
        assert (run / "workspace/result.txt").read_text() == "expensive result"
        assert (run / "input/data.txt").read_text() == "data"
        assert json.loads((run / "recovery.json").read_text())["python_kernel_restored"] is False
    assert (source / "session.json").read_bytes() == original
