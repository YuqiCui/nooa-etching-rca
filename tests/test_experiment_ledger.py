"""验证实验台账保留失败、缺交付、零分和原始证据口径。"""

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

spec = importlib.util.spec_from_file_location(
    "experiment_ledger", Path(__file__).parents[1] / "scripts/build_experiment_ledger.py",
)
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


def write(path, value):
    """创建最小的宿主证据文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_completed_missing_delivery_does_not_become_valid(tmp_path):
    """finish 与交付完整性分离，零分不转成缺失，根目录文件单独记录。"""
    write(tmp_path / "status.json", {"status": "completed", "elapsed_s": 60})
    write(tmp_path / "test_score.json", {"primary_metric": "average_precision", "primary_score": 0})
    write(tmp_path / "case/input/task.json", {"required_artifacts": ["predictions.csv", "research.json"]})
    write(tmp_path / "case/workspace/research.json", {"goal": "研究"})
    row = ledger.collect_job(tmp_path, {"case": "01-ai4i", "directory": str(tmp_path)},
                             {"id": "V01"}, datetime.now(ZoneInfo("Asia/Shanghai")))
    assert row["status"] == "completed"
    assert row["test_score"] == 0
    assert row["delivery_valid_recorded"] is None
    assert row["missing"] == ["predictions.csv", "research.json"]
    assert row["misplaced"] == ["research.json"]


def test_legacy_top_level_artifact_checks_are_supported(tmp_path):
    """旧单案例的平铺检查结构不被误报为全部缺失。"""
    write(tmp_path / "artifact_checks.json", {
        "predictions.csv": {"exists": True, "bytes": 100},
        "report.md": {"exists": False}, "source_manifest_unchanged": True,
    })
    row = ledger.collect_job(tmp_path, {"case": "05-concrete"}, {"id": "V01"},
                             datetime.now(ZoneInfo("Asia/Shanghai")))
    assert row["present"] == ["predictions.csv"]
    assert row["missing"] == ["report.md"]


def test_excluded_stale_suite_does_not_count_as_active(tmp_path):
    """启动排除的陈旧运行标记不让维护进程一直等待。"""
    suite = tmp_path / "attempt"
    write(suite / "protocol.json", {"jobs": [{"case": "01-ai4i", "directory": str(suite / "job")}]})
    write(suite / "job/status.json", {"status": "running"})
    data = ledger.collect({"versions": [{"id": "V01", "directory": "attempt", "scope": "启动排除"}]}, tmp_path)
    assert len(data["records"]) == 1
    assert data["active_jobs"] == 0


@pytest.mark.parametrize("value,expected", [(0, "0.00000"), (None, "—"), (float("nan"), "—")])
def test_numeric_missing_values(value, expected):
    """分数缺失和无效数值不伪装为有效零分。"""
    assert ledger.number(value) == expected
