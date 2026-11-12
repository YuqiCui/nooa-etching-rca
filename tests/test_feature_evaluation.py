"""保留历史固定评估器的数值与数据边界回归，不包含独立 Agent。"""


import numpy as np
import pandas as pd
import pytest

from factor_demo.legacy.feature_evaluation import evaluate, save_results


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
