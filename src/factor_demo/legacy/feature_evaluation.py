"""可独立交付的五折特征验证程序；仅在执行沙箱内运行。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate(table: pd.DataFrame, spec: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """执行固定分类协议，返回结论、逐折指标和逐样本预测。

    Args:
        table: 仅包含 train/dev 的候选特征表。
        spec: 特征、目标、分组、划分与发现范围字段。

    Returns:
        验证状态、逐折指标表及折外/独立 dev 预测表。

    Raises:
        ValueError: 数据泄漏、字段错误或缺少五折所需的独立组。
    """
    features = spec["features"]
    meta = [spec[k] for k in ("target", "group", "split", "sample_id")]
    if not features or len(set(features)) != len(features):
        raise ValueError("候选特征必须非空且不可重复")
    if len(set(meta)) != 4 or set(features) & set(meta):
        raise ValueError("元信息字段必须不同，且不得作为特征")
    missing = set(features + meta) - set(table.columns)
    if missing:
        raise ValueError(f"缺少字段：{sorted(missing)}")
    forbidden = {"label", "target", "period", "period_id", "timestamp", "run_id", "batch_id",
                 "lot", "lot_id", "split", "sample_id"}
    if any(f.lower() in forbidden for f in features):
        raise ValueError("标识、时间和目标代理字段不得作为候选特征")
    if table[meta].isna().any().any() or table[spec["sample_id"]].duplicated().any():
        raise ValueError("元信息不可缺失，sample_id 必须唯一")
    if not set(table[spec["split"]].unique()) <= {"train", "dev"}:
        raise ValueError("验证表只能包含 train/dev，拒绝 test/holdout 或未知划分")
    if spec["selection_scope"] not in {"predefined", "train_discovered"}:
        raise ValueError("必须声明特征发现范围")
    if spec["dev_usage"] not in {"untouched", "used_in_research", "absent"}:
        raise ValueError("必须声明 dev 是否参与研究决策")
    x = table[features].apply(pd.to_numeric, errors="raise")
    if np.isinf(x.to_numpy()).any():
        raise ValueError("候选特征不可含无穷值")
    class_names, y = np.unique(table[spec["target"]].astype(str).to_numpy(),
                               return_inverse=True)
    groups = table[spec["group"]].astype(str).to_numpy()
    train = np.flatnonzero(table[spec["split"]].eq("train").to_numpy())
    dev = np.flatnonzero(table[spec["split"]].eq("dev").to_numpy())
    classes = sorted(set(y[train]))
    if len(classes) < 2:
        raise ValueError("第一版验证器要求至少两个类别的分类目标")
    if set(groups[train]) & set(groups[dev]):
        raise ValueError("train/dev 存在相同独立组，拒绝泄漏")
    for label in classes:
        if len(set(groups[train][y[train] == label])) < 5:
            raise ValueError(f"类别 {label} 少于五个独立训练组，无法做稳定五折验证")
    if dev.size and set(y[dev]) != set(classes):
        raise ValueError("dev 必须覆盖全部训练类别，不得包含新类别")

    models = {
        "dummy": make_pipeline(SimpleImputer(keep_empty_features=True),
                               DummyClassifier(strategy="prior")),
        "logistic_regression": make_pipeline(
            SimpleImputer(keep_empty_features=True), StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        "random_forest": make_pipeline(
            SimpleImputer(keep_empty_features=True),
            RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=2,
                                   class_weight="balanced", n_jobs=1, random_state=42)),
    }
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                 .split(x.iloc[train], y[train], groups[train]))
    scores, predictions = [], []
    for fold, (fit_local, val_local) in enumerate(folds, 1):
        fit, val = train[fit_local], train[val_local]
        if set(groups[fit]) & set(groups[val]):
            raise ValueError("交叉验证组发生重叠")
        if set(y[fit]) != set(classes) or set(y[val]) != set(classes):
            raise ValueError("至少一折缺少类别；当前分组结构不支持完整五折验证")
        _score_split(models, x, y, groups, table, spec, fit, val, "cv", fold,
                     scores, predictions, class_names)
    if dev.size:
        _score_split(models, x, y, groups, table, spec, train, dev, "dev", 0,
                     scores, predictions, class_names)

    metrics = pd.DataFrame(scores)
    cv = metrics[metrics.partition.eq("cv")]
    baseline = cv[cv.model.eq("dummy")].set_index("fold").balanced_accuracy
    summary = {}
    for name in models:
        selected = cv[cv.model.eq(name)].set_index("fold").balanced_accuracy
        dev_rows = metrics[metrics.partition.eq("dev") & metrics.model.eq(name)]
        summary[name] = {
            "mean_balanced_accuracy": float(selected.mean()),
            "std_balanced_accuracy": float(selected.std(ddof=1)),
            "min_balanced_accuracy": float(selected.min()),
            "folds_above_dummy_margin": int(((selected - baseline) >= 0.05).sum()),
            "dev_balanced_accuracy": None if dev_rows.empty else float(
                dev_rows.balanced_accuracy.iloc[0]),
        }
    # 固定规则只评价已提交的候选集，不按结果搜索或调参。
    passed = [name for name in models if name != "dummy"
              and summary[name]["folds_above_dummy_margin"] >= 4]
    if spec["selection_scope"] == "train_discovered":
        passed = [name for name in passed if dev.size and spec["dev_usage"] == "untouched" and
                  summary[name]["dev_balanced_accuracy"] >=
                  summary["dummy"]["dev_balanced_accuracy"] + 0.05]
    missing_dev = spec["selection_scope"] == "train_discovered" and (
        not dev.size or spec["dev_usage"] != "untouched")
    result = {
        "status": "inconclusive" if missing_dev else ("accepted" if passed else "rejected"),
        "reason": "已筛选特征缺少未参与研究决策的独立 dev，不能确认有效性" if missing_dev else
                  ("候选特征集通过预设稳定性规则" if passed else "未通过预设稳定性规则"),
        "models": summary, "passing_models": passed,
        "policy": {"folds": 5, "min_passing_folds": 4, "dummy_margin": 0.05,
                   "group_weighting": "每个独立组在评分中总权重相同",
                   "selection_scope": spec["selection_scope"], "dev_usage": spec["dev_usage"]},
        "scope": "固定候选特征集的分类预测有效性；不证明每个特征独立有效或物理因果",
        "limitations": ["通过阈值是工程验收规则，不是统计显著性检验",
                        "train_discovered 的训练交叉验证含发现偏差，独立 dev 为补充检查",
                        "验证器依赖提交的分组、划分及特征来源声明，需要核对 Notebook"],
    }
    return result, metrics, pd.DataFrame(predictions)


def _score_split(models, x, y, groups, table, spec, fit, val, partition, fold,
                 scores, predictions, class_names):
    """在训练范围拟合全部预处理，并对独立组等权评分。"""
    counts = pd.Series(groups[val]).value_counts()
    weights = np.array([1.0 / counts[g] for g in groups[val]])
    for name, prototype in models.items():
        model = clone(prototype).fit(x.iloc[fit], y[fit])
        pred = model.predict(x.iloc[val])
        scores.append({
            "partition": partition, "fold": fold, "model": name,
            "balanced_accuracy": balanced_accuracy_score(y[val], pred, sample_weight=weights),
            "macro_f1": f1_score(y[val], pred, average="macro", sample_weight=weights),
            "train_groups": len(set(groups[fit])), "validation_groups": len(set(groups[val])),
        })
        for index, prediction, weight in zip(val, pred, weights, strict=True):
            predictions.append({"partition": partition, "fold": fold, "model": name,
                                "sample_id": table[spec["sample_id"]].iloc[index],
                                "group": groups[index], "target": class_names[y[index]],
                                "prediction": class_names[prediction], "weight": weight})


def save_results(result, metrics, predictions, output: Path) -> None:
    """保存机器可读证据、逐折分数图及各模型混淆矩阵。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    (output / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    cv = metrics[metrics.partition.eq("cv")]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for name, rows in cv.groupby("model"):
        ax.plot(rows.fold, rows.balanced_accuracy, marker="o", label=name)
    ax.set(xlabel="Fold", ylabel="Group-weighted balanced accuracy", ylim=(0, 1.05),
           xticks=range(1, 6))
    ax.legend()
    fig.savefig(output / "cv_scores.png", dpi=160)
    plt.close(fig)
    names = sorted(cv.model.unique())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4),
                             constrained_layout=True, squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        rows = predictions[predictions.partition.eq("cv") & predictions.model.eq(name)]
        labels = sorted(rows.target.unique())
        matrix = confusion_matrix(rows.target, rows.prediction, labels=labels,
                                  sample_weight=rows.weight, normalize="true")
        ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set(title=name, xlabel="Predicted", ylabel="Observed",
               xticks=range(len(labels)), yticks=range(len(labels)),
               xticklabels=labels, yticklabels=labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="重跑固定候选集的五折分组分类验证")
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    spec = json.loads((args.directory / "spec.json").read_text())
    result, metrics, predictions = evaluate(
        pd.read_parquet(args.directory / "features.parquet"), spec)
    save_results(result, metrics, predictions, args.directory)
    print(json.dumps(result, ensure_ascii=False))
