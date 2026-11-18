"""汇总冻结实验的逐案例结果，生成可重复更新的 HTML 台账。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
RENDERER = Path.home() / ".codex/skills/answer-me-with-html/scripts/am.mjs"
STATUS = {
    "completed": "已结束", "running": "运行中", "timeout": "超时", "failed": "失败",
    "cancelled": "已取消", "queued": "待运行", "missing": "无状态记录",
}
TRAINING = {
    "train_only_in_submitted_code": "提交代码仅 train",
    "train_validation_refit": "train+validation 重拟合",
    "missing_script": "缺脚本", "incomplete_script": "脚本不完整",
    "not_reviewed": "未审", "needs_review": "需重审",
}


def read_json(path: Path, default=None):
    """读取可选证据，保留正在写入或损坏文件的异常记录。"""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {} if default is None else default
    except (OSError, ValueError) as error:
        return {"read_error": type(error).__name__, "path": str(path)}


def save_json(path: Path, value) -> None:
    """原子写入完整快照。"""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def cell(value) -> str:
    """转义 Markdown 表格中的分隔符和换行。"""
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers, rows) -> str:
    """生成列数固定的 Markdown 表格。"""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def number(value, digits=5) -> str:
    """显示有限数值，不把缺失或失败转换成零。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.{digits}f}"
    return "—"


def link(output: Path, path: Path, label: str) -> str:
    """生成相对于台账的可点击本地证据链接。"""
    return f"[{label}]({os.path.relpath(path, output)})"


def snapshot_hashes(code: Path) -> dict[str, str]:
    """只比较源码及技能，不读取连接配置和凭据。"""
    source = code / "src" if (code / "src").is_dir() else code
    if not source.is_dir():
        return {}
    return {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".md", ".jsonc"}
        and "__pycache__" not in path.parts
    }


def required_files(folder: Path, workspace: Path) -> tuple[list[str], list[str], list[str]]:
    """按公开任务要求检查规定目录，并单列根目录中的错位文件。"""
    task = read_json(folder / "case/input/task.json")
    required = list(task.get("required_artifacts") or [
        "predictions.csv", "report.md", "research.json", "analysis.py", "ablation.csv",
    ])
    if task.get("category") == "feature_mining":
        required = list(dict.fromkeys([
            *required, "features_train.csv", "features_validation.csv", "features_test.csv",
        ]))
    artifacts = workspace / "artifacts/final"
    present = [name for name in required if (artifacts / name).is_file()
               and (artifacts / name).stat().st_size > 0]
    missing = [name for name in required if name not in present]
    misplaced = [name for name in missing if (workspace / name).is_file()]
    return required, present, misplaced


def collect_job(folder: Path, job: dict, version: dict, now: datetime) -> dict:
    """以单个运行目录中的原始状态和评分文件为主要证据。"""
    status = read_json(folder / "status.json")
    config = read_json(folder / "run_config.json")
    checks = read_json(folder / "artifact_checks.json")
    score = read_json(folder / "test_score.json")
    feature = read_json(folder / "test_feature_score.json")
    training = read_json(folder / "training_protocol_check.json")
    workspace = folder / "case/workspace"
    required, present, misplaced = required_files(folder, workspace)
    # 单案例旧目录使用 artifacts/concrete，由当时验收清单判断文件交付情况。
    legacy = "directory" not in job
    legacy_checks = checks.get("files") or {name: value for name, value in checks.items()
                                          if name in required and isinstance(value, dict)}
    if legacy and legacy_checks:
        required = list(legacy_checks)
        present = [name for name, value in legacy_checks.items()
                   if value is True or (isinstance(value, dict) and value.get("exists"))]
    elapsed = status.get("elapsed_s")
    if elapsed is None and status.get("started_at"):
        end = status.get("finished_at")
        try:
            elapsed = ((datetime.fromisoformat(end) if end else now)
                       - datetime.fromisoformat(status["started_at"])).total_seconds()
        except (TypeError, ValueError):
            elapsed = None
    state = status.get("status", "missing")
    if state == "missing" and version.get("protocol", {}).get("jobs"):
        state = "queued"
    note = []
    if status.get("error_type"):
        note.append(status["error_type"])
    if misplaced:
        note.append("根目录：" + "、".join(misplaced))
    if state == "completed" and not present:
        note.append("已结束但无规定目录产物")
    training_status = training.get("status", "not_reviewed")
    script = workspace / "artifacts/final/analysis.py"
    reviewed = training.get("reviewed_sha256")
    if reviewed and script.is_file() and hashlib.sha256(script.read_bytes()).hexdigest() != reviewed:
        training_status = "needs_review"
    return {
        "version": version["id"], "case": job.get("case", config.get("case", "05-concrete")),
        "arm": job.get("arm", "—"), "directory": str(folder), "status": state,
        "status_evidence": status, "elapsed_s": elapsed,
        "required": required, "present": present,
        "missing": [name for name in required if name not in present], "misplaced": misplaced,
        "delivery_valid_recorded": checks.get("delivery_valid"),
        "test_score": score.get("primary_score"), "primary_metric": score.get("primary_metric"),
        "feature_rmse_reduction": feature.get("rmse_reduction_fraction"),
        "training": training_status, "training_evidence": training,
        "resume_count": status.get("resume_count", 0), "notes": note,
        "settings": config.get("settings", {}), "score_evidence": str(folder / "test_score.json"),
        "artifact_evidence": str(folder / "artifact_checks.json"),
        "read_errors": [obj for obj in (status, config, checks, score, feature, training)
                        if obj.get("read_error")],
    }


def collect(catalog: dict, runs: Path) -> dict:
    """收集每个版本与每个案例，不启动研究或补算测试分数。"""
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    versions, records, diffs = [], [], {}
    hashes_by_directory = {}
    for entry in catalog["versions"]:
        folder = runs / entry["directory"]
        protocol = read_json(folder / "protocol.json") or read_json(folder / "comparison_protocol.json")
        code = Path(protocol.get("source_code", protocol.get("source_snapshot", folder / "code")))
        hashes = snapshot_hashes(code)
        base = hashes_by_directory.get(entry.get("baseline"), {})
        changed = sorted(name for name in hashes.keys() | base.keys()
                         if hashes.get(name) != base.get(name)) if base else []
        hashes_by_directory[entry["directory"]] = hashes
        skills = {name: digest for name, digest in hashes.items() if name.endswith("SKILL.md")}
        version = {**entry, "protocol": protocol, "source": str(code),
                   "source_files": len(hashes), "changed_files": changed,
                   "skill_hashes": skills, "state": read_json(folder / "suite_status.json")}
        jobs = protocol.get("jobs") or [{"case": "05-concrete"}]
        for job in jobs:
            target = Path(job["directory"]) if "directory" in job else folder
            records.append(collect_job(target, job, version, now))
        versions.append(version)
        diffs[entry["id"]] = {"baseline": entry.get("baseline"), "changed_files": changed,
                              "source": str(code), "skill_sha256": skills}
    active = sum(row["status"] in {"running", "queued"}
                 for row in records if next(v for v in versions if v["id"] == row["version"])["scope"] == "正式")
    pending_evaluations = []
    for version in versions:
        folder = runs / version["directory"]
        supervisor = folder / "supervisor.py"
        if (version["scope"] == "正式" and supervisor.exists()
                and "evaluation_status.json" in supervisor.read_text()
                and not (folder / "evaluation_status.json").exists()):
            pending_evaluations.append(version["id"])
    return {"updated_at": now.isoformat(timespec="seconds"), "versions": versions,
            "records": records, "diffs": diffs, "active_jobs": active,
            "pending_evaluations": pending_evaluations}


def group_policy(version: dict) -> str:
    """显示各实验组的真实含义，避免跨轮误读 A/B。"""
    protocol = version["protocol"]
    arms = protocol.get("arms", {})
    parts = []
    for name, options in arms.items():
        if "router_skill_enabled" in options:
            parts.append(f"{name.upper()}：Skill {'开' if options['router_skill_enabled'] else '关'}")
        elif "compact_on_worker_reassignment" in options:
            parts.append(f"{name.upper()}：换任务压缩{'开' if options['compact_on_worker_reassignment'] else '关'}")
    return "；".join(parts) or ("Router 按需加载 feature-research" if protocol.get("skills") else "Concrete 单案例")


def draft(data: dict, output: Path, runs: Path) -> str:
    """生成版本表、逐案例结果与指标口径说明。"""
    versions, records = data["versions"], data["records"]
    formal = {v["id"] for v in versions if v["scope"] == "正式"}
    included = [row for row in records if row["version"] in formal]
    scored = sum(row["test_score"] is not None for row in included)
    excluded = len(records) - len(included)
    lines = [
        "---", "title: NOOA 实验结果台账 · 11 月 17—18 日", "lang: zh", "layout: doc", "---",
        "## 两天总览",
        f"更新于 {data['updated_at']}。收录 {len(versions)} 轮、{len(records)} 条案例记录。",
        f"正式轮次 {len(formal)} 轮、{len(included)} 条；另列启动排除记录 {excluded} 条。",
        f"正式记录中，已有测试评分 {scored} 条，运行或待运行 {data['active_jobs']} 条。",
        "范围按实验启动目录归入 2026 年 11 月 17—18 日。11 月 16 日基线仅作为数据来源，不重复计入。",
        "所有登记轮次均使用 deepseek-v4.1-flash、medium 推理强度和原生 macOS 内部隔离。",
        "## 结果口径",
        "已结束表示模型结束了任务；文件齐全与宿主验收分开记录。超时任务也可能留下可评分预测。",
        "主分数只读取已有 test_score.json，不用模型自报的 validation 分数代替。缺失分数显示为 —。",
        "特征增益是固定学习器的测试 RMSE 相对降幅，正值更好；它与提交预测的主分数分开列出。",
        "训练审计只覆盖已记录的静态检查。train+validation 重拟合不等于符合任务的 train-only 要求。",
        "文件数按规定的最终目录检查；账本或报告留在根目录时会单列，不能算规定位置齐全。",
        "不同 case 的指标不混合求平均。对照轮次若同时改了多个设置，不能把差异归因给单一改动。",
        table(["Case", "主指标", "方向"], [
            ("AI4I", "Machine failure AP", "越高越好"), ("Hydraulic", "四目标平均 macro-F1", "越高越好"),
            ("SECOM", "AP", "越高越好"), ("Steel", "macro-F1", "越高越好"),
            ("Concrete / Airfoil / Bike / Wine / Energy", "RMSE", "越低越好"),
            ("Bank", "AP", "越高越好"),
        ]),
        "## 版本改动",
    ]
    version_rows = []
    for version in versions:
        p = version["protocol"]
        timeout = p.get("task_timeout_s", p.get("task_active_timeout_s"))
        duration = f"{timeout / 60:g} 分" if timeout else "见运行配置"
        limit = p.get("code_execution_limit", p.get("code_budget"))
        budget = p.get("router_time_budget_s")
        if budget:
            duration += f"；提醒 {budget / 60:g} 分"
        policy = group_policy(version)
        if limit is not None:
            policy += f"；代码 {limit} 次"
        folder = runs / version["directory"]
        source_file = folder / ("protocol.json" if (folder / "protocol.json").exists() else "comparison_protocol.json")
        title = link(output, source_file if source_file.exists() else folder / "run_config.json", version["id"] + " " + version["title"])
        change = version["changes"]
        if version["baseline"]:
            baseline = next((v["id"] for v in versions if v["directory"] == version["baseline"]), version["baseline"])
            change += f" 相对 {baseline}，源码/技能差异 {len(version['changed_files'])} 个文件。"
        version_rows.append((title, version["scope"], duration, policy, change))
    lines.append(table(["版本", "纳入范围", "总时限", "组别与执行限制", "改动及依据"], version_rows))
    lines += ["## 各轮结果总览"]
    overview = []
    for version in versions:
        rows = [row for row in records if row["version"] == version["id"]]
        counts = Counter(row["status"] for row in rows)
        overview.append((version["id"], len(rows),
                         "；".join(f"{STATUS.get(k, k)} {n}" for k, n in counts.items()),
                         sum(not row["missing"] for row in rows),
                         sum(row["test_score"] is not None for row in rows),
                         sum(row["delivery_valid_recorded"] is True for row in rows)))
    lines.append(table(["版本", "案例数", "执行状态", "规定文件齐全", "已有测试分数", "已有宿主交付通过记录"], overview))
    for version in versions:
        lines += [f"## {version['id']} {version['title']}", version["changes"], group_policy(version)]
        rows = []
        for row in (item for item in records if item["version"] == version["id"]):
            folder = Path(row["directory"])
            evidence = folder / "test_score.json" if row["test_score"] is not None else folder / "status.json"
            label = link(output, evidence if evidence.exists() else folder, row["case"])
            files = f"{len(row['present'])}/{len(row['required'])}"
            if row["misplaced"]:
                files += "；根目录有 " + "、".join(row["misplaced"])
            if row["status"] == "completed" and not row["present"]:
                files += "；未交付"
            score = number(row["test_score"])
            if row["primary_metric"]:
                score += " " + row["primary_metric"]
            gain = (f"{row['feature_rmse_reduction']:+.2%}"
                    if row["feature_rmse_reduction"] is not None else "—")
            minutes = number(row["elapsed_s"] / 60 if row["elapsed_s"] is not None else None, 1)
            if row["resume_count"]:
                minutes += f"；续接 {row['resume_count']} 次"
            rows.append((label, row["arm"].upper(), STATUS.get(row["status"], row["status"]),
                         files, score, gain, minutes, TRAINING.get(row["training"], row["training"])))
        lines.append(table(["Case / 原始证据", "组", "执行", "规定文件", "测试主分数", "特征 RMSE 降幅", "分钟", "训练审计"], rows))
    lines += [
        "## 维护与证据",
        "台账从原始状态、评分和交付检查文件重新生成，不修改历史实验，也不启动或重跑研究。",
        "运行中的任务每分钟刷新；全部任务结束后等待评分文件稳定，再停止本次自动维护。",
        "新增实验需在 catalog.json 中登记版本说明和对照基线，逐案例结果自动采集。",
        "版本差异比较实际冻结的 Python、Markdown 与 JSONC 文件，完整文件清单和 Skill 哈希保存在 version_diffs.json。",
        "API 首输出计时是后续工作区改动；当前登记的冻结版本未自动获得它，不能据此回溯本轮首 token 耗时。",
        link(output, output / "catalog.json", "版本目录") + " · "
        + link(output, output / "ledger.json", "结构化结果") + " · "
        + link(output, output / "version_diffs.json", "源码与 Skill 差异"),
    ]
    return "\n\n".join(lines) + "\n"


def build(catalog_path: Path, output: Path) -> dict:
    """采集并生成 HTML，以固定路径覆盖更新。"""
    catalog = read_json(catalog_path)
    output.mkdir(parents=True, exist_ok=True)
    data = collect(catalog, PROJECT / "runs")
    save_json(output / "ledger.json", data)
    save_json(output / "version_diffs.json", data["diffs"])
    source = output / "index.am.md"
    source.write_text(draft(data, output, PROJECT / "runs"))
    temporary_page = output / "index.next.html"
    result = subprocess.run([
        "node", str(RENDERER), "render", str(source), "--no-open", "-o", str(temporary_page),
        "--template", "doc",
    ], check=False, capture_output=True, text=True, timeout=60)
    (output / "render.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    temporary_page.replace(output / "index.html")
    print(json.dumps({"updated_at": data["updated_at"], "versions": len(data["versions"]),
                      "records": len(data["records"]), "active_jobs": data["active_jobs"],
                      "page": str(output / "index.html")}, ensure_ascii=False), flush=True)
    return data


def main() -> None:
    """按需单次更新，或持续到本批运行与评分完成。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    stable_rounds = 0
    for _ in range(180 if args.watch else 1):
        try:
            data = build(args.catalog.resolve(), args.output.resolve())
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            if not args.watch:
                raise
            save_json(args.output / "maintenance.json", {
                "pid": os.getpid(), "status": "retrying", "error_type": type(error).__name__,
                "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            })
            time.sleep(60)
            continue
        idle = data["active_jobs"] == 0 and not data["pending_evaluations"]
        stable_rounds = stable_rounds + 1 if idle else 0
        save_json(args.output / "maintenance.json", {
            "pid": os.getpid(), "updated_at": data["updated_at"],
            "status": "watching" if args.watch and stable_rounds < 8 else "snapshot",
            "active_jobs": data["active_jobs"],
            "pending_evaluations": data["pending_evaluations"],
        })
        if not args.watch or stable_rounds >= 8:
            break
        time.sleep(60)
    save_json(args.output / "maintenance.json", {
        "pid": os.getpid(), "status": "stopped" if args.watch else "snapshot",
        "reason": "runs_and_evaluations_settled" if stable_rounds >= 8 else "bounded_refresh_finished",
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    })


if __name__ == "__main__":
    main()
