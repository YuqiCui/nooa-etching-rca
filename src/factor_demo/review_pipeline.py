"""Evidence checks, native Predict review and bounded targeted revisions."""

import json
from pathlib import Path

from nooa.unifiedllm.admission import AdmissionCallCapError

from .artifact_store import snapshot_file, snapshot_tree
from .budget import BudgetExceeded
from .evidence import EvidenceVerifier
from .io_utils import write_json
from .live_events import public_artifact_path
from .research_models import ResearchReply, ReviewIssue, ReviewResult


def freeze_revision(run: Path, folder: Path, report: ResearchReply, verification: dict):
    folder.mkdir(parents=True, exist_ok=False)
    if (run / "workspace/artifacts").exists():
        snapshot_tree(run / "workspace/artifacts", folder / "artifacts", run / ".artifact-store")
    cited = (
        set(report.artifacts)
        | {p for h in report.hypotheses for p in h.evidence_files}
        | {e.artifact for e in report.evidence}
    )
    for name in cited:
        source = public_artifact_path(run, name)
        # Only already-validated relative artifact names enter this immutable snapshot.
        target = folder / "cited" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(source, target, run / ".artifact-store")
    write_json(folder / "report.json", report.model_dump())
    write_json(folder / "verification.json", verification)


def review_bundle(report, verification, events=None):
    observed = []
    if events:
        observed = events.reflection_events()
    return {
        "report": report.model_dump(),
        "saved_table_verification": verification,
        "host_observed_reflection_events_latest_first": observed,
        "rubric": "先识别本轮任务及报告实际声明，仅检查适用项目。通用项：证据一致性、数据范围、独立样本粒度、监督选择泄漏、因果越界。不预设工艺、传感器、步骤、时段数量或固定配方；只依据实际schema、明确提供的领域资料和报告声明审查物理/统计结论。仅对涉及的声明检查闭环非独立、设定值/瞬态、批次伪重复、OPLS数学一致性、置换和恢复比例。数据划分或计数验收不要求根因推断、建模或OPLS。只给有依据的问题，不重复探索，不强制补与用户任务无关的实验。",
        "scope": "只有这里提供的已保存证据被复核；没有重算的事项不能假装重算。",
        "allowed_claim_ids": [
            "summary",
            "scope",
            *[h.id for h in report.hypotheses],
            *[e.id for e in report.evidence],
        ],
        "allowed_evidence_ids": [e.id for e in report.evidence],
        "reference_contract": "claim_id必须是上列单个ID。多项引用用evidence_ids数组；不要把自身输出的引用格式错误归罪于报告。正文用标点列举多项ID是允许的。",
    }


def validate_review(review: ReviewResult, report: ResearchReply):
    known = (
        {h.id for h in report.hypotheses} | {e.id for e in report.evidence} | {"summary", "scope"}
    )
    evidence = {e.id for e in report.evidence}
    for issue in review.issues:
        if issue.claim_id not in known or set(issue.evidence_ids) - evidence:
            return ReviewResult(
                issues=[
                    ReviewIssue(
                        category="evidence",
                        severity="blocking",
                        claim_id="scope",
                        problem="审查包含未定位到当前报告的claim/evidence引用，尚不能接受。",
                        counterevidence=f"Unknown review reference: {issue.claim_id}",
                        suggested_check="提供清楚的结构化证据和报告ID，再审查。",
                    )
                ],
                limitations=["Raw reviewer output retained in review.json for audit."],
            )
    return review


def budget_checkpoint(agent):
    """Deliver saved branch pointers without inventing a merged scientific conclusion."""
    rows = agent._manager.store.list() if agent._manager else []
    saved = [agent._manager.result(r["id"]) for r in rows if r["state"] == "completed"]
    lab = getattr(agent, "_formula_lab", None)
    studies = lab.active() if lab else []
    from .formula_models import FormulaHandoff

    return ResearchReply(
        summary="本轮共享模型预算已耗尽，尚未形成通过审查的主报告。已有文件和分支检查点已保存，可继续本会话。"
        + "".join(
            f"\n- {r['title']}：{r['state']}，分支审查状态 {r.get('review_status') or '未审查'}。"
            for r in saved
        ),
        artifacts=list(
            dict.fromkeys(p for r in saved for p in r.get("report", {}).get("artifacts", []))
        ),
        next_experiments=[
            "继续本会话，读取已保存的分支证据，完成合并和审查；不重复运行已完成的分支。"
        ],
        formula_comparison=FormulaHandoff(
            study_id=studies[-1]["id"],
            status="incomplete",
            reason="模型预算耗尽，尚未完成公式比较交接",
        )
        if studies
        else None,
    )


async def research_with_review(agent, message, folder, policy, budget_client=None, reviewer=None):
    events = agent._workspace.events
    run = agent._workspace.run
    from .strategy_factory import research_strategy

    try:
        report = await agent.research(message, _strategy=research_strategy(policy))
    except Exception:
        if not getattr(budget_client, "budget_exhausted", False):
            raise
        report = budget_checkpoint(agent)
        verified = EvidenceVerifier(run, events).verify(report)
        if getattr(agent, "_formula_lab", None):
            verified["formula_comparison"] = agent._formula_lab.handoff(report.formula_comparison)
        revision = folder / "revisions" / "attempt_00"
        freeze_revision(run, revision, report, verified)
        outcome = {
            "status": "budget_exhausted",
            "revision": 0,
            "verification": verified,
            "review": None,
            "reason": "shared_model_budget",
        }
        write_json(revision / "outcome.json", outcome)
        events.emit("review.end", **outcome)
        return report, outcome
    last_review = None
    attempt = 0
    while True:
        verified = EvidenceVerifier(run, events).verify(report)
        if getattr(agent, "_formula_lab", None):
            verified["formula_comparison"] = agent._formula_lab.handoff(report.formula_comparison)
        revision = folder / "revisions" / f"attempt_{attempt:02d}"
        freeze_revision(run, revision, report, verified)
        if (
            report.intent == "visualization"
            and not report.hypotheses
            and not report.formula_comparison
            and not (getattr(agent, "_formula_lab", None) and agent._formula_lab.active())
        ):
            outcome = {
                "status": "not_required",
                "revision": attempt,
                "verification": verified,
                "review": None,
                "reason": "visualization_only",
            }
            write_json(revision / "outcome.json", outcome)
            events.emit("review.end", **outcome)
            return report, outcome
        if not policy.review_enabled or reviewer is None:
            outcome = {
                "status": "not_reviewed",
                "revision": attempt,
                "verification": verified,
                "review": None,
                "reason": "review_disabled"
                if not policy.review_enabled
                else "review_provider_unavailable",
            }
            write_json(revision / "outcome.json", outcome)
            return report, outcome
        events.emit("review.start", revision=attempt)
        try:
            raw = await reviewer.review(review_bundle(report, verified, events))
            write_json(revision / "review.json", raw.model_dump())
            last_review = validate_review(raw, report)
            if last_review is not raw:
                events.emit(
                    "review.invalid", message="审查引用格式无效，要求审查者修正，不改写科学报告。"
                )
                bundle = review_bundle(report, verified, events)
                bundle["repair_reviewer_output"] = {
                    "invalid_review": raw.model_dump(),
                    "instruction": "仅修复你的审查引用，使用allowed_claim_ids中单个ID；不要要求研究者修正文中正常的多ID列举。",
                }
                raw = await reviewer.review(bundle)
                write_json(revision / "review_repaired.json", raw.model_dump())
                last_review = validate_review(raw, report)
        except Exception as error:  # noqa: BLE001 - untrusted execution/service boundary
            if not isinstance(error, (BudgetExceeded, AdmissionCallCapError)) and not getattr(
                budget_client, "budget_exhausted", False
            ):
                outcome = {
                    "status": "not_reviewed",
                    "revision": attempt,
                    "verification": verified,
                    "review": None,
                    "reason": "review_service_error",
                    "error_type": type(error).__name__,
                }
                write_json(revision / "outcome.json", outcome)
                events.emit("review.end", **outcome)
                return report, outcome
            outcome = {
                "status": "budget_exhausted",
                "revision": attempt,
                "verification": verified,
                "review": None,
                "reason": "shared_model_budget",
            }
            write_json(revision / "outcome.json", outcome)
            events.emit("review.end", **outcome)
            return report, outcome
        if last_review is not raw:
            outcome = {
                "status": "not_reviewed",
                "revision": attempt,
                "verification": verified,
                "review": last_review.model_dump(),
                "reason": "invalid_reviewer_references",
            }
            write_json(revision / "outcome.json", outcome)
            events.emit("review.end", **outcome)
            return report, outcome
        blockers = [i for i in last_review.issues if i.severity == "blocking"]
        status = (
            "passed"
            if not blockers
            else "needs_revision"
            if attempt < policy.max_revisions
            else "revision_limit"
        )
        outcome = {
            "status": status,
            "revision": attempt,
            "verification": verified,
            "review": last_review.model_dump(),
            "reason": "revision_limit" if status == "revision_limit" else None,
        }
        write_json(revision / "outcome.json", outcome)
        events.emit(
            "review.end",
            status=status,
            revision=attempt,
            issues=[i.model_dump() for i in last_review.issues],
        )
        if status != "needs_revision":
            return report, outcome
        attempt += 1
        agent.context["revision_request"] = json.dumps(
            {"revision": attempt, "review": last_review.model_dump()}, ensure_ascii=False
        )
        try:
            report = await agent.revise(
                message, report, last_review.model_dump(), _strategy=research_strategy(policy)
            )
        except Exception as error:
            if not isinstance(error, (BudgetExceeded, AdmissionCallCapError)) and not getattr(
                budget_client, "budget_exhausted", False
            ):
                raise
            outcome["status"] = "budget_exhausted"
            outcome["reason"] = "shared_model_budget"
            lab = getattr(agent, "_formula_lab", None)
            active = lab.active() if lab else []
            if active:
                from .formula_models import FormulaHandoff

                report = report.model_copy(
                    update={
                        "summary": "定向修订未完成，以下保留上一个检查点，不能当作新的选优结论。\n"
                        + report.summary,
                        "formula_comparison": FormulaHandoff(
                            study_id=active[-1]["id"],
                            status="incomplete",
                            reason="Revision model budget exhausted; active comparisons are unfinished",
                        ),
                    }
                )
            write_json(revision / "outcome.json", outcome)
            events.emit("review.end", status="budget_exhausted", revision=attempt)
            return report, outcome
