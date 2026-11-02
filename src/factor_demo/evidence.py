"""Deterministic scope and saved-table verification; never mines numbers from prose."""

import hashlib
import json
import math
from pathlib import Path

import polars as pl
from nooa.strategy_validation import InvariantError

from .autoresearch_agent_v2.database import create_dataset_service
from .live_events import public_artifact_path as artifact_path
from .research_models import ResearchReply


class EvidenceError(InvariantError):
    pass


class EvidenceVerifier:
    def __init__(self, run: Path, events=None, dataset_ref=None):
        self.run = run.resolve()
        self.events = events
        self.dataset_ref = dataset_ref
        self.registry = None
        root = (
            events.root
            if events
            else self.run.parent.parent
            if (self.run / "branch.json").is_file()
            else self.run
        )
        self.root = root
        if create_dataset_service(root).is_initialized():
            from .datasets import DatasetRegistry

            self.registry = DatasetRegistry(root)
            owner = (
                events.agent_id
                if events
                else self.run.name
                if (self.run / "branch.json").is_file()
                else "main"
            )
            if self.dataset_ref is None:
                self.dataset_ref = self.registry.scope(owner)["current"]
            if owner != "main":
                bound = self.registry.scope(owner)["root"]
                ancestry = self.registry.lineage(self.dataset_ref)
                if bound not in {d["id"] for d in ancestry["nodes"]}:
                    self.fail("Evidence dataset is outside this agent's assigned scope")

    def fail(self, message):
        if self.events:
            self.events.emit("validation.issue", message=message)
        raise EvidenceError(message)

    def scope(self):
        if self.registry is not None and self.dataset_ref:
            from .dataset_adapter import canonical_index
            from .dataset_models import DatasetSchema

            doc = self.registry.get(self.dataset_ref)
            schema = DatasetSchema.model_validate(doc["schema"])
            root = self.registry.path(self.dataset_ref, verify=True)
            runs = canonical_index(root, schema)
            if schema.split_column and set(runs["split"]) & set(schema.excluded_splits):
                self.fail("Held-out entries are outside discovery scope")
            if schema.fixed_recipe and (
                "recipe_id" not in runs.columns or runs["recipe_id"].n_unique() != 1
            ):
                self.fail("All discovery runs must share one fixed recipe")
            return runs
        public = self.run / "input"
        if not (public / "runs.parquet").is_file():
            self.fail("Public runs.parquet is missing")
        runs = pl.read_parquet(public / "runs.parquet")
        if "split" not in runs.columns or set(runs["split"]) - {"train", "dev"}:
            self.fail("Only public train/dev runs are permitted")
        if "recipe_id" not in runs.columns or runs["recipe_id"].n_unique() != 1:
            self.fail("All discovery runs must share one fixed recipe")
        config = json.loads((self.run / "session.json").read_text())
        for name in ["recipe.json", "runs.parquet"]:
            expected = config.get("public_sha256", {}).get(name)
            if expected and hashlib.sha256((public / name).read_bytes()).hexdigest() != expected:
                self.fail(f"Public scope changed: {name}")
        return runs

    def verify(self, report: ResearchReply):
        if self.registry is None and create_dataset_service(self.root).is_initialized():
            return EvidenceVerifier(
                self.run, self.events, report.dataset_ref or self.dataset_ref
            ).verify(report)
        if report.dataset_ref and report.dataset_ref != self.dataset_ref:
            return EvidenceVerifier(self.run, self.events, report.dataset_ref).verify(report)
        runs = self.scope()
        public_ids = set(runs["run_id"])
        recipe = str(runs["recipe_id"][0]) if "recipe_id" in runs.columns and runs.height else None
        paths = set(report.artifacts)
        for h in report.hypotheses:
            paths.update(h.evidence_files)
        for path in paths:
            try:
                artifact_path(self.run, path)
            except ValueError:
                self.fail(f"Evidence artifact is missing or outside artifacts/: {path}")
        ids = [e.id for e in report.evidence]
        if len(ids) != len(set(ids)):
            self.fail("Evidence IDs must be unique")
        checks = []
        for e in report.evidence:
            scoped = self
            if e.dataset_ref and e.dataset_ref != self.dataset_ref:
                scoped = EvidenceVerifier(self.run, self.events, e.dataset_ref)
                evidence_runs = scoped.scope()
                evidence_ids = set(evidence_runs["run_id"])
                evidence_recipes = (
                    set(evidence_runs["recipe_id"])
                    if "recipe_id" in evidence_runs.columns
                    else set()
                )
            else:
                evidence_ids = public_ids
                evidence_recipes = set(runs["recipe_id"]) if "recipe_id" in runs.columns else set()
            if not e.unit.strip():
                self.fail(f"{e.id}: specify units (use 1 for dimensionless or runs for counts)")
            try:
                path = artifact_path(self.run, e.artifact)
            except ValueError:
                self.fail(f"{e.id}: artifact must exist under artifacts/")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if e.sha256 and e.sha256 != sha:
                self.fail(f"{e.id}: evidence file hash changed")
            if set(e.run_ids) - evidence_ids:
                self.fail(f"{e.id}: contains a run outside the public discovery set")
            if e.recipe_id and e.recipe_id not in evidence_recipes:
                self.fail(f"{e.id}: recipe mismatch")
            try:
                if path.suffix.lower() == ".csv":
                    frame = pl.read_csv(path)
                elif path.suffix.lower() == ".parquet":
                    frame = pl.read_parquet(path)
                else:
                    self.fail(f"{e.id}: numerical evidence must be CSV or Parquet")
                identity_column = "run_id"
                if scoped.registry is not None and scoped.dataset_ref:
                    identity_column = scoped.registry.get(scoped.dataset_ref)["schema"]["id_column"]
                if (
                    identity_column in frame.columns
                    and set(frame[identity_column].cast(pl.String).drop_nulls()) - evidence_ids
                ):
                    self.fail(f"{e.id}: table contains withheld/unknown runs")
                if (
                    "recipe_id" in frame.columns
                    and set(frame["recipe_id"].drop_nulls()) - evidence_recipes
                ):
                    self.fail(f"{e.id}: table includes another recipe")
                for key, value in e.where.items():
                    frame = frame.filter(pl.col(key) == value)
                if frame.is_empty():
                    self.fail(f"{e.id}: row selector matches no evidence")
                if e.statistic == "count":
                    actual = frame.height
                else:
                    series = frame[e.column]
                    if e.statistic == "value":
                        if len(series) != 1:
                            self.fail(f"{e.id}: value requires exactly one selected row")
                        actual = series[0]
                    else:
                        actual = getattr(series, e.statistic)()
            except EvidenceError:
                raise
            except Exception as error:  # noqa: BLE001 - return a model-correctable evidence error
                self.fail(f"{e.id}: cannot evaluate table selection ({type(error).__name__})")
            if (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and isinstance(e.value, (int, float))
                and not isinstance(e.value, bool)
            ):
                match = (
                    math.isfinite(actual)
                    and math.isfinite(e.value)
                    and math.isclose(actual, e.value, rel_tol=1e-6, abs_tol=1e-8)
                )
            else:
                match = type(actual) == type(e.value) and actual == e.value
            if not match:
                self.fail(f"{e.id}: claimed {e.value!r}, saved-table calculation is {actual!r}")
            from .execution_provenance import artifact_proof
            from .live_events import branch_path
            from .trusted_evidence import verify_proof

            owner_run = self.run
            local_name = e.artifact
            parts = Path(e.artifact).parts
            if len(parts) > 2 and parts[0] == "branches":
                owner_run = branch_path(self.run, parts[1])
                local_name = str(Path(*parts[2:]))
            try:
                if e.proof_id:
                    local_evidence = e.model_copy(update={"artifact": local_name})
                    proof = verify_proof(owner_run, local_evidence)
                else:
                    proof = artifact_proof(owner_run, local_name, e.execution_ref)
                    if proof["dataset_ref"] != (e.dataset_ref or self.dataset_ref):
                        raise ValueError("Execution and evidence dataset scopes differ")
                    proof = {
                        "grade": "exploratory_reproducible",
                        "execution_ref": proof["execution_ref"],
                        "source_sha256": proof["source_sha256"],
                    }
            except (ValueError, OSError) as error:
                self.fail(f"{e.id}: {error}")
            checks.append(
                {
                    "id": e.id,
                    "claim": e.claim,
                    "artifact": e.artifact,
                    "column": e.column,
                    "where": e.where,
                    "statistic": e.statistic,
                    "actual": actual,
                    "unit": e.unit,
                    "sha256": sha,
                    "dataset_ref": e.dataset_ref or self.dataset_ref,
                    "provenance": proof,
                }
            )
        for h in report.hypotheses:
            if set(h.evidence_ids) - set(ids):
                self.fail(f"{h.id}: unknown evidence_ids")
            if h.status in {"supported", "weak"} and not h.evidence_ids:
                self.fail(f"{h.id}: supporting claims need structured evidence_ids")
        verified_by_id = {c["id"]: c for c in checks}
        hypothesis_checks = {}
        import operator

        comparisons = {
            "lt": operator.lt,
            "le": operator.le,
            "gt": operator.gt,
            "ge": operator.ge,
            "eq": operator.eq,
            "abs_le": lambda a, b: abs(a) <= b,
            "abs_gt": lambda a, b: abs(a) > b,
        }
        for h in report.hypotheses:
            completed = []
            for check in h.checks:
                if check.status != "completed":
                    continue
                if check.evidence_id not in verified_by_id or (
                    check.subtract_evidence_id and check.subtract_evidence_id not in verified_by_id
                ):
                    self.fail(f"{h.id}: check refers to unknown evidence")
                left = verified_by_id[check.evidence_id]
                related = [left]
                value = left["actual"]
                if check.subtract_evidence_id:
                    right = verified_by_id[check.subtract_evidence_id]
                    related.append(right)
                    if left["unit"] != right["unit"]:
                        self.fail(f"{h.id}: check subtracts different units")
                    value -= right["actual"]
                if any(
                    r["provenance"]["grade"] not in {"raw_recomputed", "formula_recomputed"}
                    for r in related
                ):
                    self.fail(f"{h.id}: completed checks require independently recomputed evidence")
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    self.fail(f"{h.id}: check needs a finite numerical value")
                passed = comparisons[check.operator](value, check.threshold)
                completed.append(
                    {
                        "description": check.description,
                        "value": value,
                        "operator": check.operator,
                        "threshold": check.threshold,
                        "passed": passed,
                    }
                )
            if h.status == "supported" and (
                not completed or not all(c["passed"] for c in completed)
            ):
                self.fail(
                    f"{h.id}: supported requires completed, recomputed checks; otherwise use weak/unresolved"
                )
            if h.status == "supported":
                for evidence_id in h.evidence_ids:
                    grade = verified_by_id[evidence_id]["provenance"]["grade"]
                    if grade not in {"raw_recomputed", "formula_recomputed"}:
                        self.fail(
                            f"{h.id}: supported may only cite recomputed evidence "
                            f"(rejected exploratory {evidence_id})"
                        )
                for check in h.checks:
                    if check.status != "completed":
                        continue
                    if check.evidence_id not in set(h.evidence_ids):
                        self.fail(
                            f"{h.id}: supported checks must reference hypothesis evidence_ids"
                        )
                    if check.subtract_evidence_id and check.subtract_evidence_id not in set(
                        h.evidence_ids
                    ):
                        self.fail(
                            f"{h.id}: supported checks must reference hypothesis evidence_ids"
                        )
            if h.status == "rejected" and (not completed or all(c["passed"] for c in completed)):
                self.fail(
                    f"{h.id}: rejected requires a completed failed check; otherwise use unresolved"
                )
            hypothesis_checks[h.id] = completed
        if self.events:
            self.events.emit("validation.passed", checks=len(checks), files=len(paths))
        return {
            "recipe_id": recipe,
            "dataset_ref": self.dataset_ref,
            "public_runs": len(public_ids),
            "checks": checks,
            "hypothesis_checks": hypothesis_checks,
            "hypotheses": len(report.hypotheses),
            "limits": "Checks verify public scope and saved-table references, not the correctness of arbitrary transformations, sensor calibration or causal truth.",
        }


def research_precondition(agent, call):
    workspace = getattr(agent, "_workspace", None)
    if workspace is not None and call.method_name in {"research", "revise"}:
        EvidenceVerifier(workspace.run, workspace.events).scope()


def research_postcondition(agent, result, call):
    workspace = getattr(agent, "_workspace", None)
    if (
        workspace is not None
        and isinstance(result, ResearchReply)
        and call.method_name in {"research", "revise"}
    ):
        datasets = getattr(agent, "_datasets", None)
        if datasets is not None and result.dataset_ref is None:
            result.dataset_ref = datasets.current()["id"]
        if result.intent == "visualization" and (
            result.hypotheses
            or result.formula_comparison
            or (getattr(agent, "_formula_lab", None) and agent._formula_lab.active())
        ):
            raise EvidenceError(
                "Visualization-only delivery cannot include hypotheses or active formula selection; use report/comparison intent"
            )
        manager = getattr(agent, "_manager", None)
        if manager is not None and any(
            row["mission"] == manager.mission
            and row["state"] not in {"completed", "cancelled", "failed", "interrupted"}
            for row in manager.store.list()
            if row["parent_id"] == manager.identity
        ):
            raise EvidenceError(
                "Research branches are still active; wait for their results or cancel them explicitly before the final checkpoint"
            )
        agent._last_verification = EvidenceVerifier(workspace.run, workspace.events).verify(result)
        lab = getattr(agent, "_formula_lab", None)
        if lab is not None:
            try:
                agent._last_verification["formula_comparison"] = lab.handoff(
                    result.formula_comparison
                )
            except ValueError as error:
                raise EvidenceError(str(error)) from error


def snapshot_ignore(directory, names):
    return [
        name for name in names if name == "__pycache__" or (Path(directory) / name).is_symlink()
    ]
