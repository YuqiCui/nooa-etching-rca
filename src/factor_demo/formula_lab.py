"""Host-owned formula registry, grouped validation, evaluation and replay gate.

Scientific exploration remains unrestricted. Only comparisons entered in this lab
receive the reproducible selection contract; scores supplied by an agent are never
accepted. SQLite serializes short state changes across branch hosts.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import numpy as np
import polars as pl
import psutil
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from .formula_models import FormulaHandoff, FormulaPlan, FormulaSpec
from .live_events import artifact_path
from .research_workspace import PythonWorkspace

MODEL_PARAMS = {
    "logistic": {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"},
    "random_forest": {
        "n_estimators": 200,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": 2,
    },
    "extra_trees": {
        "n_estimators": 200,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": 2,
    },
    "hist_gradient_boosting": {"max_iter": 100, "max_leaf_nodes": 15, "l2_regularization": 1.0},
}
LIMITS = (
    "仅在本研究已登记且未排除的候选、固定模型和开发集上选优；开发集参与多轮选择，"
    "不是独立最终测试。lot交叉验证不能消除时段混杂；分类分数不能证明物理根因。"
    "任意Python公式的文字描述与代码语义仍需专家核对，隔离不能防止人为硬编码标签。"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def environment():
    return {
        "python": platform.python_version(),
        **{
            n: importlib.metadata.version(n)
            for n in ["numpy", "polars", "scipy", "scikit-learn", "joblib", "cloudpickle"]
        },
        "runner_sha256": digest(Path(__file__).with_name("formula_runner.py")),
        "feature_runner_sha256": digest(Path(__file__).with_name("feature_runner.py")),
    }


class FormulaLab:
    def __init__(self, run: Path, events=None):
        self.run = run.resolve()
        self.events = events
        self.root = events.root if events else self.run
        self.owner = events.agent_id if events else "main"
        self.base = self.root / ".formula-lab"
        from .study_store import StudyStore

        self.store = StudyStore(self.base)
        self.data_session = None
        # Turn-local, deliberately absent from the durable study registry/snapshot.
        self._active_studies = set()

    def dataset_ref(self, study=None):
        if study is not None:
            return study["plan"].get("dataset_ref")
        return self.data_session.current()["id"] if self.data_session else None

    def input_path(self, study=None, dataset_ref=None):
        identity = dataset_ref or self.dataset_ref(study)
        if identity:
            from .datasets import DatasetRegistry

            if self.data_session:
                self.data_session.authorize(identity)
            return DatasetRegistry(self.root).path(identity)
        return self.root / "input"

    def data_schema(self, study=None, dataset_ref=None):
        from .dataset_adapter import detect_schema

        return (
            detect_schema(self.input_path(study, dataset_ref))
            if not (dataset_ref or self.dataset_ref(study))
            else self._manifest_schema(dataset_ref or self.dataset_ref(study))
        )

    def _manifest_schema(self, identity):
        from .dataset_models import DatasetSchema
        from .datasets import DatasetRegistry

        return DatasetSchema.model_validate(DatasetRegistry(self.root).get(identity)["schema"])

    def runs(self, study=None, dataset_ref=None):
        from .dataset_adapter import canonical_index

        return canonical_index(
            self.input_path(study, dataset_ref), self.data_schema(study, dataset_ref)
        )

    def begin_turn(self):
        self._active_studies.clear()

    def begin(self, identity):
        study = self.get(identity)
        self._active_studies.add(identity)
        return self.public(study)

    def active(self):
        return [s for s in self.catalog() if s["id"] in self._active_studies]

    def transaction(self, write=True):
        return self.store.transaction(write)

    def get(self, identity, transaction=None):
        return self.store.get(identity, transaction)

    def save(self, transaction, study):
        self.store.save(transaction, study)

    def emit(self, kind, study_id, **data):
        if self.events:
            self.events.emit("formula." + kind, study_id=study_id, **data)

    def catalog(self):
        return self.store.catalog()

    def public(self, study):
        from .study_store import public_study

        return public_study(study)

    def bound(self):
        return [s for s in self.catalog() if self.owner in s["plan"]["participants"]]

    def context_summary(self):
        recent = self.store.recent(5)
        chosen = {s["id"]: s for s in recent}
        chosen.update({i: self.get(i) for i in self._active_studies})
        return json.dumps(
            {
                "formula_studies": [
                    {
                        "id": s["id"],
                        "title": s["plan"]["title"],
                        "labels": s["plan"]["labels"],
                        "versions": [f["ref"] for f in s["formulas"][-8:]],
                        "selection_id": s["selection_id"],
                        "required_this_turn": s["id"] in self._active_studies,
                        "incomplete_trials": [
                            t["id"] for t in s["trials"][-8:] if t["status"] != "succeeded"
                        ],
                    }
                    for s in chosen.values()
                ],
                "contract": "公式比较按本轮任务需要启用。普通画图、数据查看、解释旧结果、自由探索可直接run_code/read，不启用选优。需要比较或确认最佳公式时create_formula_study或begin_formula_comparison；登记/修改/执行比较也自动启用。本轮启用后交接必须含formula_comparison，未完成则incomplete+reason。历史研究不会自动要求本轮继续比较。",
            },
            ensure_ascii=False,
        )

    def verify_input(self, study=None, dataset_ref=None, require_environment=True):
        identity = dataset_ref or self.dataset_ref(study)
        if identity:
            from .datasets import DatasetRegistry

            registry = DatasetRegistry(self.root)
            registry.path(identity, verify=True)
            fingerprint = registry.get(identity)["content_sha256"]
        else:
            public = self.root / "input"
            expected = json.loads((self.root / "session.json").read_text())["public_sha256"]
            actual = {
                str(p.relative_to(public)): digest(p) for p in public.rglob("*") if p.is_file()
            }
            if expected != actual:
                raise ValueError(
                    "Public input fingerprint changed; restore data before comparing formulas"
                )
            fingerprint = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
        if study and (
            study["plan"]["input_sha256"] != fingerprint
            or (require_environment and study["plan"]["environment"] != environment())
        ):
            raise ValueError("Frozen data/software protocol changed; create a new study")
        return fingerprint

    def create(self, spec: FormulaPlan):
        dataset_ref = spec.dataset_ref or self.dataset_ref()
        if dataset_ref and self.data_session:
            self.data_session.authorize(dataset_ref)
        fingerprint = self.verify_input(dataset_ref=dataset_ref)
        if len(set(spec.labels)) != 2 or len(set(spec.models)) != len(spec.models):
            raise ValueError("Choose two different labels and unique models")
        source_runs = self.runs(dataset_ref=dataset_ref)
        if {"label", "batch_id", "split"} - set(source_runs.columns):
            raise ValueError(
                "Declare label, independent group and train/dev split roles before formal comparison"
            )
        runs = (
            self.runs(dataset_ref=dataset_ref)
            .filter(~pl.col("is_control") & pl.col("label").is_in(spec.labels))
            .sort("run_id")
        )
        if set(runs["label"]) != set(spec.labels) or set(runs["split"]) - {"train", "dev"}:
            raise ValueError("Need both labels in public train/dev only")
        if ("recipe_id" in runs.columns and runs["recipe_id"].n_unique() != 1) or runs[
            "run_id"
        ].n_unique() != runs.height:
            raise ValueError("Need unique runs from one fixed recipe")
        train, dev = (runs.filter(pl.col("split") == split) for split in ["train", "dev"])
        if set(train["batch_id"]) & set(dev["batch_id"]):
            raise ValueError("A lot crosses train/dev boundaries")
        dev_lots = {
            label: dev.filter(pl.col("label") == label)["batch_id"].n_unique()
            for label in spec.labels
        }
        if any(n < spec.min_dev_lots_per_label for n in dev_lots.values()):
            raise ValueError(
                "Too few independent development lots per label; use a larger cohort or explicitly declare a limited protocol"
            )
        if any(set(frame["label"]) != set(spec.labels) for frame in [train, dev]):
            raise ValueError("Both labels must occur in train and dev")
        if train["batch_id"].n_unique() < spec.folds:
            raise ValueError("Too few independent training lots for requested folds")
        folds = []
        splitter = StratifiedGroupKFold(n_splits=spec.folds, shuffle=True, random_state=spec.seed)
        for i, (a, b) in enumerate(
            splitter.split(
                np.zeros((train.height, 1)), train["label"].to_numpy(), train["batch_id"].to_numpy()
            )
        ):
            left, right = train[a.tolist()], train[b.tolist()]
            if any(set(f["label"]) != set(spec.labels) for f in [left, right]):
                raise ValueError("A grouped fold lacks a label; use fewer folds or more lots")
            assert not set(left["batch_id"]) & set(right["batch_id"])
            folds.append(
                {
                    "id": f"cv-{i}",
                    "train": left["run_id"].to_list(),
                    "valid": right["run_id"].to_list(),
                }
            )
        identity = "study-" + uuid.uuid4().hex[:12]
        study = {
            "id": identity,
            "owner": self.owner,
            "created": time.time(),
            "revision": 0,
            "plan": {
                **spec.model_dump(),
                "dev_lots_by_label": dev_lots,
                "input_sha256": fingerprint,
                "environment": environment(),
                "recipe_id": runs["recipe_id"][0] if "recipe_id" in runs.columns else None,
                "dataset_ref": dataset_ref,
                "extraction_protocol": "isolated-per-sample-and-fold-fit-v2",
                "group": "batch_id",
                "train": train["run_id"].to_list(),
                "dev": dev["run_id"].to_list(),
                "fold_manifest": folds,
                "model_params": {m: MODEL_PARAMS[m] for m in spec.models},
                "participants": [self.owner],
                "preprocessing": "fold-fit median imputation; logistic only: fold-fit standard scaling",
                "ranking": "grouped CV mean descending, CV std ascending, dev metric descending, fewer formulas, stable result id",
            },
            "formulas": [],
            "sets": [],
            "trials": [],
            "selections": [],
            "replays": [],
            "selection_id": None,
            "limits": LIMITS,
        }
        with self.transaction() as transaction:
            self.save(transaction, study)
        self._active_studies.add(identity)
        self.export(study)
        self.emit("study.created", identity, title=spec.title)
        return self.public(study)

    def register(self, identity: str, spec: FormulaSpec):
        study = self.get(identity)
        self.verify_input(study)
        source = artifact_path(self.run, spec.source).read_text()
        tree = ast.parse(source)
        functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        required = {"extract"} if spec.kind == "per_sample" else {"fit", "transform"}
        if required - functions:
            raise ValueError("Source must define " + ", ".join(sorted(required)))
        if len(set(spec.sensors)) != len(spec.sensors) or len(set(spec.steps)) != len(spec.steps):
            raise ValueError("Sensors and steps must be unique")
        from .feature_execution import FeatureExecutor

        FeatureExecutor(self).validate(study, spec.model_dump())
        json.dumps(spec.params, allow_nan=False)
        with self.transaction() as transaction:
            study = self.get(identity, transaction)
            version = 1 + max(
                (f["version"] for f in study["formulas"] if f["name"] == spec.name), default=0
            )
            if version > 1 and not spec.change_reason.strip():
                raise ValueError(
                    "Revised formula requires change_reason; previous code/results are retained"
                )
            ref = f"{spec.name}@{version}"
            folder = self.base / identity / "formulas" / ref
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "source.py").write_text(source)
            row = {
                **spec.model_dump(),
                "ref": ref,
                "version": version,
                "owner": self.owner,
                "sha256": digest(folder / "source.py"),
            }
            dump(folder / "spec.json", row)
            row["spec_sha256"] = digest(folder / "spec.json")
            study["formulas"].append(row)
            study["sets"].append(
                {"id": "set-" + uuid.uuid4().hex[:10], "formulas": [ref], "excluded": None}
            )
            study["revision"] += 1
            study["selection_id"] = None
            if self.owner not in study["plan"]["participants"]:
                study["plan"]["participants"].append(self.owner)
            self.save(transaction, study)
        self._active_studies.add(identity)
        self.export(study)
        self.emit("formula.registered", identity, ref=ref)
        return row

    def feature_set(self, identity, refs):
        refs = sorted(set(refs))
        with self.transaction() as transaction:
            study = self.get(identity, transaction)
            if not refs or set(refs) - {f["ref"] for f in study["formulas"]}:
                raise ValueError("Choose registered formula versions")
            existing = next((s for s in study["sets"] if s["formulas"] == refs), None)
            if existing:
                return existing
            item = {"id": "set-" + uuid.uuid4().hex[:10], "formulas": refs, "excluded": None}
            study["sets"].append(item)
            study["revision"] += 1
            study["selection_id"] = None
            self.save(transaction, study)
        self._active_studies.add(identity)
        self.emit("set.created", identity, set_id=item["id"])
        return item

    def retire(self, identity, set_id, reason):
        if not reason.strip():
            raise ValueError("Excluding a comparison requires an explicit reason")
        with self.transaction() as transaction:
            study = self.get(identity, transaction)
            item = next(s for s in study["sets"] if s["id"] == set_id)
            item["excluded"] = {"reason": reason, "owner": self.owner, "time": time.time()}
            study["selection_id"] = None
            study["revision"] += 1
            self.save(transaction, study)
        self._active_studies.add(identity)
        self.export(study)
        self.emit("set.excluded", identity, set_id=set_id, reason=reason)
        return item

    async def worker(self, job, code):
        worker = PythonWorkspace(job, read_scope_root=self.root)
        try:
            result = await worker.execute(code, 300)
            if not result["ok"]:
                raise ValueError("Scientific job failed: " + result["output"][-3000:])
        finally:
            await worker.close()

    def new_job(self, identity, kind):
        job = self.base / identity / "jobs" / (kind + "-" + uuid.uuid4().hex)
        (job / "workspace").mkdir(parents=True)
        (job / "input").mkdir()
        shutil.copyfile(Path(__file__).with_name("formula_runner.py"), job / "workspace/runner.py")
        return job

    async def matrix(self, study, item, replay=False):
        from .feature_execution import FeatureExecutor

        return await FeatureExecutor(self).compute(study, item, replay=replay)

    async def model_result(self, study, item, matrix, model):
        plan = study["plan"]
        job = self.new_job(study["id"], "evaluate")
        matrix.write_parquet(job / "input/matrix.parquet")
        dump(
            job / "input/job.json",
            {
                "features": item["formulas"],
                "folds": plan["fold_manifest"],
                "train": plan["train"],
                "dev": plan["dev"],
                "model": model,
                "seed": plan["seed"],
                "params": plan["model_params"][model],
            },
        )
        await self.worker(job, "import runpy\nrunpy.run_path('runner.py')['evaluate']()")
        predictions = pl.read_parquet(job / "workspace/predictions.parquet")
        scores = self.score(study, predictions)
        return scores, job

    def score(self, study, predictions):
        plan = study["plan"]
        runs = self.runs(study).select("run_id", "label")
        metric = {
            "balanced_accuracy": balanced_accuracy_score,
            "accuracy": accuracy_score,
            "f1_macro": lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        }[plan["metric"]]
        values = {}
        expected = [*plan["fold_manifest"], {"id": "dev", "valid": plan["dev"]}]
        if set(predictions["fold"]) != {f["id"] for f in expected}:
            raise ValueError("Prediction fold coverage differs from frozen plan")
        for fold in expected:
            selected = predictions.filter(pl.col("fold") == fold["id"])
            if selected.height != len(fold["valid"]) or set(selected["run_id"]) != set(
                fold["valid"]
            ):
                raise ValueError("Missing/duplicate prediction runs")
            if set(selected["prediction"]) - set(plan["labels"]):
                raise ValueError("Unknown predicted label")
            selected = selected.join(runs, on="run_id", validate="1:1")
            values[fold["id"]] = float(
                metric(selected["label"].to_numpy(), selected["prediction"].to_numpy())
            )
        cv = [values[f["id"]] for f in plan["fold_manifest"]]
        return {
            "dev_score": values["dev"],
            "cv_mean": float(np.mean(cv)),
            "cv_std": float(np.std(cv)),
            "fold_scores": values,
        }

    def update_trial(self, identity, trial_id, **updates):
        with self.transaction() as transaction:
            study = self.get(identity, transaction)
            row = next(t for t in study["trials"] if t["id"] == trial_id)
            row.update(updates)
            study["selection_id"] = None
            self.save(transaction, study)
        self.emit("trial.updated", identity, trial_id=trial_id, status=row["status"])

    async def evaluate(self, identity, refs):
        self.begin(identity)
        item = self.feature_set(identity, refs)
        study = self.get(identity)
        self.verify_input(study)
        if item["excluded"]:
            raise ValueError(
                "This set was explicitly excluded; create a new formula version to reconsider"
            )
        # Claim the entire feature set before any await. Another process cannot duplicate it.
        with self.transaction() as transaction:
            study = self.get(identity, transaction)
            previous = [t for t in study["trials"] if t["set_id"] == item["id"]]
            for row in previous:
                if row["status"] == "running":
                    try:
                        process = psutil.Process(row["pid"])
                        if process.create_time() != row["process_created"]:
                            raise psutil.NoSuchProcess(row["pid"])
                    except psutil.NoSuchProcess:
                        row.update(
                            status="interrupted", error="Host process exited", finished=time.time()
                        )
                    else:
                        raise ValueError("Comparison is already running; read its saved state")
            completed = {t["model"] for t in previous if t["status"] == "succeeded"}
            pending = [m for m in study["plan"]["models"] if m not in completed]
            if not pending:
                return self.public(study)
            new = [
                {
                    "id": "trial-" + uuid.uuid4().hex[:12],
                    "set_id": item["id"],
                    "formulas": item["formulas"],
                    "model": m,
                    "status": "running",
                    "owner": self.owner,
                    "pid": os.getpid(),
                    "process_created": psutil.Process().create_time(),
                    "started": time.time(),
                }
                for m in pending
            ]
            study["trials"].extend(new)
            study["selection_id"] = None
            self.save(transaction, study)
        self.emit("evaluation.start", identity, set_id=item["id"])
        try:
            matrix = await self.matrix(study, item)
            for row in new:
                try:
                    scores, job = await self.model_result(study, item, matrix, row["model"])
                    folder = self.base / identity / "results" / row["id"]
                    folder.mkdir(parents=True)
                    matrix.write_parquet(folder / "matrix.parquet")
                    for filename in ["predictions.parquet", "model.joblib", "model_details.json"]:
                        shutil.copyfile(job / "workspace" / filename, folder / filename)
                    feature_manifest = []
                    for ref in item["formulas"]:
                        definition = next(f for f in study["formulas"] if f["ref"] == ref)
                        feature_manifest.append(
                            {
                                k: definition[k]
                                for k in [
                                    "ref",
                                    "kind",
                                    "sha256",
                                    "params",
                                    "sensors",
                                    "steps",
                                    "expression",
                                    "unit",
                                ]
                            }
                        )
                        if definition.get("kind", "per_sample") == "learned":
                            target = folder / "operators" / ref
                            target.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(
                                self.base / identity / "features" / ref / "dev.state.joblib",
                                target / "state.joblib",
                            )
                            shutil.copyfile(
                                self.base / identity / "formulas" / ref / "source.py",
                                target / "source.py",
                            )
                    dump(
                        folder / "feature_manifest.json",
                        {
                            "dataset_ref": study["plan"].get("dataset_ref"),
                            "training_ids": study["plan"]["train"],
                            "operators": feature_manifest,
                        },
                    )
                    hashes = {
                        str(p.relative_to(folder)): digest(p)
                        for p in folder.rglob("*")
                        if p.is_file()
                    }
                    self.update_trial(
                        identity,
                        row["id"],
                        status="succeeded",
                        finished=time.time(),
                        **scores,
                        hashes=hashes,
                        model_details=json.loads((folder / "model_details.json").read_text()),
                    )
                except Exception as error:  # noqa: BLE001 - persist scientific failures for retry
                    self.update_trial(
                        identity,
                        row["id"],
                        status="failed",
                        error=str(error)[-3500:],
                        finished=time.time(),
                    )
        except BaseException as error:
            for row in self.get(identity)["trials"]:
                if row["id"] in {r["id"] for r in new} and row["status"] == "running":
                    self.update_trial(
                        identity,
                        row["id"],
                        status="interrupted"
                        if isinstance(error, asyncio.CancelledError)
                        else "failed",
                        error=f"{type(error).__name__}: {str(error)[-3000:]}",
                        finished=time.time(),
                    )
            if isinstance(error, asyncio.CancelledError):
                raise
        result = self.get(identity)
        self.export(result)
        return self.public(result)

    def ranking(self, study):
        active = {s["id"] for s in study["sets"] if not s["excluded"]}
        rows = [t for t in study["trials"] if t["status"] == "succeeded" and t["set_id"] in active]
        return sorted(
            rows,
            key=lambda r: (
                -r["cv_mean"],
                r["cv_std"],
                -r["dev_score"],
                len(r["formulas"]),
                r["id"],
            ),
        )

    def complete(self, study):
        active = [s for s in study["sets"] if not s["excluded"]]
        if not active:
            raise ValueError("No active formula candidates")
        missing = [
            (s["id"], m)
            for s in active
            for m in study["plan"]["models"]
            if not any(
                t["set_id"] == s["id"] and t["model"] == m and t["status"] == "succeeded"
                for t in study["trials"]
            )
        ]
        if missing or any(t["status"] == "running" for t in study["trials"]):
            raise ValueError(
                f"Comparison incomplete: {missing}; finish or explicitly exclude sets with a reason"
            )

    def verify_result(self, study, row):
        folder = self.base / study["id"] / "results" / row["id"]
        for name, expected in row["hashes"].items():
            if digest(folder / name) != expected:
                raise ValueError("Canonical result artifact changed")
        if self.score(study, pl.read_parquet(folder / "predictions.parquet")) != {
            k: row[k] for k in ["dev_score", "cv_mean", "cv_std", "fold_scores"]
        }:
            raise ValueError("Saved prediction metrics disagree")
        return folder

    async def freeze(self, identity):
        self.begin(identity)
        attempt = {
            "id": "replay-" + uuid.uuid4().hex[:12],
            "owner": self.owner,
            "started": time.time(),
        }
        try:
            result = await self._freeze(identity)
            attempt.update(status="passed", selection_id=result["selection_id"])
            return result
        except BaseException as error:
            attempt.update(
                status="interrupted" if isinstance(error, asyncio.CancelledError) else "failed",
                error=f"{type(error).__name__}: {str(error)[-3000:]}",
            )
            raise
        finally:
            attempt["finished"] = time.time()
            with self.transaction() as transaction:
                current = self.get(identity, transaction)
                current["replays"].append(attempt)
                self.save(transaction, current)
            self.export(current)
            self.emit("replay.finished", identity, status=attempt["status"])

    async def _freeze(self, identity):
        study = self.get(identity)
        self.verify_input(study)
        self.complete(study)
        ranked = self.ranking(study)
        for row in ranked:
            self.verify_result(study, row)
        best = ranked[0]
        item = next(s for s in study["sets"] if s["id"] == best["set_id"])
        self.emit("replay.start", identity, trial_id=best["id"])
        matrix = await self.matrix(study, item, replay=True)
        saved = pl.read_parquet(self.base / identity / "results" / best["id"] / "matrix.parquet")
        if matrix.columns != saved.columns or not matrix.equals(saved):
            raise ValueError("Best formula replay changed feature values; selection refused")
        scores, job = await self.model_result(study, item, matrix, best["model"])
        old_predictions = pl.read_parquet(
            self.base / identity / "results" / best["id"] / "predictions.parquet"
        )
        predictions = pl.read_parquet(job / "workspace/predictions.parquet")
        if not predictions.equals(old_predictions) or any(scores[k] != best[k] for k in scores):
            raise ValueError("Best model replay changed predictions/scores; selection refused")
        selection = {
            "id": "selection-" + uuid.uuid4().hex[:12],
            "revision": study["revision"],
            "winner": best["id"],
            "ranking": [r["id"] for r in ranked],
            "replay": {
                "status": "passed",
                "time": time.time(),
                "features_equal": True,
                "predictions_equal": True,
            },
            "created": time.time(),
        }
        with self.transaction() as transaction:
            current = self.get(identity, transaction)
            if current["revision"] != study["revision"] or current["trials"] != study["trials"]:
                raise ValueError("Study changed during replay; freeze again")
            current["selections"].append(selection)
            current["selection_id"] = selection["id"]
            self.save(transaction, current)
        self.export(current)
        self.emit("selection.frozen", identity, selection_id=selection["id"], winner=best["id"])
        return {
            "study_id": identity,
            "status": "selected",
            "selection_id": selection["id"],
            "winner": best,
            "limits": LIMITS,
        }

    def export(self, study):
        import fcntl

        self.base.mkdir(parents=True, exist_ok=True)
        with (self.base / "export.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._export(self.get(study["id"]))

    def _export(self, study):
        folder = self.root / "workspace/artifacts/formula-lab" / study["id"]
        public_root = (self.root / "workspace/artifacts").resolve()
        if not folder.resolve().is_relative_to(public_root):
            raise ValueError("Formula export escaped public artifacts")
        folder.mkdir(parents=True, exist_ok=True)

        def publish(name, content):
            target = folder / name
            if not target.parent.resolve().is_relative_to(public_root):
                raise ValueError("Formula export escaped public artifacts")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and digest(target) == hashlib.sha256(content).hexdigest():
                return
            temporary = target.with_name(".publish-" + uuid.uuid4().hex)
            temporary.write_bytes(content)
            temporary.replace(target)

        def publish_file(name, source, sha):
            target = folder / name
            if not target.parent.resolve().is_relative_to(public_root):
                raise ValueError("Formula export escaped public artifacts")
            if target.is_file() and digest(target) == sha:
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(".publish-" + uuid.uuid4().hex)
            shutil.copyfile(source, temporary)
            temporary.replace(target)

        publish("study.json", json.dumps(self.public(study), ensure_ascii=False, indent=2).encode())
        ranking = self.ranking(study)
        if ranking:
            table = pl.DataFrame(
                [
                    {k: t[k] for k in ["id", "set_id", "model", "dev_score", "cv_mean", "cv_std"]}
                    for t in ranking
                ]
            ).write_csv()
            publish("ranking.csv", table.encode())
        for formula in study["formulas"]:
            publish_file(
                formula["ref"] + ".py",
                self.base / study["id"] / "formulas" / formula["ref"] / "source.py",
                formula["sha256"],
            )
        for row in ranking:
            for name in row["hashes"]:
                publish_file(
                    row["id"] + "/" + name,
                    self.base / study["id"] / "results" / row["id"] / name,
                    row["hashes"][name],
                )
        return folder

    def handoff(self, handoff: FormulaHandoff | None):
        bound = self.active()
        if not bound and handoff is None:
            return None
        if handoff is None:
            raise ValueError(
                "Formula comparison active this turn: ResearchReply.formula_comparison must select a replayed result or explicitly report incomplete"
            )
        if bound and handoff.study_id not in self._active_studies:
            raise ValueError("Handoff must reference a formula comparison active this turn")
        study = self.get(handoff.study_id)
        self.verify_input(study, require_environment=False)
        if handoff.status == "incomplete":
            self.export(study)
            if not handoff.reason.strip() or handoff.selection_id:
                raise ValueError("Incomplete handoff requires a reason and no selection_id")
            return {
                "status": "incomplete",
                "reason": handoff.reason,
                "study": self.public(study),
                "other_studies": [s["id"] for s in bound if s["id"] != study["id"]],
            }
        if not study["selection_id"] or handoff.selection_id != study["selection_id"]:
            raise ValueError(
                "Selection missing or stale; freeze_formula_selection after completing all comparisons"
            )
        self.complete(study)
        selection = next(s for s in study["selections"] if s["id"] == handoff.selection_id)
        ranking = self.ranking(study)
        if selection["revision"] != study["revision"] or selection["ranking"] != [
            r["id"] for r in ranking
        ]:
            raise ValueError("Selected comparison universe changed")
        for row in ranking:
            self.verify_result(study, row)
        for formula in study["formulas"]:
            canonical = self.base / study["id"] / "formulas" / formula["ref"]
            if (
                digest(canonical / "source.py") != formula["sha256"]
                or digest(canonical / "spec.json") != formula["spec_sha256"]
            ):
                raise ValueError("Canonical formula snapshot changed")
        self.export(study)
        return {
            "status": "selected",
            "study": self.public(study),
            "selection": selection,
            "winner": ranking[0],
            "ranking": ranking,
            "other_studies": [s["id"] for s in bound if s["id"] != study["id"]],
        }

    def markdown(self, report):
        if not report:
            return ""
        study = report["study"]
        plan = study["plan"]
        base = f"/api/sessions/{quote(self.root.name)}/artifacts/"
        prefix = ""
        url = base + prefix + "artifacts/formula-lab/" + study["id"] + "/"
        lines = [
            "### 公式比较 · 程序核验记录",
            "",
            f"{plan['title']} · {', '.join(plan['labels'])}"
            + (f" · 固定 recipe `{plan['recipe_id']}`" if plan["recipe_id"] else ""),
            f"数据版本：`{plan.get('dataset_ref') or 'legacy discovery input'}`。",
            "",
            f"主指标：`{plan['metric']}`；训练 {len(plan['train'])} 片，开发集 {len(plan['dev'])} 片；按 lot 分组 {plan['folds']} 折。输入摘要 `{plan['input_sha256'][:16]}`。",
            "",
        ]
        if report["status"] == "incomplete":
            lines += ["**比较未完成，尚未确认最佳公式。** " + report["reason"], ""]
            pending = [t for t in study["trials"] if t["status"] != "succeeded"]
            if pending:
                lines += [
                    "未完成尝试："
                    + "；".join(
                        f"{'+'.join(t['formulas'])} / {t['model']}：{t['status']}" for t in pending
                    ),
                    "",
                ]
        else:
            winner = report["winner"]
            lines += [
                f"**当前候选中最佳：{' + '.join(winner['formulas'])}，模型 {winner['model']}。** 重算特征和预测一致。",
                "",
                "| 公式版本 | 模型 | 开发集分数 | lot CV均值 ± 标准差 |",
                "| --- | --- | ---: | ---: |",
            ]
            for row in report["ranking"]:
                lines.append(
                    f"| {' + '.join(row['formulas'])} | {row['model']} | {row['dev_score']:.4f} | {row['cv_mean']:.4f} ± {row['cv_std']:.4f} |"
                )
            lines += [
                "",
                "模型参数：`"
                + json.dumps(plan["model_params"][winner["model"]], ensure_ascii=False)
                + "`。",
                "",
            ]
            for ref in winner["formulas"]:
                formula = next(f for f in study["formulas"] if f["ref"] == ref)
                lines += [
                    f"**{ref}**：{formula['expression']}",
                    "",
                    formula["description"],
                    "",
                    f"传感器：{', '.join(formula['sensors'])}；step：{formula['steps']}；单位：{formula['unit']}；参数：`{json.dumps(formula['params'], ensure_ascii=False)}`。",
                    "",
                    f"[提取代码]({url + quote(ref + '.py')}) · SHA256 `{formula['sha256']}`",
                    "",
                ]
                if formula["change_reason"]:
                    lines += ["修订原因：" + formula["change_reason"], ""]
            lines += [
                f"[逐片预测]({url + winner['id'] + '/predictions.parquet'}) · [模型系数与预处理参数]({url + winner['id'] + '/model_details.json'}) · [已拟合模型]({url + winner['id'] + '/model.joblib'})",
                "",
            ]
        excluded = [s for s in study["sets"] if s["excluded"]]
        if excluded:
            lines += [
                "已明确排除的比较："
                + "；".join(
                    f"{'+'.join(s['formulas'])}：{s['excluded']['reason']}" for s in excluded
                ),
                "",
            ]
        record_link = f"[完整协议、版本与失败记录]({url}study.json)"
        if self.ranking(study):
            record_link += f" · [比较表]({url}ranking.csv)"
        lines += [
            record_link,
            "",
            LIMITS,
        ]
        if report["other_studies"]:
            lines += ["", "其他研究仍单独保存，未合并排名：" + ", ".join(report["other_studies"])]
        return "\n".join(lines)
