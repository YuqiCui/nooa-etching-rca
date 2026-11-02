"""Small transparent fact checks accompany unrestricted exploratory Python.

Raw facts are recomputed from immutable inputs. Formula metrics are recomputed
from canonical predictions. Neither mode attests the semantics of arbitrary code.
"""

import hashlib
import json
from pathlib import Path

import polars as pl

from .autoresearch_agent_v2.database import create_proof_service
from .dataset_adapter import detect_schema, index_frame, read_table, safe_path
from .datasets import DatasetSession, canonical, file_hash
from .research_models import EvidenceRef, RawEvidenceSpec


def raw_value(root: Path, spec: RawEvidenceSpec):
    path = safe_path(root, spec.file)
    frame = read_table(path)
    schema = detect_schema(root)
    if spec.file != schema.index_file and schema.id_column in frame.columns:
        index = index_frame(root, schema)
        missing = [c for c in index.columns if c not in frame.columns]
        frame = frame.with_columns(pl.col(schema.id_column).cast(pl.String)).join(
            index.select(schema.id_column, *missing),
            on=schema.id_column,
            how="inner",
            validate="m:1",
        )
    for name, value in spec.where.items():
        frame = frame.filter(pl.col(name) == value)
    if spec.step is not None:
        schema = detect_schema(root)
        if not schema.steps_file or not schema.asset_column:
            raise ValueError("Step windows require declared timing and per-sample assets")
        sample = index_frame(root, schema).filter(pl.col(schema.asset_column) == spec.file)
        if sample.height != 1:
            raise ValueError("Step window must identify exactly one sample asset")
        times = read_table(safe_path(root, schema.steps_file))
        identity = schema.id_column if schema.id_column in times.columns else "run_id"
        timing = times.filter(
            (pl.col(identity).cast(pl.String) == str(sample[schema.id_column][0]))
            & (pl.col(schema.step_column or "step_id") == spec.step)
        )
        if timing.height != 1 or spec.end_fraction <= spec.start_fraction:
            raise ValueError("Invalid step window")
        row = timing.row(0, named=True)
        begin = row["start_s"] + spec.start_fraction * (row["end_s"] - row["start_s"])
        end = row["start_s"] + spec.end_fraction * (row["end_s"] - row["start_s"])
        time_column = spec.time_column or schema.time_column or "time_s"
        frame = frame.filter(
            (pl.col(schema.step_column or "step_id") == spec.step)
            & (pl.col(time_column) >= begin)
            & (pl.col(time_column) < end)
        )
    if spec.time_column:
        if spec.start is not None:
            frame = frame.filter(pl.col(spec.time_column) >= spec.start)
        if spec.end is not None:
            frame = frame.filter(pl.col(spec.time_column) < spec.end)
    if frame.is_empty():
        raise ValueError("Raw evidence selects no input rows")
    if spec.finite_only:
        for column in [spec.column, spec.product_column]:
            if column:
                frame = frame.filter(pl.col(column).is_finite())
    if spec.statistic == "count":
        return frame.height
    if spec.statistic == "unique_count":
        return frame[spec.column].n_unique()
    series = frame[spec.column]
    if spec.product_column:
        series = series * frame[spec.product_column]
    if spec.statistic == "value":
        if len(series) != 1:
            raise ValueError("Raw value requires one selected input row")
        return series[0]
    if spec.statistic == "std":
        return series.std(ddof=spec.ddof)
    return getattr(series, spec.statistic)()


def _publish(run, spec, actual, dataset_ref, source, mode):
    definition = {
        "spec": spec,
        "dataset_ref": dataset_ref,
        "actual": actual,
        "source": source,
        "mode": mode,
    }
    identity = "proof-" + hashlib.sha256(canonical(definition)).hexdigest()[:32]
    relative = "artifacts/verified-evidence/" + identity + ".csv"
    path = run / "workspace" / relative
    from .live_events import artifact_root

    root = artifact_root(run)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Verified artifact path escaped workspace")
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"evidence_id": [spec["id"]], "value": [actual]}).write_csv(path)
    reference = EvidenceRef(
        id=spec["id"],
        claim=spec["claim"],
        artifact=relative,
        column="value",
        where={"evidence_id": spec["id"]},
        value=actual,
        unit=spec["unit"],
        dataset_ref=dataset_ref,
        sha256=file_hash(path),
        proof_id=identity,
    )
    document = {**definition, "reference": reference.model_dump()}
    create_proof_service(run).save(identity, document)
    return reference.model_dump()


def materialize_raw(run, spec: RawEvidenceSpec, events=None):
    session = DatasetSession(run, events)
    identity = spec.dataset_ref or session.current()["id"]
    session.authorize(identity)
    root = session.registry.path(identity, verify=True)
    doc = session.registry.get(identity)
    if spec.file not in doc["files"]:
        raise ValueError("Choose a file in this dataset manifest")
    units = dict(session.schema(identity).units)
    if "catalog.json" in doc["files"]:
        catalog = json.loads(safe_path(root, "catalog.json").read_text())
        if isinstance(catalog, list):
            units.update(
                {
                    x["name"]: x["unit"]
                    for x in catalog
                    if isinstance(x, dict) and "name" in x and "unit" in x
                }
            )
    if "sensors.json" in doc["files"]:
        catalog = json.loads(safe_path(root, "sensors.json").read_text())
        if isinstance(catalog, list):
            units.update(
                {
                    x["name"]: x["unit"]
                    for x in catalog
                    if isinstance(x, dict) and "name" in x and "unit" in x
                }
            )
    expected = units.get(spec.column)
    if spec.statistic in {"count", "unique_count"}:
        if spec.unit not in {"samples", "rows", "runs", "objects", "1"}:
            raise ValueError("Use count units for a row count")
    elif expected:
        if spec.product_column:
            second = units.get(spec.product_column)
            if second:
                expected = expected + "*" + second
        equivalent = {"V*A": "W", "A*V": "W"}.get(expected, expected)
        if spec.unit not in {expected, equivalent}:
            raise ValueError("Declared evidence unit differs from its input metadata")
    actual = raw_value(root, spec)

    def authoritative_ref(ref):
        d = session.registry.get(ref)
        if d["provenance"].get("kind") in {"session_input", "supplied_local_input"}:
            return True
        return (
            d["relation"] == "subset"
            and bool(d["parents"])
            and all(authoritative_ref(p) for p in d["parents"])
        )

    authoritative = authoritative_ref(identity)
    return _publish(
        run,
        spec.model_dump(),
        actual,
        identity,
        {"file_sha256": doc["files"][spec.file]},
        "raw_recomputed" if authoritative else "derived_recomputed",
    )


def materialize_formula(run, lab, study_id, trial_id, metric, evidence_id, claim):
    if metric not in {"dev_score", "cv_mean", "cv_std"}:
        raise ValueError("Choose a canonical model metric")
    study = lab.get(study_id)
    lab.verify_input(study, require_environment=False)
    row = next(
        (r for r in study["trials"] if r["id"] == trial_id and r["status"] == "succeeded"), None
    )
    if row is None:
        raise ValueError("Unknown successful formula trial")
    lab.verify_result(study, row)
    return _publish(
        run,
        {
            "id": evidence_id,
            "claim": claim,
            "unit": "1",
            "study_id": study_id,
            "trial_id": trial_id,
            "metric": metric,
        },
        row[metric],
        study["plan"].get("dataset_ref"),
        {"study_revision": study["revision"], "trial_hashes": row["hashes"]},
        "formula_recomputed",
    )


def verify_proof(run, evidence, session=None, lab=None):
    doc = create_proof_service(run).get(evidence.proof_id)
    if doc is None:
        raise ValueError("Unknown trusted evidence proof")
    expected = doc["reference"]
    for key in [
        "id",
        "artifact",
        "column",
        "where",
        "statistic",
        "value",
        "unit",
        "dataset_ref",
        "sha256",
    ]:
        if getattr(evidence, key) != expected[key]:
            raise ValueError("Evidence differs from its trusted proof: " + key)
    if doc["mode"] in {"raw_recomputed", "derived_recomputed"}:
        session = session or DatasetSession(run)
        session.authorize(doc["dataset_ref"])
        root = session.registry.path(doc["dataset_ref"], verify=True)
        if file_hash(safe_path(root, doc["spec"]["file"])) != doc["source"]["file_sha256"]:
            raise ValueError("Raw evidence input changed")
        actual = raw_value(root, RawEvidenceSpec.model_validate(doc["spec"]))
    else:
        if lab is None:
            from .formula_lab import FormulaLab

            lab = FormulaLab(run)
        study = lab.get(doc["spec"]["study_id"])
        row = next(r for r in study["trials"] if r["id"] == doc["spec"]["trial_id"])
        lab.verify_result(study, row)
        actual = row[doc["spec"]["metric"]]
    if actual != doc["actual"]:
        raise ValueError("Trusted evidence no longer recomputes")
    return {
        "grade": doc["mode"],
        "proof_id": evidence.proof_id,
        "source": doc["spec"],
        "actual": actual,
    }
