"""Host-owned, bounded execution/artifact ledger; never trust child-written receipts."""

import hashlib
import shutil
from pathlib import Path

from .autoresearch_agent_v2.database import create_execution_service
from .datasets import file_hash


def next_cell(run: Path, previous: int = 0):
    return create_execution_service(run).next_cell(previous)


def artifact_hashes(run):
    from .live_events import artifact_root

    root = artifact_root(run)
    return {
        str(p.relative_to(run / "workspace")): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and not p.is_symlink()
        and p.resolve().is_relative_to(root)
        and "formula-lab" not in p.relative_to(root).parts
    }


def record_cell(run: Path, record, source: Path, data_root: Path, before=None, *, source_text=None):
    base = run / ".execution"
    base.mkdir(exist_ok=True)
    identity = f"cell-{record['cell']:06d}"
    try:
        after = artifact_hashes(run)
    except ValueError:
        after = {}
        record["ok"] = False
        record["output"] += "\nArtifact directory escaped its declared workspace"
    artifacts = {k: v for k, v in after.items() if (before or {}).get(k) != v}
    files = {}
    if data_root.exists():
        for p in sorted(data_root.rglob("*")):
            if p.is_file() and not p.is_symlink():
                files[str(p.relative_to(data_root))] = file_hash(p)
    payload = {
        k: record[k] for k in ["cell", "dataset_ref", "ok", "returncode", "started", "elapsed_s"]
    }
    payload.update(
        source_sha256=hashlib.sha256(source_text.encode()).hexdigest()
        if source_text is not None
        else file_hash(source),
        input_files=files,
        artifacts=artifacts,
    )
    create_execution_service(run).record(identity, payload)
    snapshot = base / identity / "artifacts"
    for name in artifacts:
        original = run / "workspace" / name
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
    return identity


def artifact_proof(run, name, execution_ref=None):
    # Never let a later ordinary cell launder stale experiment output.
    if len(Path(name).parts) > 1 and Path(name).parts[1].startswith("experiment-"):
        from .code_experiments import CodeExperiments
        from .research_workspace import PythonWorkspace

        return CodeExperiments(PythonWorkspace(run)).proof(name, execution_ref)

    rows = create_execution_service(run).list()
    if not rows:
        raise ValueError("Evidence has no trusted execution provenance")
    for identity, doc in rows:
        if execution_ref and identity != execution_ref:
            continue
        expected = doc["artifacts"].get(name)
        if expected and doc["ok"]:
            from .live_events import artifact_path

            target = artifact_path(run, name)
            if file_hash(target) != expected:
                raise ValueError("Evidence changed after its recorded execution")
            return {"execution_ref": identity, "artifact_sha256": expected, **doc}
    raise ValueError("Evidence is not bound to a successful recorded cell")
