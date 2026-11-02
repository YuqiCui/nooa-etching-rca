"""Supervisor-owned experiment manifests and disposable scientific workspaces."""

import asyncio
import hashlib
import json
import shutil
import uuid

from .autoresearch_agent_v2.database import create_experiment_service
from .code_buffers import CodeBuffers
from .datasets import file_hash
from .path_guard import copy_regular_nofollow


def hashes(root):
    result = {}
    if root.is_symlink():
        raise ValueError("Snapshot root cannot be a symlink")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Snapshot cannot contain symlinks")
        if path.is_file():
            result[str(path.relative_to(root))] = file_hash(path)
    return result


class CodeExperiments:
    def __init__(self, workspace):
        self.workspace = workspace
        self.run = workspace.run
        self.base = self.run / ".execution" / "experiments"
        self._storage = create_experiment_service(self.run)

    def save(self, doc):
        self._storage.save(doc["experiment_id"], doc)

    def get(self, identity):
        doc = self._storage.get(identity)
        if doc is None:
            raise ValueError("Unknown experiment")
        store = CodeBuffers(self.run)
        stale = any(
            store.get(src["name"])["sha256"] != src["sha256"]
            or store.get(src["name"])["revision"] != src["revision"]
            for src in doc["sources"]
        )
        workspace = self.workspace
        current_id = (
            workspace.dataset_session.current()["id"]
            if workspace.dataset_session
            else workspace.dataset_id
        )
        data = (
            workspace.dataset_session.path(current_id)
            if workspace.dataset_session
            else workspace.data_root
        )
        stale = stale or current_id != doc["dataset_ref"] or hashes(data) != doc["input_files"]
        doc["validity"] = "stale" if stale else "current"
        return doc

    def catalog(self, limit=20):
        return [self.get(doc["experiment_id"]) for doc in self._storage.list(limit)]

    async def execute(self, name, revision, dependencies, parameters, checks, timeout_s):
        from .research_workspace import PythonWorkspace

        if not checks.strip():
            raise ValueError("Explicit validation source is required")
        if revision < 1 or any(v < 1 for v in dependencies.values()):
            raise ValueError("Every source needs an explicit positive revision")
        if name in dependencies:
            raise ValueError("Candidate cannot also be a dependency")
        # JSON is the explicit input boundary: no live objects or credentials are inherited.
        parameters = json.loads(json.dumps(parameters, allow_nan=False))
        store = CodeBuffers(self.run)
        sources = [store.get(n, r) for n, r in dependencies.items()] + [store.get(name, revision)]
        parent = self.workspace
        async with parent.lock:
            dataset_id = (
                parent.dataset_session.current()["id"]
                if parent.dataset_session
                else parent.dataset_id
            )
            data = (
                parent.dataset_session.path(dataset_id)
                if parent.dataset_session
                else parent.data_root
            )
            identity = "experiment-" + uuid.uuid4().hex
            folder = self.base / identity
            snapshot = folder / "input"
            before = hashes(data)
            snapshot.mkdir(parents=True)
            for rel in before:
                copy_regular_nofollow(data / rel, snapshot / rel, required_root=data)
            if hashes(data) != before or hashes(snapshot) != before:
                raise ValueError("Inputs changed while creating experiment snapshot")
        doc = {
            "experiment_id": identity,
            "status": "running",
            "sources": sources,
            "parameters": parameters,
            "checks": checks,
            "dataset_ref": dataset_id,
            "input_files": before,
            "executions": [],
            "artifacts": {},
            "published": {},
        }
        from .execution_observations import parent_execution

        observation_id = parent.observations.record(
            "experiment",
            "running",
            experiment_id=identity,
            checks=checks,
            code="\n\n".join(src["source"] for src in sources),
        )
        doc["observation_id"] = observation_id
        observation_token = parent_execution.set(observation_id)
        self.save(doc)
        worker = PythonWorkspace(folder, events=parent.events)
        worker.observations = parent.observations
        worker.dataset_id = dataset_id
        (worker.work / "artifacts").mkdir(parents=True)
        try:
            setup = "import json\nPARAMETERS = json.loads(" + repr(json.dumps(parameters)) + ")"
            steps = [(setup, "experiment://parameters")]
            steps += [(src["source"], src["source_ref"]) for src in sources]
            steps += [(checks, "experiment://checks")]
            # One total budget across setup, dependencies, candidate and validation.
            async with asyncio.timeout(min(300, max(0.1, timeout_s))):
                for code, ref in steps:
                    result = await worker.execute(code, timeout_s, source_ref=ref)
                    doc["executions"].append(result)
                    self.save(doc)
                    if not result["ok"]:
                        doc["status"] = "failed"
                        break
                else:
                    doc["status"] = "checked"
        except asyncio.CancelledError:
            doc["status"] = "cancelled"
            raise
        except TimeoutError:
            doc["status"] = "timeout"
        except Exception as error:  # noqa: BLE001 - archive host-side experiment failures
            doc["status"] = "failed"
            doc["host_error"] = f"{type(error).__name__}: {error}"
        finally:
            await worker.close()
            if doc["status"] == "checked":
                try:
                    doc["artifacts"] = hashes(worker.work / "artifacts")
                except (ValueError, OSError) as error:
                    doc["status"] = "failed"
                    doc["host_error"] = str(error)
            parent_execution.reset(observation_token)
            observation = parent.observations.get(observation_id)
            observation.update(
                status="passed" if doc["status"] == "checked" else "failed",
                output="\n".join(item["output"] for item in doc["executions"]),
                error=doc.get("host_error", ""),
            )
            parent.observations.save(observation)
            self.save(doc)
        return self.get(identity)

    def publish(self, identity):
        """Publish a checked immutable artifact set after the scratch worker is closed."""
        from .live_events import artifact_root

        doc = self.get(identity)
        if doc["status"] != "checked" or doc["validity"] != "current":
            raise ValueError("Only current checked experiments can be published")
        source = self.base / identity / "workspace" / "artifacts"
        if hashes(source) != doc["artifacts"]:
            raise ValueError("Experiment artifacts changed after validation")
        root = artifact_root(self.run)
        root.mkdir(parents=True, exist_ok=True)
        dest = root / identity
        if doc["published"]:
            if hashes(dest) != doc["artifacts"]:
                raise ValueError("Published artifacts changed")
            return doc
        if dest.exists() or dest.is_symlink():
            raise ValueError("Publication destination already exists")
        staging = root / (".publish-" + uuid.uuid4().hex)
        staging.mkdir()
        try:
            for rel in doc["artifacts"]:
                copy_regular_nofollow(source / rel, staging / rel, required_root=source)
            if hashes(staging) != doc["artifacts"]:
                raise ValueError("Artifact changed during publication")
            staging.rename(dest)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        doc["published"] = {
            str((dest / rel).relative_to(self.workspace.work)): digest
            for rel, digest in doc["artifacts"].items()
        }
        self.save(doc)
        return self.get(identity)

    def proof(self, name, execution_ref=None):
        for doc in self.catalog(limit=-1):
            identity = doc["experiment_id"]
            if execution_ref and execution_ref != identity:
                continue
            digest = doc["published"].get(name)
            if not digest:
                continue
            if doc["validity"] != "current":
                raise ValueError("Experiment evidence is stale; rerun edited sources/inputs")
            if file_hash(self.workspace.work / name) != digest:
                raise ValueError("Published experiment artifact changed")
            return {
                **doc,
                "execution_ref": identity,
                "artifact_sha256": digest,
                "source_sha256": hashlib.sha256(
                    json.dumps(doc["sources"], sort_keys=True).encode()
                ).hexdigest(),
                "ok": True,
            }
        raise ValueError("No checked experiment provenance")
