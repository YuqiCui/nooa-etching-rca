"""Durable host observations shared by execution, correction and read-only reflection."""

import json
import time
import uuid
from contextvars import ContextVar

from .autoresearch_agent_v2.database import create_observation_service

parent_execution = ContextVar("parent_execution", default=None)


class ExecutionObservations:
    def __init__(self, run):
        self._storage = create_observation_service(run)

    def save(self, doc):
        self._storage.save(doc["observation_id"], doc)
        return doc["observation_id"]

    def record(self, kind, status, **fields):
        identity = "observation-" + uuid.uuid4().hex
        return self.save(
            {
                "observation_id": identity,
                "kind": kind,
                "status": status,
                "parent_id": parent_execution.get(),
                "created": time.time(),
                **fields,
            }
        )

    def get(self, identity):
        doc = self._storage.get(identity)
        if doc is None:
            raise ValueError("Unknown execution observation")
        return doc

    def resolve(self, failure, verification):
        self._storage.resolve(failure, verification)

    def read(self, identity, offset=0, limit=12000):
        text = json.dumps(self.get(identity), ensure_ascii=False, indent=2)
        offset, limit = max(0, offset), min(32000, max(1, limit))
        return {
            "observation_id": identity,
            "content": text[offset : offset + limit],
            "next_offset": offset + limit if len(text) > offset + limit else None,
        }

    def context(self):
        rows = [(doc["observation_id"], doc) for doc in self._storage.list()]
        if not rows:
            return {"recent": [], "unresolved_failures": []}
        resolved = self._storage.resolved_failures()

        def compact(body):
            doc = body
            result = {
                k: doc[k]
                for k in (
                    "observation_id",
                    "parent_id",
                    "kind",
                    "status",
                    "execution_ref",
                    "experiment_id",
                    "epoch",
                    "state_effect",
                    "runtime_status",
                    "exception_type",
                    "repairs",
                    "goal",
                )
                if k in doc
            }
            for key in ("code", "source_ref", "output", "checks", "error"):
                if key in doc:
                    value = str(doc[key])
                    result[key] = (
                        value
                        if len(value) <= 2000
                        else value[:900]
                        + "\n[read_observation for full evidence]\n"
                        + value[-1000:]
                    )
            return result

        failed = [
            (key, body)
            for key, body in rows
            if body["status"] == "failed"
            and body["kind"] != "orchestration"
            and key not in resolved
        ]
        return {
            "recent": [compact(body) for _, body in rows[:3]],
            "unresolved_failures": [compact(body) for _, body in failed[:6]],
            "unresolved_count": len(failed),
            "semantics": "Supplied checks and repair-path verification are distinct from goal acceptance. Unrelated success never resolves failures.",
        }
