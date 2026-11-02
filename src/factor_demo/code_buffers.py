"""Supervisor-owned immutable source revisions; no importable script files."""

import hashlib
import re
from pathlib import Path

from .autoresearch_agent_v2.database import create_code_service


class CodeBuffers:
    def __init__(self, run: Path):
        self._storage = create_code_service(run)

    @staticmethod
    def validate(name):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", name):
            raise ValueError("Use a code buffer name: letter followed by letters, digits, _ or -")

    def get(self, name, revision=0):
        self.validate(name)
        row = self._storage.get(name, revision)
        if row is None:
            raise ValueError("Unknown code buffer/revision")
        return {
            "name": name,
            "revision": row["revision"],
            "source": row["source"],
            "sha256": row["sha256"],
            "source_ref": f"cell://{name}@{row['revision']}",
        }

    def write(self, name, source, expected_revision=0, *, old=None, new=None):
        self.validate(name)
        with self._storage.transaction() as versions:
            row = versions.get(name)
            current = row["revision"] if row else 0
            if expected_revision != current:
                raise ValueError(
                    f"Revision conflict: expected {expected_revision}, current {current}"
                )
            if old is not None:
                if not row or not old or row["source"].count(old) != 1:
                    raise ValueError("Edit requires exactly one nonempty old-text match")
                source = row["source"].replace(old, new, 1)
            if len(source) > 1_000_000:
                raise ValueError("Source exceeds 1,000,000 characters")
            revision = current + 1
            digest = hashlib.sha256(source.encode()).hexdigest()
            versions.add(name, revision, source, digest)
        return {
            "name": name,
            "revision": revision,
            "sha256": digest,
            "source_ref": f"cell://{name}@{revision}",
            "executed": False,
        }

    def catalog(self):
        return self._storage.catalog()
