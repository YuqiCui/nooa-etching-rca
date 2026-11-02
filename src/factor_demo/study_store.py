"""通过业务服务保存公式协议与试验历史。"""

import re
from pathlib import Path

from .autoresearch_agent_v2.database import create_study_service


def public_study(study):
    """Saved report projection; viewing metadata never loads the numerical engine."""
    study.setdefault("replays", [])
    return {
        key: study[key]
        for key in (
            "id",
            "owner",
            "created",
            "plan",
            "formulas",
            "sets",
            "trials",
            "selections",
            "replays",
            "selection_id",
            "revision",
            "limits",
        )
    }


class StudyStore:
    def __init__(self, base: Path):
        self._storage = create_study_service(base)

    def transaction(self, write=True):
        return self._storage.transaction(write)

    def get(self, identity, transaction=None):
        if not re.fullmatch(r"study-[a-f0-9]{12}", identity):
            raise ValueError("Invalid study id")
        study = (transaction or self._storage).get(identity)
        if study is None:
            raise ValueError("Unknown formula study")
        study.setdefault("replays", [])
        return study

    def save(self, transaction, study):
        transaction.save(study["id"], study)

    def list(self):
        return self._storage.list_by_id()

    def catalog(self):
        return [public_study(study) for study in self.list()]

    def recent(self, limit=5):
        return self._storage.list(limit)
