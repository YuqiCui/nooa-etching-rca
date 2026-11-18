"""在宿主数据库内保存 Worker 协议记录与幂等回执。"""

import json
from typing import Any

from .agent_runtime import RuntimeStore


class WorkerStore(RuntimeStore):
    """复用宿主独占租约，旧版委派记录保持原样。"""

    def __init__(self, root):
        """创建独立协议表，不覆盖已有会话或历史结果。"""
        super().__init__(root)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS worker_documents ("
            "kind TEXT NOT NULL, identity TEXT NOT NULL, body TEXT NOT NULL, "
            "PRIMARY KEY(kind, identity))"
        )
        self.connection.commit()

    def get(self, kind: str, identity: str) -> dict[str, Any] | None:
        """按协议对象标识读取独立副本。"""
        row = self.connection.execute(
            "SELECT body FROM worker_documents WHERE kind=? AND identity=?", (kind, identity)
        ).fetchone()
        return json.loads(row["body"]) if row else None

    def records(self, kind: str) -> list[dict[str, Any]]:
        """按首次创建顺序读取同类对象，规模由宿主限制。"""
        return [json.loads(row["body"]) for row in self.connection.execute(
            "SELECT body FROM worker_documents WHERE kind=? ORDER BY rowid", (kind,)
        )]

    def put(self, kind: str, identity: str, body: dict[str, Any]) -> None:
        """写入当前事务，由调用者一起提交状态、结果与回执。"""
        self.connection.execute(
            "INSERT INTO worker_documents VALUES (?,?,?) "
            "ON CONFLICT(kind,identity) DO UPDATE SET body=excluded.body",
            (kind, identity, json.dumps(body, ensure_ascii=False, allow_nan=False)),
        )

    def remove(self, kind: str, identity: str) -> None:
        """初始化失败时移除尚不可用的 Worker。"""
        self.connection.execute(
            "DELETE FROM worker_documents WHERE kind=? AND identity=?", (kind, identity)
        )
