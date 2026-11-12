"""用事务保存带所有者与版本的通用 JSON 文档。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import JsonValue, TypeAdapter

MAX_RECORD_BYTES = 2 * 1024 * 1024
_JSON_VALUE = TypeAdapter(JsonValue)


class RecordStore:
    """按任务范围和所有者隔离文档，条件写入使用跨进程事务。"""

    def __init__(self, storage_root: Path) -> None:
        """打开宿主提供的存储目录，不接受模型选择数据库路径。"""
        self.root = storage_root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "records.sqlite3"
        if self.path.is_symlink():
            raise ValueError("文档数据库不接受符号链接")
        self._closed = False
        with self._transaction(write=True) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS records ("
                "scope TEXT NOT NULL, owner TEXT NOT NULL, document_id TEXT NOT NULL, "
                "revision INTEGER NOT NULL, content TEXT NOT NULL, "
                "PRIMARY KEY (scope, owner, document_id))"
            )

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """持有短事务，成功提交，异常回滚并关闭连接。"""
        if self._closed:
            raise RuntimeError("文档存储已经关闭")
        if self.path.is_symlink():
            raise ValueError("文档数据库不接受符号链接")
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _identity(document_id: str) -> None:
        """限制文档标识大小，标识不作为文件路径。"""
        if not document_id.strip() or len(document_id) > 200 or "\x00" in document_id:
            raise ValueError("文档标识必须为不超过 200 字符的非空文本")

    def get(self, scope: str, owner: str, document_id: str) -> dict[str, Any]:
        """读取当前所有者的文档，不提供跨所有者查询参数。"""
        self._identity(document_id)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT revision, content FROM records "
                "WHERE scope=? AND owner=? AND document_id=?", (scope, owner, document_id)
            ).fetchone()
        if row is None:
            return {"document": None}
        return {"document": {
            "document_id": document_id,
            "revision": row["revision"],
            "content": json.loads(row["content"]),
        }}

    def put(
        self,
        scope: str,
        owner: str,
        document_id: str,
        content: JsonValue,
        expected_revision: int,
    ) -> dict[str, Any]:
        """仅在版本匹配时完整替换文档；零版本表示创建。"""
        self._identity(document_id)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision 必须是非负整数")
        checked = _JSON_VALUE.validate_python(content, strict=True)
        serialized = json.dumps(checked, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode()) > MAX_RECORD_BYTES:
            raise ValueError("JSON 文档超过大小限制")
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT revision FROM records WHERE scope=? AND owner=? AND document_id=?",
                (scope, owner, document_id),
            ).fetchone()
            revision = row["revision"] if row else 0
            if revision != expected_revision:
                return {"error": "revision_conflict", "current_revision": revision}
            connection.execute(
                "INSERT INTO records (scope,owner,document_id,revision,content) VALUES (?,?,?,?,?) "
                "ON CONFLICT(scope,owner,document_id) DO UPDATE "
                "SET revision=excluded.revision, content=excluded.content",
                (scope, owner, document_id, revision + 1, serialized),
            )
        return {"document_id": document_id, "revision": revision + 1}

    def close(self) -> None:
        """撤销当前存储实例，已提交数据仍保留。"""
        self._closed = True
