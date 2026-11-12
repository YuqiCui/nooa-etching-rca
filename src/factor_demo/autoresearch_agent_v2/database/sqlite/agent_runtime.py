"""以事务保存通用委派、会话检查点和父会话收件箱。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

ACTIVE_STATUSES = ("queued", "running", "interrupted")


class RuntimeStore:
    """维护运行状态与结果投递，不引入业务字段。"""

    def __init__(self, root: Path) -> None:
        """在宿主控制的目录建立数据库。"""
        root.mkdir(parents=True, exist_ok=True)
        path = root / "runtime.sqlite3"
        if path.is_symlink():
            raise ValueError("运行时数据库不能是符号链接")
        self._lease_fd = self._acquire_lease(root)
        try:
            self.connection = sqlite3.connect(path)
        except BaseException:
            os.close(self._lease_fd)
            raise
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                parent_agent_id TEXT,
                assignment_id TEXT NOT NULL UNIQUE,
                request TEXT NOT NULL,
                checkpoint TEXT NOT NULL,
                status TEXT NOT NULL,
                depth INTEGER NOT NULL,
                execution_key TEXT UNIQUE,
                output TEXT,
                result_id TEXT
            );
            CREATE TABLE IF NOT EXISTS inbox (
                ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id TEXT UNIQUE NOT NULL,
                parent_agent_id TEXT NOT NULL,
                sender_agent_id TEXT NOT NULL,
                assignment_id TEXT NOT NULL,
                status TEXT NOT NULL,
                content TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                acknowledged INTEGER NOT NULL DEFAULT 0
            );
        """)

    @staticmethod
    def _acquire_lease(root: Path) -> int:
        """独占同一运行目录，防止多个宿主进程重复驱动委派。"""
        descriptor = os.open(root / "runtime.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException as error:
            os.close(descriptor)
            raise RuntimeError("运行时存储已经由其他驱动者占用") from error
        return descriptor

    def close(self) -> None:
        """先关闭连接，再释放跨进程独占租约。"""
        self.connection.close()
        fcntl.flock(self._lease_fd, fcntl.LOCK_UN)
        os.close(self._lease_fd)

    def agent(self, agent_id: str) -> dict[str, Any] | None:
        """读取一个委派记录，不修改状态。"""
        row = self.connection.execute(
            "SELECT * FROM agents WHERE agent_id=?", (agent_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def register_root(self, state_json: str, agent_id: str) -> None:
        """登记根会话，保持已有持久化记录。"""
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO agents VALUES (?,NULL,?,?,?,'running',0,NULL,NULL,NULL)",
                (agent_id, uuid4().hex, "{}", state_json),
            )

    def begin_root(self, agent_id: str) -> None:
        """开始同一根会话的新用户请求。"""
        with self.connection:
            self.connection.execute(
                "UPDATE agents SET status='running', output=NULL, result_id=NULL WHERE agent_id=?",
                (agent_id,),
            )

    def enqueue(
        self,
        *,
        agent_id: str,
        parent_id: str,
        assignment_id: str,
        request: dict,
        checkpoint: str,
        depth: int,
        execution_key: str | None,
    ) -> dict[str, Any]:
        """原子登记委派；相同调用标识只创建一次。"""
        encoded = json.dumps(request, ensure_ascii=False, sort_keys=True)
        with self.connection:
            if execution_key is not None:
                previous = self.connection.execute(
                    "SELECT * FROM agents WHERE execution_key=?", (execution_key,)
                ).fetchone()
                if previous is not None:
                    if previous["request"] != encoded:
                        raise ValueError("同一次委派调用不能改变任务或权限")
                    return dict(previous)
            self.connection.execute(
                "INSERT INTO agents VALUES (?,?,?,?,?,'queued',?,?,NULL,NULL)",
                (agent_id, parent_id, assignment_id, encoded, checkpoint, depth, execution_key),
            )
        return self.agent(agent_id)

    def save_checkpoint(self, agent_id: str, checkpoint: str) -> None:
        """保存完整对话和已经消费的事件身份。"""
        with self.connection:
            self.connection.execute(
                "UPDATE agents SET checkpoint=? WHERE agent_id=?", (checkpoint, agent_id)
            )

    def set_status(self, agent_id: str, status: str) -> None:
        """记录非终态执行变化。"""
        with self.connection:
            self.connection.execute(
                "UPDATE agents SET status=? WHERE agent_id=?", (status, agent_id)
            )

    def children(self, parent_id: str, *, active_only: bool = False) -> list[dict[str, Any]]:
        """按登记顺序返回直接子委派。"""
        rows = self.connection.execute(
            "SELECT * FROM agents WHERE parent_agent_id=? ORDER BY rowid", (parent_id,)
        ).fetchall()
        return [dict(row) for row in rows if not active_only or row["status"] in ACTIVE_STATUSES]

    def pending(self, parent_id: str, limit: int) -> list[dict[str, Any]]:
        """读取有界的未确认事件。"""
        rows = self.connection.execute(
            "SELECT * FROM inbox WHERE parent_agent_id=? AND acknowledged=0 "
            "ORDER BY ordinal LIMIT ?",
            (parent_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def acknowledge(self, result_ids: list[str]) -> None:
        """确认已经写入检查点的事件。"""
        with self.connection:
            self.connection.executemany(
                "UPDATE inbox SET acknowledged=1 WHERE result_id=?",
                [(identity,) for identity in result_ids],
            )

    def complete(self, agent_id: str, content: str, status: str, checkpoint: str) -> str:
        """在同一事务提交结果、终态和父收件箱事件。"""
        with self.connection:
            record = self.agent(agent_id)
            if record is None:
                raise ValueError("未知 agent")
            if record["result_id"] is not None:
                return record["result_id"]
            identity = uuid4().hex
            self.connection.execute(
                "UPDATE agents SET status=?,output=?,result_id=?,checkpoint=? WHERE agent_id=?",
                (status, content, identity, checkpoint, agent_id),
            )
            if record["parent_agent_id"] is not None:
                self.connection.execute(
                    "INSERT INTO inbox(result_id,parent_agent_id,sender_agent_id,assignment_id,"
                    "status,content,sha256) VALUES (?,?,?,?,?,?,?)",
                    (
                        identity,
                        record["parent_agent_id"],
                        agent_id,
                        record["assignment_id"],
                        status,
                        content,
                        hashlib.sha256(content.encode()).hexdigest(),
                    ),
                )
            return identity

    def result(self, result_id: str) -> dict[str, Any] | None:
        """按稳定结果标识读取全文与身份。"""
        row = self.connection.execute(
            "SELECT * FROM inbox WHERE result_id=?", (result_id,)
        ).fetchone()
        return dict(row) if row is not None else None
