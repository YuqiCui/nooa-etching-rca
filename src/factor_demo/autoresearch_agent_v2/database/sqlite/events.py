"""界面事件、指导消息和请求身份的 SQLite 实现。"""

import json
import time

from sqlalchemy import func, select, update
from sqlalchemy.dialects.sqlite import insert

from ..tables import events, inbox, requests


class SQLiteEventService:
    """对外只返回事件文档和业务标识。"""

    def __init__(self, database):
        self._database = database

    def emit(self, task_id, kind, data, agent_id, parent_agent_id):
        """保存事件并分配递增编号。"""
        with self._database.transaction(write=True) as connection:
            result = connection.execute(
                insert(events).values(
                    task_id=task_id,
                    type=kind,
                    data=json.dumps(data, ensure_ascii=False),
                    created=time.time(),
                    agent_id=agent_id,
                    parent_agent_id=parent_agent_id,
                )
            )
            return result.inserted_primary_key[0]

    def since(self, after, limit=256):
        """读取可重放事件。"""
        with self._database.transaction() as connection:
            rows = (
                connection.execute(
                    select(events).where(events.c.id > after).order_by(events.c.id).limit(limit)
                )
                .mappings()
                .all()
            )
        return [
            {
                "id": row["id"],
                "task_id": row["task_id"],
                "type": row["type"],
                "data": json.loads(row["data"]),
                "time": row["created"],
                "agent_id": row["agent_id"],
                "parent_agent_id": row["parent_agent_id"],
            }
            for row in rows
        ]

    def latest_id(self):
        """返回最新事件编号。"""
        with self._database.transaction() as connection:
            return connection.scalar(select(func.coalesce(func.max(events.c.id), 0)))

    def add_guidance(self, task_id, agent_id, message, source):
        """向指定方向投递指导。"""
        with self._database.transaction(write=True) as connection:
            connection.execute(
                insert(inbox).values(
                    task_id=task_id,
                    agent_id=agent_id,
                    message=message,
                    source=source,
                )
            )

    def take_guidance(self, task_id, agent_id):
        """在写锁内读取并消费指导，避免多个进程重复消费。"""
        with self._database.transaction(write=True) as connection:
            condition = (
                (inbox.c.task_id == task_id)
                & (inbox.c.agent_id == agent_id)
                & (inbox.c.consumed == 0)
            )
            rows = (
                connection.execute(
                    select(inbox.c.message, inbox.c.source).where(condition).order_by(inbox.c.id)
                )
                .mappings()
                .all()
            )
            connection.execute(update(inbox).where(condition).values(consumed=1))
            return [dict(row) for row in rows]

    def request_task(self, request_id):
        """查询已经登记的请求。"""
        with self._database.transaction() as connection:
            return connection.scalar(
                select(requests.c.task_id).where(requests.c.request_id == request_id)
            )

    def claim_request(self, request_id, task_id):
        """在写锁内登记幂等请求。"""
        with self._database.transaction(write=True) as connection:
            old = connection.scalar(
                select(requests.c.task_id).where(requests.c.request_id == request_id)
            )
            if old is not None:
                return old, False
            connection.execute(insert(requests).values(request_id=request_id, task_id=task_id))
            return task_id, True

    def reflection_events(self, agent_id):
        """读取审查所需的最近反思事件。"""
        with self._database.transaction() as connection:
            rows = connection.execute(
                select(events.c.type, events.c.data)
                .where(
                    events.c.agent_id == agent_id,
                    events.c.type.in_(("reflection.verdict", "reflection.end")),
                )
                .order_by(events.c.id.desc())
                .limit(8)
            ).all()
        return [{"event": kind, "data": json.loads(data)} for kind, data in rows]

    def progress(self, agent_id, task_id):
        """读取指定任务方向的进度水位。"""
        with self._database.transaction() as connection:
            row = connection.execute(
                select(
                    func.coalesce(func.max(events.c.id), 0),
                    func.coalesce(func.max(events.c.created), 0),
                ).where(
                    events.c.agent_id == agent_id,
                    events.c.task_id == task_id,
                    events.c.type.in_(
                        (
                            "model.reasoning.delta",
                            "model.content.delta",
                            "tool.output",
                            "tool.end",
                            "turn.checkpoint",
                        )
                    ),
                )
            ).one()
        return tuple(row)

    def close(self):
        """关闭服务。"""
        self._database.close()
