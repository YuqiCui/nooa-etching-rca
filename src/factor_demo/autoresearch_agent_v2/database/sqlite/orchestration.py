"""方向生命周期和共享 token 预算的 SQLite 实现。"""

import json
import time
import uuid

from sqlalchemy import case, func, select, update
from sqlalchemy.dialects.sqlite import insert

from ..services import BudgetExceeded
from ..tables import branches, tokens


class SQLiteBranchService:
    """保存方向身份及允许变更的生命周期字段。"""

    def __init__(self, database):
        self._database = database

    def list(self):
        """按创建时间读取方向。"""
        with self._database.transaction() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(branches).order_by(branches.c.created)
                ).mappings()
            ]

    def get(self, identity):
        """读取单个方向，未知标识报错。"""
        with self._database.transaction() as connection:
            row = (
                connection.execute(
                    select(branches).where(
                        branches.c.id == identity,
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise ValueError("Unknown research branch")
        return dict(row)

    def create(self, identity, parent, mission, title, task):
        """追加排队状态的方向。"""
        with self._database.transaction(write=True) as connection:
            connection.execute(
                insert(branches).values(
                    id=identity,
                    parent_id=parent,
                    mission=mission,
                    title=title,
                    task=task,
                    state="queued",
                    created=time.time(),
                    updated=time.time(),
                )
            )

    def update(self, identity, values):
        """校验允许变更的字段后更新方向。"""
        allowed = {"mission", "state", "pid", "process_created", "review_status", "error"}
        if set(values) - allowed:
            raise ValueError("Invalid branch update")
        with self._database.transaction(write=True) as connection:
            connection.execute(
                update(branches)
                .where(branches.c.id == identity)
                .values(
                    **values,
                    updated=time.time(),
                )
            )


class SQLiteBudgetService:
    """按任务原子检查在途预留，未知用量保守计费。"""

    def __init__(self, database, mission, limit):
        self._database = database
        self._mission = mission
        self._limit = limit

    def reserve(self, amount, agent, purpose):
        """在写锁内检查余额并登记本次预留。"""
        if amount < 0:
            raise ValueError("Token reservation must be nonnegative")
        key = uuid.uuid4().hex
        with self._database.transaction(write=True) as connection:
            used = connection.scalar(
                select(
                    func.coalesce(
                        func.sum(
                            case(
                                (tokens.c.state == "reserved", tokens.c.reserved),
                                else_=tokens.c.charged,
                            )
                        ),
                        0,
                    )
                ).where(tokens.c.mission == self._mission)
            )
            if used + amount > self._limit:
                raise BudgetExceeded(
                    "Shared token budget exhausted (including in-flight reservations)"
                )
            connection.execute(
                insert(tokens).values(
                    id=key,
                    mission=self._mission,
                    agent=agent,
                    purpose=purpose,
                    reserved=amount,
                    state="reserved",
                    created=time.time(),
                )
            )
        return key

    def settle(
        self, key, reported_tokens, reported_usage, dispatched=True, unreported_retry_tokens=0
    ):
        """结算指定任务的预留，未知用量按预留收费。"""
        with self._database.transaction(write=True) as connection:
            condition = (tokens.c.id == key) & (tokens.c.mission == self._mission)
            reserved = connection.scalar(select(tokens.c.reserved).where(condition))
            if reserved is None:
                raise ValueError("Unknown token reservation")
            charged = (
                reported_tokens if reported_tokens is not None else reserved if dispatched else 0
            )
            state = (
                "reported_plus_estimate"
                if reported_tokens is not None and unreported_retry_tokens
                else "reported"
                if reported_tokens is not None
                else "estimated"
                if dispatched
                else "not_dispatched"
            )
            connection.execute(
                update(tokens)
                .where(condition)
                .values(
                    state=state,
                    charged=charged + unreported_retry_tokens,
                    usage=json.dumps(
                        {
                            "reported": reported_usage,
                            "unreported_retry_tokens": unreported_retry_tokens,
                        }
                    ),
                )
            )

    def abandon_agent(self, agent):
        """将退出方向的在途预留转为估计费用。"""
        with self._database.transaction(write=True) as connection:
            connection.execute(
                update(tokens)
                .where(
                    tokens.c.mission == self._mission,
                    tokens.c.agent == agent,
                    tokens.c.state == "reserved",
                )
                .values(state="estimated", charged=tokens.c.reserved)
            )

    def snapshot(self):
        """汇总指定任务已计费和在途费用。"""
        with self._database.transaction() as connection:
            rows = connection.execute(
                select(
                    tokens.c.state,
                    func.coalesce(func.sum(tokens.c.reserved), 0),
                    func.coalesce(func.sum(tokens.c.charged), 0),
                    func.count(),
                )
                .where(tokens.c.mission == self._mission)
                .group_by(tokens.c.state)
            ).all()
        charged = sum(row[2] for row in rows)
        held = sum(row[1] for row in rows if row[0] == "reserved")
        return {
            "token_limit": self._limit,
            "charged_tokens": charged,
            "reserved_tokens": held,
            "available_tokens": max(0, self._limit - charged - held),
            "logical_calls": sum(row[3] for row in rows),
            "estimated_calls": sum(
                row[3] for row in rows if row[0] in {"estimated", "reported_plus_estimate"}
            ),
        }
