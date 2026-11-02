"""在短写事务中保存研究计划、变更记录和执行去重结果。"""

import json
from datetime import UTC, datetime

from sqlalchemy import func, select

from .. import tables


def _encode(value):
    """统一保存有限 JSON 值，保留文本且稳定排列对象键。"""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class SQLiteTaskPlanService:
    """每个实例固定绑定一个研究任务，工作项标识不参与路径拼接。"""

    def __init__(self, database, research_task_id: str):
        """绑定宿主选择的存储与研究任务。

        Args:
            database: 项目统一的 SQLite 事务管理器。
            research_task_id: 整次研究任务的标识。
        """
        self._database = database
        self._research_task_id = research_task_id
        self._closed = False

    def close(self):
        """关闭资源并拒绝后续访问。"""
        self._closed = True
        self._database.close()

    def execute(self, operation, arguments, *, actor_id, execution_id=None):
        """执行工作项操作，写入与去重记录在同一事务提交。

        Args:
            operation: create、get、update 或 list。
            arguments: 已校验的操作参数。
            actor_id: 宿主调用者身份。
            execution_id: 同一次执行重试时保持不变的标识。

        Returns:
            操作结果或简短的业务错误。

        Raises:
            RuntimeError: 资源已经关闭。
            StorageError: 数据库无法读取或写入。
        """
        if self._closed:
            raise RuntimeError("任务存储已经关闭")
        write = operation in {"create", "update"}
        with self._database.transaction(write=write) as connection:
            request = _encode({"operation": operation, "arguments": arguments, "actor": actor_id})
            key = (tables.task_calls.c.research_task_id == self._research_task_id) & (
                tables.task_calls.c.execution_id == execution_id
            )
            if write and execution_id is not None:
                previous = (
                    connection.execute(select(tables.task_calls).where(key)).mappings().first()
                )
                if previous:
                    if previous["request"] != request:
                        return {"error": "execution_conflict", "hint": "同一执行标识不能改变参数。"}
                    return json.loads(previous["result"])
            if operation == "list":
                docs = connection.scalars(
                    select(tables.task_items.c.doc)
                    .where(tables.task_items.c.research_task_id == self._research_task_id)
                    .order_by(tables.task_items.c.ordinal)
                )
                items = [json.loads(doc) for doc in docs]
                return {
                    "tasks": [self._summary(item) for item in items if item["status"] != "deleted"]
                }
            if operation == "get":
                item = self._get_item(connection, arguments["task_id"])
                return {"task": item if item and item["status"] != "deleted" else None}
            if operation == "create":
                result = self._create_item(connection, arguments, actor_id)
            elif operation == "update":
                result = self._update_item(connection, arguments, actor_id)
            else:
                raise ValueError("未知工作项操作")
            if execution_id is not None:
                connection.execute(
                    tables.task_calls.insert().values(
                        research_task_id=self._research_task_id,
                        execution_id=execution_id,
                        request=request,
                        result=_encode(result),
                    )
                )
            return result

    @staticmethod
    def _summary(item):
        """列表只返回定位和进度展示所需字段。"""
        return {key: item[key] for key in ("id", "subject", "status", "active_form")}

    def _get_item(self, connection, item_id):
        """按研究任务和工作项双重限定读取记录。"""
        body = connection.scalar(
            select(tables.task_items.c.doc).where(
                tables.task_items.c.research_task_id == self._research_task_id,
                tables.task_items.c.item_id == item_id,
            )
        )
        return json.loads(body) if body is not None else None

    def _record_change(self, connection, before, after, actor_id):
        """在当前写事务追加完整变更记录。"""
        connection.execute(
            tables.task_changes.insert().values(
                research_task_id=self._research_task_id,
                item_id=after["id"],
                actor_id=actor_id,
                created_at=after["updated_at"],
                before_doc=_encode(before) if before else None,
                after_doc=_encode(after),
            )
        )

    def _create_item(self, connection, arguments, actor_id):
        """取得写锁后分配不复用的任务内递增标识。"""
        ordinal = 1 + (
            connection.scalar(
                select(func.max(tables.task_items.c.ordinal)).where(
                    tables.task_items.c.research_task_id == self._research_task_id,
                )
            )
            or 0
        )
        now = datetime.now(UTC).isoformat()
        item = {
            **arguments,
            "id": f"T{ordinal}",
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }
        connection.execute(
            tables.task_items.insert().values(
                research_task_id=self._research_task_id,
                item_id=item["id"],
                ordinal=ordinal,
                doc=_encode(item),
            )
        )
        self._record_change(connection, None, item, actor_id)
        return {"task": {key: item[key] for key in ("id", "subject", "status")}}

    def _update_item(self, connection, arguments, actor_id):
        """应用字段补丁，元数据按顶层键合并，删除保留墓碑。"""
        item_id = arguments["task_id"]
        before = self._get_item(connection, item_id)
        if before is None:
            return {"error": "task_not_found", "hint": "请调用 task_list 查看可用标识。"}
        patch = {key: value for key, value in arguments.items() if key != "task_id"}
        if before["status"] == "deleted":
            if patch.get("status") == "deleted":
                return {"success": True, "task_id": item_id, "updated_fields": []}
            return {"error": "task_deleted", "hint": "已删除的工作项不能修改，请创建新条目。"}
        after = {**before, **patch}
        if "metadata" in patch:
            metadata = before["metadata"].copy()
            for key, value in patch["metadata"].items():
                if value is None:
                    metadata.pop(key, None)
                else:
                    metadata[key] = value
            after["metadata"] = metadata
        changed = [key for key in patch if _encode(before[key]) != _encode(after[key])]
        result = {"success": True, "task_id": item_id, "updated_fields": changed}
        if not changed:
            return result
        after["updated_at"] = datetime.now(UTC).isoformat()
        connection.execute(
            tables.task_items.update()
            .where(
                tables.task_items.c.research_task_id == self._research_task_id,
                tables.task_items.c.item_id == item_id,
            )
            .values(doc=_encode(after))
        )
        self._record_change(connection, before, after, actor_id)
        if "status" in changed:
            result["status_change"] = {"from": before["status"], "to": after["status"]}
        return result
