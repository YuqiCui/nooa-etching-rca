"""只读查询 NOOA 管理的会话，不注册或迁移其表。"""

import json

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

from ..constants import EVENTS_TABLE, NOOA_ACTIVE_TAGS_TABLE
from ..services import StorageError


class SQLiteConversationAuditService:
    """通过只读 URI 获取审计汇总，不创建会话文件。"""

    def __init__(self, path):
        self._path = path

    def summary(self):
        """读取事件类型计数、摘要文档及活跃标签数量。"""
        engine = create_engine(
            f"sqlite+pysqlite:///{self._path.resolve().as_uri()}?mode=ro&uri=true",
            poolclass=NullPool,
        )
        try:
            with engine.connect() as connection:
                # NOOA 是这些表的所有者；本适配器只固定读取已知审计字段。
                counts = dict(
                    connection.execute(
                        text(f"SELECT event_type, COUNT(*) FROM {EVENTS_TABLE} GROUP BY event_type")
                    )
                    .tuples()
                    .all()
                )
                summaries = [
                    json.loads(body)
                    for body in connection.scalars(
                        text(f"SELECT data FROM {EVENTS_TABLE} WHERE event_type=:kind"),
                        {"kind": "Summary"},
                    )
                ]
                active_count = connection.scalar(
                    text(f"SELECT COUNT(*) FROM {NOOA_ACTIVE_TAGS_TABLE}")
                )
                return counts, summaries, active_count
        except SQLAlchemyError as error:
            raise StorageError("无法读取 NOOA 会话审计") from error
        finally:
            engine.dispose()
