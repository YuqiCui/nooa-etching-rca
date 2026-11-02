"""封装 SQLAlchemy 引擎、事务和旧数据库的增量补列。"""

from contextlib import contextmanager
from pathlib import Path
from sqlite3 import SQLITE_BUSY, SQLITE_LOCKED, OperationalError
from time import monotonic, sleep

from sqlalchemy import MetaData, create_engine, event, inspect
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

from .. import tables
from ..services import StorageError


class SQLiteDatabase:
    """管理单个文件，连接与 SQLAlchemy 对象仅供本实现包使用。"""

    def __init__(self, path: Path, metadata: MetaData):
        self.path = path
        self._metadata = metadata
        self._ready = False
        self._engine = create_engine(
            URL.create("sqlite+pysqlite", database=str(path.resolve())),
            poolclass=NullPool,
            connect_args={"timeout": 30, "check_same_thread": False},
        )
        event.listen(self._engine, "connect", self._configure_connection)
        event.listen(self._engine, "begin", self._begin_transaction)

    @staticmethod
    def _configure_connection(connection, _record):
        """由 SQLAlchemy 管理 BEGIN，关闭 sqlite3 的隐式事务。"""
        connection.isolation_level = None
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA busy_timeout=30000")
            # 首次切换 WAL 时 SQLite 可能跳过 busy_timeout，限定仅重试连接配置。
            deadline = monotonic() + 30
            while True:
                try:
                    if cursor.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                        cursor.execute("PRAGMA journal_mode=WAL")
                    break
                except OperationalError as error:
                    if (error.sqlite_errorcode & 0xFF) not in {SQLITE_BUSY, SQLITE_LOCKED}:
                        raise
                    if monotonic() >= deadline:
                        raise
                    sleep(0.01)
        finally:
            cursor.close()

    @staticmethod
    def _begin_transaction(connection):
        """写事务在读取业务状态前取得跨进程写锁。"""
        connection.exec_driver_sql("BEGIN IMMEDIATE" if connection.info.get("write") else "BEGIN")

    def _initialize(self):
        """串行建表及迁移，保持既有文件布局和数据。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._engine.connect() as connection:
            connection.info["write"] = True
            with connection.begin():
                self._metadata.create_all(connection)
                if self._metadata is tables.ui:
                    for table, column, definition in (
                        (tables.events, "agent_id", "TEXT NOT NULL DEFAULT 'main'"),
                        (tables.events, "parent_agent_id", "TEXT NOT NULL DEFAULT ''"),
                        (tables.inbox, "agent_id", "TEXT NOT NULL DEFAULT 'main'"),
                        (tables.inbox, "source", "TEXT NOT NULL DEFAULT 'user'"),
                    ):
                        columns = {
                            item["name"] for item in inspect(connection).get_columns(table.name)
                        }
                        if column not in columns:
                            connection.exec_driver_sql(
                                f'ALTER TABLE "{table.name}" ADD COLUMN "{column}" {definition}'
                            )
        self._ready = True

    @contextmanager
    def transaction(self, write=False):
        """提供内部事务，提交或回滚后关闭连接。"""
        try:
            if not self._ready:
                self._initialize()
            with self._engine.connect() as connection:
                connection.info["write"] = write
                with connection.begin():
                    yield connection
        except SQLAlchemyError as error:
            raise StorageError("数据库操作失败") from error

    def close(self):
        """释放引擎持有的资源。"""
        self._engine.dispose()
