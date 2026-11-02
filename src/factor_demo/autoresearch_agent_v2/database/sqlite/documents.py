"""公式研究、实验、观察、证据和执行记录的持久化实现。"""

import json
from contextlib import contextmanager

from sqlalchemy import func, literal_column, select, update
from sqlalchemy.dialects.sqlite import insert

from .. import tables


class SQLiteDocumentService:
    """访问内部固定绑定的文档表，不向上层开放表选择。"""

    def __init__(self, database, table, body_column="doc", connection=None):
        self._database = database
        self._table = table
        self._body = table.c[body_column]
        self._connection = connection

    @contextmanager
    def _connect(self, write=False):
        """复用业务事务内的连接，否则创建短事务。"""
        if self._connection is not None:
            yield self._connection
        else:
            with self._database.transaction(write=write) as connection:
                yield connection

    def get(self, identity):
        """读取业务文档。"""
        if not self._database.path.exists():
            return None
        with self._connect() as connection:
            body = connection.scalar(select(self._body).where(self._table.c.id == identity))
        return json.loads(body) if body is not None else None

    def save(self, identity, document):
        """替换文档时保留原实现的最近写入顺序。"""
        with self._connect(write=True) as connection:
            connection.execute(
                insert(self._table)
                .prefix_with("OR REPLACE")
                .values(
                    {
                        "id": identity,
                        self._body.name: json.dumps(document, allow_nan=False),
                    }
                )
            )

    def list(self, limit=None, offset=0):
        """返回最近写入的文档，缺失文件视为空列表。"""
        if not self._database.path.exists():
            return []
        query = select(self._body).order_by(literal_column("rowid").desc()).offset(offset)
        if limit is not None:
            query = query.limit(limit)
        with self._connect() as connection:
            return [json.loads(body) for body in connection.scalars(query)]


class SQLiteStudyService(SQLiteDocumentService):
    """在事务中维护公式研究文档。"""

    def __init__(self, database, connection=None):
        super().__init__(database, tables.studies, "body", connection)

    @contextmanager
    def transaction(self, write=True):
        """事务上下文仅交付研究服务，不交付连接。"""
        with self._database.transaction(write=write) as connection:
            yield SQLiteStudyService(self._database, connection)

    def list_by_id(self):
        """按研究标识升序读取文档。"""
        if not self._database.path.exists():
            return []
        with self._connect() as connection:
            return [
                json.loads(body)
                for body in connection.scalars(
                    select(tables.studies.c.body).order_by(tables.studies.c.id)
                )
            ]


class SQLiteObservationService(SQLiteDocumentService):
    """原子保存通过原路径复验的失败及其子观察。"""

    def __init__(self, database, connection=None):
        super().__init__(database, tables.observations, connection=connection)

    def resolve(self, failure, verification):
        """在同一事务中检查复验并记录修复关系。"""
        with self._database.transaction(write=True) as connection:
            scoped = SQLiteObservationService(self._database, connection)
            original, checked = scoped.get(failure), scoped.get(verification)
            if original is None or checked is None:
                raise ValueError("Unknown execution observation")
            if (
                original["status"] != "failed"
                or checked["status"] != "passed"
                or checked.get("repairs") != failure
            ):
                raise ValueError(
                    "Repair requires a failed observation and its passed original-path verification"
                )
            rows = scoped.list()
            parents = {failure}
            for _ in range(len(rows)):
                children = {r["observation_id"] for r in rows if r.get("parent_id") in parents}
                if children <= parents:
                    break
                parents |= children
            for child in rows:
                if child["observation_id"] in parents and child["status"] == "failed":
                    connection.execute(
                        insert(tables.repairs)
                        .values(
                            failure=child["observation_id"],
                            verification=verification,
                        )
                        .on_conflict_do_nothing()
                    )

    def resolved_failures(self):
        """返回已经通过复验的失败标识。"""
        if not self._database.path.exists():
            return set()
        with self._connect() as connection:
            return set(connection.scalars(select(tables.repairs.c.failure)))


class SQLiteExecutionService:
    """维护执行序号和只追加的可信执行记录。"""

    def __init__(self, database):
        self._database = database

    def next_cell(self, previous=0):
        """串行分配大于已知编号的下一个执行编号。"""
        with self._database.transaction(write=True) as connection:
            current = connection.scalar(select(tables.sequence.c.value))
            value = max(previous, current or 0) + 1
            if current is None:
                connection.execute(insert(tables.sequence).values(value=value))
            else:
                connection.execute(update(tables.sequence).values(value=value))
            return value

    def record(self, identity, document):
        """只追加执行文档，重复身份报错。"""
        with self._database.transaction(write=True) as connection:
            connection.execute(
                insert(tables.cells).values(
                    id=identity,
                    doc=json.dumps(document, allow_nan=False),
                )
            )

    def list(self, limit=None, offset=0):
        """返回最近执行记录及其标识。"""
        if not self._database.path.exists():
            return []
        query = select(tables.cells).order_by(literal_column("rowid").desc()).offset(offset)
        if limit is not None:
            query = query.limit(limit)
        with self._database.transaction() as connection:
            return [(identity, json.loads(body)) for identity, body in connection.execute(query)]


class SQLiteCodeService:
    """在业务事务中访问不可变源码版本。"""

    def __init__(self, database, connection=None):
        self._database = database
        self._connection = connection

    @contextmanager
    def transaction(self, write=True):
        """仅向调用方交付版本服务。"""
        with self._database.transaction(write=write) as connection:
            yield SQLiteCodeService(self._database, connection)

    @contextmanager
    def _connect(self, write=False):
        """复用当前事务或创建短事务。"""
        if self._connection is not None:
            yield self._connection
        else:
            with self._database.transaction(write=write) as connection:
                yield connection

    def get(self, name, revision=0):
        """按名称和版本读取源码，零表示最新版本。"""
        query = select(tables.versions).where(tables.versions.c.name == name)
        if revision != 0:
            query = query.where(tables.versions.c.revision == revision)
        with self._connect() as connection:
            row = (
                connection.execute(query.order_by(tables.versions.c.revision.desc()).limit(1))
                .mappings()
                .first()
            )
        return dict(row) if row is not None else None

    def add(self, name, revision, source, sha256):
        """追加新版本，禁止覆盖已有版本。"""
        with self._connect(write=True) as connection:
            connection.execute(
                insert(tables.versions).values(
                    name=name,
                    revision=revision,
                    source=source,
                    sha256=sha256,
                )
            )

    def catalog(self):
        """返回每个源码缓冲区的最新版本。"""
        with self._connect() as connection:
            rows = connection.execute(
                select(
                    tables.versions.c.name,
                    func.max(tables.versions.c.revision),
                )
                .group_by(tables.versions.c.name)
                .order_by(tables.versions.c.name)
            ).all()
        return [{"name": name, "revision": revision} for name, revision in rows]
