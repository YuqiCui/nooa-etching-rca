"""数据集版本、方向选择栈和采集请求的 SQLite 实现。"""

import json
from contextlib import contextmanager

from sqlalchemy import func, select, update
from sqlalchemy.dialects.sqlite import insert

from ..tables import acquisition, datasets, meta, scopes


class SQLiteDatasetService:
    """保存数据集元信息；文件、权限和内容校验由数据业务层负责。"""

    def __init__(self, database, connection=None):
        self._database = database
        self._connection = connection

    def is_initialized(self):
        """检查已有存储，不创建数据库文件。"""
        return self._database.path.exists()

    @contextmanager
    def transaction(self, write=True):
        """为版本发布等复合操作提供业务事务。"""
        with self._database.transaction(write=write) as connection:
            yield SQLiteDatasetService(self._database, connection)

    @contextmanager
    def _connect(self, write=False):
        """优先使用当前业务事务的连接。"""
        if self._connection is not None:
            yield self._connection
        else:
            with self._database.transaction(write=write) as connection:
                yield connection

    def get(self, identity):
        """读取不可变版本文档。"""
        with self._connect() as connection:
            body = connection.scalar(select(datasets.c.doc).where(datasets.c.id == identity))
        return json.loads(body) if body is not None else None

    def list(self, offset=0, limit=50):
        """按创建时间分页读取版本并统计总数。"""
        with self._connect() as connection:
            docs = [
                json.loads(body)
                for body in connection.scalars(
                    select(datasets.c.doc)
                    .order_by(datasets.c.created.desc(), datasets.c.id)
                    .limit(min(100, max(1, limit)))
                    .offset(max(0, offset))
                )
            ]
            total = connection.scalar(select(func.count()).select_from(datasets))
        return docs, total

    def family_versions(self, family):
        """读取同族已发布的版本号。"""
        with self._connect() as connection:
            docs = [json.loads(body) for body in connection.scalars(select(datasets.c.doc))]
        return [doc["version"] for doc in docs if doc["family"] == family]

    def publish(self, document):
        """追加版本；已有身份不覆盖。"""
        with self._connect(write=True) as connection:
            connection.execute(
                insert(datasets)
                .values(
                    id=document["id"],
                    doc=json.dumps(document, ensure_ascii=False),
                    created=document["created"],
                )
                .on_conflict_do_nothing()
            )

    def root(self):
        """读取会话初始版本。"""
        with self._connect() as connection:
            return connection.scalar(select(meta.c.value).where(meta.c.key == "root"))

    def bind_root(self, identity):
        """仅在首次初始化时绑定会话初始版本。"""
        with self._connect(write=True) as connection:
            connection.execute(
                insert(meta).values(key="root", value=identity).on_conflict_do_nothing()
            )

    def bind_scope(self, owner, root_ref):
        """仅在首次初始化时建立方向选择栈。"""
        with self._connect(write=True) as connection:
            connection.execute(
                insert(scopes)
                .values(
                    owner=owner,
                    root_id=root_ref,
                    current_id=root_ref,
                    stack="[]",
                )
                .on_conflict_do_nothing()
            )

    def scope(self, owner):
        """读取方向的根版本、当前版本和选择历史。"""
        with self._connect() as connection:
            row = (
                connection.execute(select(scopes).where(scopes.c.owner == owner)).mappings().first()
            )
        if row is None:
            raise ValueError("Dataset scope is not initialized")
        return {
            "root": row["root_id"],
            "current": row["current_id"],
            "stack": json.loads(row["stack"]),
        }

    def select(self, owner, identity, back=""):
        """在写锁内切换版本并维护选择栈。"""
        with self._connect(write=True) as connection:
            scoped = SQLiteDatasetService(self._database, connection)
            scope = scoped.scope(owner)
            stack = scope["stack"]
            if back == "root":
                identity, stack = scope["root"], []
            elif back == "parent":
                identity = stack.pop() if stack else scope["root"]
            elif identity != scope["current"]:
                stack.append(scope["current"])
            connection.execute(
                update(scopes)
                .where(scopes.c.owner == owner)
                .values(
                    current_id=identity,
                    stack=json.dumps(stack),
                )
            )
        return identity

    def acquisition(self, owner, request_id):
        """按方向和请求标识读取已保存采集结果。"""
        with self._connect() as connection:
            row = connection.execute(
                select(acquisition.c.request, acquisition.c.result).where(
                    acquisition.c.owner == owner,
                    acquisition.c.request_id == request_id,
                )
            ).first()
        return (json.loads(row[0]), json.loads(row[1])) if row is not None else None

    def save_acquisition(self, owner, request_id, request, result):
        """保存采集请求及结果，不允许覆盖同方向的请求身份。"""
        with self._connect(write=True) as connection:
            connection.execute(
                insert(acquisition).values(
                    owner=owner,
                    request_id=request_id,
                    request=json.dumps(request),
                    result=json.dumps(result),
                )
            )
