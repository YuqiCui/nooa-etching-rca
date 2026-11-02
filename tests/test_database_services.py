"""验证数据库分层、旧库兼容、业务事务和跨进程一致性。"""

import ast
import multiprocessing
import re
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from factor_demo.autoresearch_agent_v2.database import (
    BudgetExceeded,
    StorageError,
    create_budget_service,
    create_code_service,
    create_conversation_audit_service,
    create_dataset_service,
    create_event_service,
    create_execution_service,
    create_experiment_service,
    create_observation_service,
    create_proof_service,
    create_study_service,
)
from factor_demo.autoresearch_agent_v2.database.constants import (
    CELLS_TABLE,
    EVENTS_TABLE,
    INBOX_TABLE,
    REQUESTS_TABLE,
    SEQUENCE_TABLE,
    STUDIES_TABLE,
    VERSIONS_TABLE,
)
from factor_demo.code_buffers import CodeBuffers
from factor_demo.live_events import LiveEvents


def _compete(root, operation):
    """由独立进程竞争同一个数据库的业务状态。"""
    root = Path(root)
    if operation == "budget":
        try:
            create_budget_service(root, "mission", 100).reserve(60, "main", "research")
            return True
        except BudgetExceeded:
            return False
    if operation == "code":
        try:
            CodeBuffers(root).write("candidate", "print(1)")
            return True
        except ValueError:
            return False
    if operation == "guidance":
        service = create_event_service(root)
        try:
            return service.take_guidance("mission", "main")
        finally:
            service.close()
    if operation == "request":
        service = create_event_service(root)
        try:
            return service.claim_request("request", "mission")[1]
        finally:
            service.close()
    return create_execution_service(root).next_cell()


@pytest.mark.parametrize("operation", ["budget", "code", "guidance", "request", "sequence"])
def test_cross_process_business_operations_are_atomic(tmp_path, operation):
    if operation == "guidance":
        service = create_event_service(tmp_path)
        service.add_guidance("mission", "main", "inspect", "user")
        service.close()
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(_compete, [str(tmp_path)] * 4, [operation] * 4))
    if operation == "guidance":
        assert [record for group in results for record in group] == [
            {"message": "inspect", "source": "user"}
        ]
    elif operation == "sequence":
        assert sorted(results) == [1, 2, 3, 4]
    else:
        assert results.count(True) == 1


def test_old_event_schema_is_migrated_without_changing_history(tmp_path):
    with sqlite3.connect(tmp_path / "ui.sqlite") as connection:
        connection.executescript(f"""
            CREATE TABLE {EVENTS_TABLE}(
                id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                type TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE {INBOX_TABLE}(
                id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                message TEXT NOT NULL, consumed INTEGER DEFAULT 0);
            CREATE TABLE {REQUESTS_TABLE}(request_id TEXT PRIMARY KEY, task_id TEXT NOT NULL);
        """)
        connection.execute(
            f"INSERT INTO {EVENTS_TABLE} VALUES(?,?,?,?,?)", (7, "mission", "old", '{"x":1}', 2.5)
        )
        connection.execute(
            f"INSERT INTO {INBOX_TABLE}(task_id,message) VALUES(?,?)", ("mission", "pending")
        )
    for _ in range(2):
        service = create_event_service(tmp_path)
        row = service.since(0)[0]
        assert row == {
            "id": 7,
            "task_id": "mission",
            "type": "old",
            "data": {"x": 1},
            "time": 2.5,
            "agent_id": "main",
            "parent_agent_id": "",
        }
        service.close()
    service = create_event_service(tmp_path)
    assert service.emit("mission", "new", {}, "child", "main") == 8
    assert service.take_guidance("mission", "child") == []
    assert service.take_guidance("mission", "main") == [{"message": "pending", "source": "user"}]
    assert service.take_guidance("mission", "main") == []
    service.close()


def test_old_code_study_and_execution_records_remain_readable(tmp_path):
    base = tmp_path / ".execution"
    base.mkdir()
    with sqlite3.connect(base / "code.sqlite") as connection:
        connection.execute(
            f"CREATE TABLE {VERSIONS_TABLE}(name TEXT,revision INTEGER,source TEXT,"
            "sha256 TEXT,PRIMARY KEY(name,revision))"
        )
        connection.execute(
            f"INSERT INTO {VERSIONS_TABLE} VALUES(?,?,?,?)", ("calc", 3, "x=1", "sha")
        )
    with sqlite3.connect(base / "ledger.sqlite") as connection:
        connection.execute(f"CREATE TABLE {SEQUENCE_TABLE}(value INTEGER NOT NULL)")
        connection.execute(f"INSERT INTO {SEQUENCE_TABLE} VALUES(9)")
        connection.execute(f"CREATE TABLE {CELLS_TABLE}(id TEXT PRIMARY KEY,doc TEXT NOT NULL)")
        connection.execute(f"INSERT INTO {CELLS_TABLE} VALUES(?,?)", ("cell-000009", '{"ok":true}'))
    study_base = tmp_path / ".formula-lab"
    study_base.mkdir()
    with sqlite3.connect(study_base / "registry.sqlite") as connection:
        connection.execute(f"CREATE TABLE {STUDIES_TABLE}(id TEXT PRIMARY KEY,body TEXT NOT NULL)")
        connection.execute(
            f"INSERT INTO {STUDIES_TABLE} VALUES(?,?)", ("study-1", '{"id":"study-1"}')
        )
    assert create_code_service(tmp_path).get("calc")["revision"] == 3
    assert create_execution_service(tmp_path).next_cell() == 10
    assert create_execution_service(tmp_path).list() == [("cell-000009", {"ok": True})]
    assert create_study_service(study_base).get("study-1") == {"id": "study-1"}


def test_transactions_yield_services_and_rollback_all_writes(tmp_path):
    studies = create_study_service(tmp_path / "studies")
    with pytest.raises(RuntimeError, match="abort"), studies.transaction() as transaction:
        assert not any(name in dir(transaction) for name in ("execute", "connection", "engine"))
        transaction.save("a", {"id": "a"})
        transaction.save("b", {"id": "b"})
        raise RuntimeError("abort")
    assert studies.list_by_id() == []
    datasets = create_dataset_service(tmp_path)
    with pytest.raises(RuntimeError, match="abort"), datasets.transaction() as transaction:
        transaction.publish({"id": "ds-1", "created": 1, "family": "f", "version": 1})
        transaction.bind_root("ds-1")
        raise RuntimeError("abort")
    assert datasets.root() is None
    assert datasets.list() == ([], 0)


def test_scope_stack_and_acquisition_are_owner_scoped(tmp_path):
    service = create_dataset_service(tmp_path)
    service.bind_scope("main", "root")
    service.bind_scope("child", "child-root")
    assert service.select("main", "derived") == "derived"
    assert service.scope("child") == {"root": "child-root", "current": "child-root", "stack": []}
    assert service.select("main", "", "parent") == "root"
    service.select("main", "derived")
    assert service.select("main", "", "root") == "root"
    assert service.scope("main")["stack"] == []
    service.save_acquisition("main", "request", {"source": "local"}, {"status": "unavailable"})
    assert service.acquisition("child", "request") is None
    assert service.acquisition("main", "request") == (
        {"source": "local"},
        {"status": "unavailable"},
    )


def test_observation_repairs_include_only_failed_descendants(tmp_path):
    service = create_observation_service(tmp_path)
    for identity, status, parent in (
        ("failure", "failed", None),
        ("child", "failed", "failure"),
        ("other", "failed", None),
    ):
        service.save(identity, {"observation_id": identity, "status": status, "parent_id": parent})
    service.save("verification", {"status": "passed", "repairs": "other"})
    with pytest.raises(ValueError, match="original-path"):
        service.resolve("failure", "verification")
    assert service.resolved_failures() == set()
    service.save(
        "verification", {"observation_id": "verification", "status": "passed", "repairs": "failure"}
    )
    service.resolve("failure", "verification")
    service.resolve("failure", "verification")
    assert service.resolved_failures() == {"failure", "child"}


def test_event_queries_and_sql_looking_data_are_business_values(tmp_path):
    service = create_event_service(tmp_path)
    payload = {"message": "'); DROP TABLE events; --"}
    first = service.emit("mission", "reflection.end", payload, "main", "")
    second = service.emit("mission", "tool.end", {}, "child", "main")
    service.emit("other", "tool.end", {}, "child", "main")
    assert service.reflection_events("main") == [{"event": "reflection.end", "data": payload}]
    assert service.progress("child", "mission")[0] == second
    assert service.since(first, 1)[0]["id"] == second
    assert service.claim_request(payload["message"], "mission") == ("mission", True)
    assert service.claim_request(payload["message"], "other") == ("mission", False)
    assert not any(name in dir(service) for name in ("execute", "db", "engine", "connection"))
    service.close()
    events = LiveEvents(tmp_path)
    assert "db" not in dir(events)
    assert events.since(0)[0]["data"] == payload
    events.close()


def test_document_services_are_fixed_to_their_business_and_persist(tmp_path):
    create_experiment_service(tmp_path).save("same-id", {"kind": "experiment"})
    create_proof_service(tmp_path).save("same-id", {"kind": "proof"})
    assert create_experiment_service(tmp_path).get("same-id") == {"kind": "experiment"}
    assert create_proof_service(tmp_path).get("same-id") == {"kind": "proof"}
    ledger = create_execution_service(tmp_path)
    ledger.record("cell-1", {"ok": True})
    with pytest.raises(StorageError):
        ledger.record("cell-1", {"ok": False})
    assert ledger.list() == [("cell-1", {"ok": True})]


def test_nooa_audit_is_readonly_and_does_not_create_missing_database(tmp_path):
    from nooa.storage.sqlite import SQLiteStorageManager

    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(StorageError):
        create_conversation_audit_service(missing).summary()
    assert list(missing.iterdir()) == []
    path = tmp_path / "conversation.sqlite"
    storage = SQLiteStorageManager(path)
    storage.close()
    before = path.read_bytes()
    counts, summaries, active = create_conversation_audit_service(tmp_path).summary()
    assert counts == {} and summaries == [] and active == 0
    assert path.read_bytes() == before


def test_application_cannot_import_storage_internals_or_embed_sql():
    root = Path(__file__).parents[1] / "src/factor_demo"
    database = root / "autoresearch_agent_v2/database"
    violations = []
    for path in root.rglob("*.py"):
        if path.is_relative_to(database):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                if any(
                    alias.name.split(".")[0] in {"sqlite3", "sqlalchemy"} for alias in node.names
                ):
                    violations.append(str(path))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if (
                    module.startswith(("sqlite3", "sqlalchemy"))
                    or any(
                        part in module
                        for part in ("database.sqlite", "database.tables", "database.constants")
                    )
                    or module.endswith("autoresearch_agent_v2.database")
                    and any(
                        alias.name.startswith("_")
                        or alias.name in {"sqlite", "tables", "constants"}
                        for alias in node.names
                    )
                ):
                    violations.append(str(path))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                statement = node.value.lstrip().upper()
                if re.match(
                    r"(?:SELECT\s+.+?\s+FROM\s|INSERT\s+INTO\s|UPDATE\s+\w+\s+SET\s"
                    r"|CREATE\s+TABLE\s|PRAGMA\s+\w+\s*=)",
                    statement,
                    re.DOTALL,
                ):
                    violations.append(f"{path}:{node.lineno}")
    assert violations == []
