"""用 SQLAlchemy MetaData 注册各文件的表，不打开数据库。"""

from sqlalchemy import REAL, Column, Integer, MetaData, Table, Text, text

from .constants import (
    ACQUISITION_TABLE,
    BRANCHES_TABLE,
    CELLS_TABLE,
    DATASETS_TABLE,
    EVENTS_TABLE,
    EXPERIMENTS_TABLE,
    INBOX_TABLE,
    META_TABLE,
    OBSERVATIONS_TABLE,
    PROOFS_TABLE,
    REPAIRS_TABLE,
    REQUESTS_TABLE,
    SCOPES_TABLE,
    SEQUENCE_TABLE,
    STUDIES_TABLE,
    TASK_CALLS_TABLE,
    TASK_CHANGES_TABLE,
    TASK_ITEMS_TABLE,
    TOKENS_TABLE,
    VERSIONS_TABLE,
)

ui = MetaData()
events = Table(
    EVENTS_TABLE,
    ui,
    Column("id", Integer, primary_key=True),
    Column("task_id", Text, nullable=False),
    Column("type", Text, nullable=False),
    Column("data", Text, nullable=False),
    Column("created", REAL, nullable=False),
    Column("agent_id", Text, nullable=False, server_default="main"),
    Column("parent_agent_id", Text, nullable=False, server_default=""),
    sqlite_autoincrement=True,
)
inbox = Table(
    INBOX_TABLE,
    ui,
    Column("id", Integer, primary_key=True),
    Column("task_id", Text, nullable=False),
    Column("message", Text, nullable=False),
    Column("consumed", Integer, server_default=text("0")),
    Column("agent_id", Text, nullable=False, server_default="main"),
    Column("source", Text, nullable=False, server_default="user"),
    sqlite_autoincrement=True,
)
requests = Table(
    REQUESTS_TABLE,
    ui,
    Column("request_id", Text, primary_key=True),
    Column("task_id", Text, nullable=False),
)

dataset = MetaData()
datasets = Table(
    DATASETS_TABLE,
    dataset,
    Column("id", Text, primary_key=True),
    Column("doc", Text, nullable=False),
    Column("created", REAL, nullable=False),
)
scopes = Table(
    SCOPES_TABLE,
    dataset,
    Column("owner", Text, primary_key=True),
    Column("root_id", Text, nullable=False),
    Column("current_id", Text, nullable=False),
    Column("stack", Text, nullable=False),
)
meta = Table(
    META_TABLE,
    dataset,
    Column("key", Text, primary_key=True),
    Column("value", Text, nullable=False),
)
acquisition = Table(
    ACQUISITION_TABLE,
    dataset,
    Column("owner", Text, primary_key=True),
    Column("request_id", Text, primary_key=True),
    Column("request", Text),
    Column("result", Text),
)

study = MetaData()
studies = Table(
    STUDIES_TABLE,
    study,
    Column("id", Text, primary_key=True),
    Column("body", Text, nullable=False),
)
branch = MetaData()
branches = Table(
    BRANCHES_TABLE,
    branch,
    Column("id", Text, primary_key=True),
    Column("parent_id", Text),
    Column("mission", Text),
    Column("title", Text),
    Column("task", Text),
    Column("state", Text),
    Column("created", REAL),
    Column("updated", REAL),
    Column("pid", Integer),
    Column("process_created", REAL),
    Column("review_status", Text),
    Column("error", Text),
)
budget = MetaData()
tokens = Table(
    TOKENS_TABLE,
    budget,
    Column("id", Text, primary_key=True),
    Column("mission", Text),
    Column("agent", Text),
    Column("purpose", Text),
    Column("reserved", Integer),
    Column("charged", Integer, server_default=text("0")),
    Column("state", Text),
    Column("usage", Text),
    Column("created", REAL),
)
code = MetaData()
versions = Table(
    VERSIONS_TABLE,
    code,
    Column("name", Text, primary_key=True),
    Column("revision", Integer, primary_key=True),
    Column("source", Text),
    Column("sha256", Text),
)
ledger = MetaData()
sequence = Table(SEQUENCE_TABLE, ledger, Column("value", Integer, nullable=False))
cells = Table(
    CELLS_TABLE,
    ledger,
    Column("id", Text, primary_key=True),
    Column("doc", Text, nullable=False),
)
experiment = MetaData()
experiments = Table(
    EXPERIMENTS_TABLE,
    experiment,
    Column("id", Text, primary_key=True),
    Column("doc", Text),
)
observation = MetaData()
observations = Table(
    OBSERVATIONS_TABLE,
    observation,
    Column("id", Text, primary_key=True),
    Column("doc", Text),
)
repairs = Table(
    REPAIRS_TABLE,
    observation,
    Column("failure", Text, primary_key=True),
    Column("verification", Text, primary_key=True),
)
proof = MetaData()
proofs = Table(
    PROOFS_TABLE,
    proof,
    Column("id", Text, primary_key=True),
    Column("doc", Text, nullable=False),
)

# 工作项存储独立于会话历史，所有查询同时限定研究任务标识。

task_plan = MetaData()
task_items = Table(
    TASK_ITEMS_TABLE,
    task_plan,
    Column("research_task_id", Text, primary_key=True),
    Column("item_id", Text, primary_key=True),
    Column("ordinal", Integer, nullable=False),
    Column("doc", Text, nullable=False),
)
task_changes = Table(
    TASK_CHANGES_TABLE,
    task_plan,
    Column("id", Integer, primary_key=True),
    Column("research_task_id", Text, nullable=False),
    Column("item_id", Text, nullable=False),
    Column("actor_id", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    Column("before_doc", Text),
    Column("after_doc", Text, nullable=False),
)
task_calls = Table(
    TASK_CALLS_TABLE,
    task_plan,
    Column("research_task_id", Text, primary_key=True),
    Column("execution_id", Text, primary_key=True),
    Column("request", Text, nullable=False),
    Column("result", Text, nullable=False),
)
