"""持久化服务入口；业务层只导入本模块和 services 中的契约。"""

from pathlib import Path

from . import tables as _tables
from .services import (
    BranchService,
    BudgetExceeded,
    BudgetService,
    CodeService,
    ConversationAuditService,
    DatasetService,
    DocumentService,
    EventService,
    ExecutionService,
    ObservationService,
    StorageError,
    StudyService,
    TaskPlanService,
)
from .sqlite.conversation import SQLiteConversationAuditService as _SQLiteConversationAuditService
from .sqlite.datasets import SQLiteDatasetService as _SQLiteDatasetService
from .sqlite.documents import (
    SQLiteCodeService as _SQLiteCodeService,
)
from .sqlite.documents import (
    SQLiteDocumentService as _SQLiteDocumentService,
)
from .sqlite.documents import (
    SQLiteExecutionService as _SQLiteExecutionService,
)
from .sqlite.documents import (
    SQLiteObservationService as _SQLiteObservationService,
)
from .sqlite.documents import (
    SQLiteStudyService as _SQLiteStudyService,
)
from .sqlite.events import SQLiteEventService as _SQLiteEventService
from .sqlite.orchestration import SQLiteBranchService as _SQLiteBranchService
from .sqlite.orchestration import SQLiteBudgetService as _SQLiteBudgetService
from .sqlite.runtime import SQLiteDatabase as _SQLiteDatabase
from .sqlite.tasks import SQLiteTaskPlanService as _SQLiteTaskPlanService

__all__ = [
    "BudgetExceeded",
    "StorageError",
    "create_branch_service",
    "create_budget_service",
    "create_code_service",
    "create_conversation_audit_service",
    "create_dataset_service",
    "create_event_service",
    "create_execution_service",
    "create_experiment_service",
    "create_observation_service",
    "create_proof_service",
    "create_study_service",
    "create_task_plan_service",
]


def create_event_service(root: Path) -> EventService:
    """创建根会话的事件服务。

    Args:
        root: 根会话目录。

    Returns:
        事件、指导和请求身份服务。
    """
    return _SQLiteEventService(_SQLiteDatabase(root / "ui.sqlite", _tables.ui))


def create_dataset_service(root: Path) -> DatasetService:
    """创建数据集元信息服务。

    Args:
        root: 根会话目录。

    Returns:
        数据集版本、选择栈和采集请求服务。
    """
    return _SQLiteDatasetService(
        _SQLiteDatabase(root / ".datasets/registry.sqlite", _tables.dataset)
    )


def create_study_service(base: Path) -> StudyService:
    """创建公式研究服务。

    Args:
        base: 公式研究目录。

    Returns:
        支持业务事务的研究文档服务。
    """
    return _SQLiteStudyService(_SQLiteDatabase(base / "registry.sqlite", _tables.study))


def create_branch_service(root: Path) -> BranchService:
    """创建方向生命周期服务。

    Args:
        root: 根会话目录。

    Returns:
        方向身份和状态服务。
    """
    return _SQLiteBranchService(_SQLiteDatabase(root / "branches.sqlite", _tables.branch))


def create_budget_service(root: Path, mission: str, limit: int) -> BudgetService:
    """创建共享预算服务。

    Args:
        root: 根会话目录。
        mission: 预算所属任务标识。
        limit: 任务总 token 上限。

    Returns:
        当前任务的预算预留和结算服务。
    """
    return _SQLiteBudgetService(
        _SQLiteDatabase(root / "budget.sqlite", _tables.budget), mission, limit
    )


def create_code_service(run: Path) -> CodeService:
    """创建源码版本服务。

    Args:
        run: 当前方向的运行目录。

    Returns:
        不可变源码版本服务。
    """
    return _SQLiteCodeService(_SQLiteDatabase(run / ".execution/code.sqlite", _tables.code))


def create_execution_service(run: Path) -> ExecutionService:
    """创建可信执行记录服务。

    Args:
        run: 当前方向的运行目录。

    Returns:
        执行编号和执行文档服务。
    """
    return _SQLiteExecutionService(
        _SQLiteDatabase(run / ".execution/ledger.sqlite", _tables.ledger)
    )


def create_experiment_service(run: Path) -> DocumentService:
    """创建实验文档服务。

    Args:
        run: 当前方向的运行目录。

    Returns:
        固定绑定实验业务的文档服务。
    """
    return _SQLiteDocumentService(
        _SQLiteDatabase(run / ".execution/experiments.sqlite", _tables.experiment),
        _tables.experiments,
    )


def create_observation_service(run: Path) -> ObservationService:
    """创建执行观察服务。

    Args:
        run: 当前方向的运行目录。

    Returns:
        观察和复验关系服务。
    """
    return _SQLiteObservationService(
        _SQLiteDatabase(run / ".execution/observations.sqlite", _tables.observation),
    )


def create_proof_service(run: Path) -> DocumentService:
    """创建可信证据文档服务。

    Args:
        run: 当前方向的运行目录。

    Returns:
        固定绑定可信证据业务的文档服务。
    """
    return _SQLiteDocumentService(
        _SQLiteDatabase(run / ".evidence/proofs.sqlite", _tables.proof),
        _tables.proofs,
    )


def create_conversation_audit_service(run: Path) -> ConversationAuditService:
    """创建第三方会话的只读审计服务。

    Args:
        run: 当前方向的运行目录。

    Returns:
        NOOA 会话审计服务。
    """
    return _SQLiteConversationAuditService(run / "conversation.sqlite")


def create_task_plan_service(root: Path, research_task_id: str) -> TaskPlanService:
    """创建固定研究任务范围的工作项存储。

    Args:
        root: 宿主专用存储目录，不由模型选择。
        research_task_id: 整次研究任务标识，不参与文件路径构造。

    Returns:
        支持原子修改、审计和执行去重的工作项服务。
    """
    return _SQLiteTaskPlanService(
        _SQLiteDatabase(root / "task_plan.sqlite", _tables.task_plan), research_task_id
    )
