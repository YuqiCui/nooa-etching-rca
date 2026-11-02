"""供业务层使用的持久化契约；参数和返回值均不包含 SQL 或连接。"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, Self, runtime_checkable

Document = dict[str, Any]


class StorageError(RuntimeError):
    """持久化操作失败，底层异常仅保留在异常链中。"""


class BudgetExceeded(RuntimeError):
    """共享预算不足，无法预留本次请求。"""


class DocumentService(Protocol):
    """访问创建时已绑定业务类型的文档，不接受表名或查询表达式。"""

    def get(self, identity: str) -> Document | None:
        """读取文档。

        Args:
            identity: 业务记录标识。

        Returns:
            文档副本；不存在时为 None。
        """
        ...

    def save(self, identity: str, document: Document) -> None:
        """保存或替换文档。

        Args:
            identity: 业务记录标识。
            document: 可序列化的业务文档。
        """
        ...

    def list(self, limit: int | None = None, offset: int = 0) -> list[Document]:
        """按最近写入顺序读取文档。

        Args:
            limit: 最大条数；None 表示读取全部。
            offset: 跳过的条数。

        Returns:
            文档副本列表。
        """
        ...


class StudyService(DocumentService, Protocol):
    """支持同一事务内读取和更新公式研究。"""

    def transaction(self, write: bool = True) -> AbstractContextManager[Self]:
        """创建业务事务，异常时回滚。

        Args:
            write: 是否在读取前取得写锁。

        Returns:
            仅暴露研究服务方法的事务上下文。
        """
        ...

    def list_by_id(self) -> list[Document]:
        """按研究标识升序返回所有文档。"""
        ...


class EventService(Protocol):
    """保存事件、消费指导消息并记录请求身份。"""

    def emit(
        self, task_id: str, kind: str, data: Document, agent_id: str, parent_agent_id: str
    ) -> int:
        """保存事件并返回可重放的递增编号。

        Args:
            task_id: 任务标识。
            kind: 事件种类。
            data: 事件负载。
            agent_id: 发出事件的方向。
            parent_agent_id: 父方向标识。

        Returns:
            事件编号。
        """
        ...

    def since(self, after: int, limit: int = 256) -> list[Document]:
        """读取编号之后的事件。

        Args:
            after: 不包含在结果中的起始编号。
            limit: 最大条数。

        Returns:
            按编号升序排列的事件。
        """
        ...

    def latest_id(self) -> int:
        """返回最新事件编号；空库为零。"""
        ...

    def add_guidance(self, task_id: str, agent_id: str, message: str, source: str) -> None:
        """保存待消费指导。

        Args:
            task_id: 任务标识。
            agent_id: 目标方向。
            message: 指导正文。
            source: 指导来源。
        """
        ...

    def take_guidance(self, task_id: str, agent_id: str) -> list[Document]:
        """在一次原子操作内读取并消费指导。

        Args:
            task_id: 任务标识。
            agent_id: 目标方向。

        Returns:
            尚未消费的消息和来源。
        """
        ...

    def request_task(self, request_id: str) -> str | None:
        """查询请求所属任务。

        Args:
            request_id: 客户端幂等请求标识。

        Returns:
            已保存的任务标识；未知请求为 None。
        """
        ...

    def claim_request(self, request_id: str, task_id: str) -> tuple[str, bool]:
        """原子登记请求，返回任务及是否首次登记。

        Args:
            request_id: 客户端幂等请求标识。
            task_id: 首次登记时绑定的任务。

        Returns:
            实际绑定的任务标识和首次登记标记。
        """
        ...

    def reflection_events(self, agent_id: str) -> list[Document]:
        """返回最近八条反思事件。

        Args:
            agent_id: 目标方向。

        Returns:
            最近事件在前的审查输入。
        """
        ...

    def progress(self, agent_id: str, task_id: str) -> tuple[int, float]:
        """返回方向的最新进度编号及时间。

        Args:
            agent_id: 目标方向。
            task_id: 所属任务。

        Returns:
            最新进度编号与 Unix 秒时间；无进度时均为零。
        """
        ...

    def close(self) -> None:
        """释放服务资源。"""
        ...


class DatasetService(Protocol):
    """持久化版本文档、选择范围和采集请求。"""

    def is_initialized(self) -> bool:
        """返回会话是否已有数据集注册记录存储，不触发初始化。"""
        ...

    def transaction(self, write: bool = True) -> AbstractContextManager[Self]:
        """创建数据集业务事务，异常时回滚。

        Args:
            write: 是否在读取状态前取得写锁；默认开启。

        Returns:
            仅暴露数据集服务方法的事务上下文。
        """
        ...

    def get(self, identity: str) -> Document | None:
        """读取不可变版本文档。

        Args:
            identity: 业务记录或数据集版本标识。

        Returns:
            版本文档；未知标识为 None。
        """
        ...

    def list(self, offset: int = 0, limit: int = 50) -> tuple[list[Document], int]:
        """分页读取数据集版本。

        Args:
            offset: 跳过的记录条数，负值按零处理。
            limit: 最大记录条数；支持 None 的接口中表示读取全部。

        Returns:
            版本文档列表及全局版本总数；权限过滤由数据集业务层负责。
        """
        ...

    def family_versions(self, family: str) -> list[int]:
        """读取同族版本号。

        Args:
            family: 数据集版本族标识。

        Returns:
            该版本族已发布的版本号列表。
        """
        ...

    def publish(self, document: Document) -> None:
        """追加版本文档，已有身份不覆盖。

        Args:
            document: 可序列化的业务文档。
        """
        ...

    def root(self) -> str | None:
        """读取会话初始数据集版本。

        Returns:
            初始版本标识；未初始化时为 None。
        """
        ...

    def bind_root(self, identity: str) -> None:
        """首次绑定会话初始版本，重复调用不覆盖。

        Args:
            identity: 业务记录或数据集版本标识。
        """
        ...

    def bind_scope(self, owner: str, root_ref: str) -> None:
        """首次创建方向的初始版本、当前版本和空选择栈。

        Args:
            owner: 方向标识；各方向的选择范围和采集请求独立。
            root_ref: 方向绑定的初始数据集版本。
        """
        ...

    def scope(self, owner: str) -> Document:
        """读取方向选择范围。

        Args:
            owner: 方向标识；各方向的选择范围和采集请求独立。

        Returns:
            包含 root、current 和 stack 的文档。
        """
        ...

    def select(self, owner: str, identity: str, back: str = "") -> str:
        """原子切换版本并维护选择栈，文件验证由数据集业务层负责。

        Args:
            owner: 方向标识；各方向的选择范围和采集请求独立。
            identity: 业务记录或数据集版本标识。
            back: 空字符串表示直接选择；root 返回初始版本，parent 返回上次选择。

        Returns:
            切换后的版本标识。
        """
        ...

    def acquisition(self, owner: str, request_id: str) -> tuple[Document, Document] | None:
        """读取方向已保存的采集结果。

        Args:
            owner: 方向标识；各方向的选择范围和采集请求独立。
            request_id: 方向内唯一的采集请求标识。

        Returns:
            请求文档与结果文档；不存在时为 None。
        """
        ...

    def save_acquisition(
        self, owner: str, request_id: str, request: Document, result: Document
    ) -> None:
        """保存采集请求及结果，重复身份报错。

        Args:
            owner: 方向标识；各方向的选择范围和采集请求独立。
            request_id: 方向内唯一的采集请求标识。
            request: 已校验的采集参数文档。
            result: 采集结果文档。
        """
        ...


class CodeService(Protocol):
    """在业务事务内维护不可变代码版本。"""

    def transaction(self, write: bool = True) -> AbstractContextManager[Self]:
        """创建源码版本业务事务，异常时回滚。

        Args:
            write: 是否在读取状态前取得写锁；默认开启。

        Returns:
            仅暴露代码服务方法的事务上下文。
        """
        ...

    def get(self, name: str, revision: int = 0) -> Document | None:
        """按名称和版本读取不可变源码。

        Args:
            name: 源码缓冲区名称。
            revision: 源码版本号；读取时零表示最新版本。

        Returns:
            包含 name、revision、source 和 sha256 的文档；不存在为 None。
        """
        ...

    def add(self, name: str, revision: int, source: str, sha256: str) -> None:
        """追加源码版本，重复版本报错；读改写必须置于同一业务事务。

        Args:
            name: 源码缓冲区名称。
            revision: 源码版本号；读取时零表示最新版本。
            source: 完整 Python 源码文本。
            sha256: 源码内容的 SHA-256 摘要。
        """
        ...

    def catalog(self) -> list[Document]:
        """列出所有源码缓冲区的最新版本。

        Returns:
            包含名称和版本号的文档列表。
        """
        ...


class ExecutionService(Protocol):
    """分配执行编号并保存可信执行文档。"""

    def next_cell(self, previous: int = 0) -> int:
        """原子分配大于持久化编号和调用方下界的执行编号。

        Args:
            previous: 调用方已知的执行编号下界。

        Returns:
            新分配的正整数编号。
        """
        ...

    def record(self, identity: str, document: Document) -> None:
        """只追加可信执行文档，重复身份报错。

        Args:
            identity: 业务记录或数据集版本标识。
            document: 可序列化的业务文档。
        """
        ...

    def list(self, limit: int | None = None, offset: int = 0) -> list[tuple[str, Document]]:
        """按最近执行顺序分页读取记录。

        Args:
            limit: 最大记录条数；支持 None 的接口中表示读取全部。
            offset: 跳过的记录条数，负值按零处理。

        Returns:
            执行标识与文档的列表；文件不存在时为空列表。
        """
        ...


class ObservationService(DocumentService, Protocol):
    """持久化执行观察和原路径复验关系。"""

    def resolve(self, failure: str, verification: str) -> None:
        """原子校验原路径复验并标记失败及其失败子观察已修复。

        Args:
            failure: 需要修复的失败观察标识。
            verification: 通过原路径复验的观察标识。
        """
        ...

    def resolved_failures(self) -> set[str]:
        """读取已经修复的观察标识。

        Returns:
            失败观察标识集合。
        """
        ...


class BranchService(Protocol):
    """保存方向身份和生命周期。"""

    def list(self) -> list[Document]:
        """按创建时间读取方向。

        Returns:
            方向身份和生命周期文档列表。
        """
        ...

    def get(self, identity: str) -> Document:
        """读取单个方向。

        Args:
            identity: 业务记录或数据集版本标识。

        Returns:
            方向身份和生命周期文档；未知身份抛出 ValueError。
        """
        ...

    def create(self, identity: str, parent: str, mission: str, title: str, task: str) -> None:
        """创建排队状态的方向；重复身份报错。

        Args:
            identity: 业务记录或数据集版本标识。
            parent: 父方向标识。
            mission: 所属任务标识。
            title: 方向标题。
            task: 方向任务正文。
        """
        ...

    def update(self, identity: str, values: Document) -> None:
        """更新方向允许变更的生命周期字段。

        Args:
            identity: 业务记录或数据集版本标识。
            values: 允许更新的生命周期字段，不接受 SQL 表达式。
        """
        ...


class BudgetService(Protocol):
    """共享任务预算的原子预留和结算。"""

    def reserve(self, amount: int, agent: str, purpose: str) -> str:
        """原子检查共享余额并登记预留；余额不足抛出 BudgetExceeded。

        Args:
            amount: 本次预留的非负 token 数量。
            agent: 预算归属方向。
            purpose: 模型调用用途。

        Returns:
            本次预留的唯一标识。
        """
        ...

    def settle(
        self,
        key: str,
        reported_tokens: int | None,
        reported_usage: Document | None,
        dispatched: bool = True,
        unreported_retry_tokens: int = 0,
    ) -> None:
        """结算当前任务的预留；未知预留抛出 ValueError。

        Args:
            key: 当前任务内的预留标识。
            reported_tokens: 模型报告的实际 token 数；None 按预留保守估计。
            reported_usage: 模型报告的用量文档；没有报告时为 None。
            dispatched: 是否已经向模型提供方发出请求。
            unreported_retry_tokens: 未报告用量的重试所需追加的估计 token 数。
        """
        ...

    def abandon_agent(self, agent: str) -> None:
        """将退出方向的在途预留转为保守估计费用。

        Args:
            agent: 预算归属方向。
        """
        ...

    def snapshot(self) -> Document:
        """汇总当前任务的 token 预算。

        Returns:
            上限、已计费、在途预留、可用额度和调用计数。
        """
        ...


class ConversationAuditService(Protocol):
    """只读审计 NOOA 所拥有的会话数据。"""

    def summary(self) -> tuple[dict[str, int], list[Document], int]:
        """只读审计 NOOA 对话，不创建或迁移其数据库。

        Returns:
            事件类型计数、摘要文档列表和活跃标签数。
        """
        ...


@runtime_checkable
class TaskPlanService(Protocol):
    """固定研究任务范围的工作项存储契约。"""

    def execute(
        self, operation: str, arguments: Document, *, actor_id: str, execution_id: str | None
    ) -> Document:
        """执行已校验的操作，并原子保存修改及重试结果。

        Args:
            operation: create、get、update 或 list。
            arguments: 工具层已校验的参数。
            actor_id: 宿主提供的调用者身份。
            execution_id: 稳定执行标识，None 表示不去重。

        Returns:
            工具结果或可恢复的业务错误。
        """
        ...

    def close(self) -> None:
        """释放存储资源。"""
        ...
