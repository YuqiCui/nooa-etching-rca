"""只向模型开放五个 Worker 管理接口。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from ....agent import Session
    from ....runtime import AgentRuntime
    from ...base import ToolFactory


class WorkerTools:
    """把工具调用绑定到宿主确认的当前会话身份。"""

    namespace: ClassVar[str] = "worker"
    tool_names: ClassVar[tuple[str, ...]] = (
        "StartWorker", "AssignTask", "CheckWorkerStatus", "WaitWorkers", "CollectWorkerResult",
    )

    def __init__(self, runtime: AgentRuntime, session: Session):
        """保存当前宿主运行时和授权会话。"""
        self.runtime = runtime
        self.session = session
        self.active = True

    @classmethod
    @asynccontextmanager
    async def bind(cls, session: Session) -> AsyncIterator[WorkerTools]:
        """绑定共享运行时，并在会话关闭后撤销旧引用。"""
        if session.runtime is None:
            raise RuntimeError("WorkerTools 需要宿主绑定运行时")
        instance = cls(session.runtime, session)
        try:
            yield instance
        finally:
            instance.active = False

    def register_tools(self, factory: ToolFactory) -> None:
        """登记固定参数接口及等待和交付规则。"""
        for name, callback, description in (
            ("StartWorker", self.start, "创建一个空闲 Worker，不开始任务。"),
            ("AssignTask", self.assign, "向空闲 Worker 分配任务和交付清单，立即返回。"),
            ("CheckWorkerStatus", self.status, "查看所有 Worker 的状态；不收集结果，不改变状态。"),
            ("WaitWorkers", self.wait, "等待任一 Worker 结果就绪；必须单独调用。"),
            ("CollectWorkerResult", self.collect, "收集 Worker 结果，将完成状态的 Worker 转为空闲。"),
        ):
            factory.register_callable(
                name, callback, description=description, recover_validation_errors=True,
            )
        factory.add_instructions(
            "Approved Worker types: "
            "* general — a general-purpose task execution Worker. "
            "StartWorker(worker_type='general') creates an idle Worker; then call "
            "AssignTask(worker_id, task, deliver_list). Independent Workers can run concurrently. "
            "Worker results are only available through CollectWorkerResult; notifications do not "
            "contain the result. Collect before reassigning. Call WaitWorkers alone when blocked. "
            "Before finish, all child assignments must have ended and their results been collected. "
            "Workers finish with a JSON content string containing outcome, summary, deliverables "
            "and limitations. Include each assigned ID once; deliverables contain id, status "
            "(provided or unavailable), content and artifact_refs. outcome is completed, failed, "
            "blocked or cancelled; completed requires all items provided. "
            "Use real workspace files for artifact_refs. Only the host binds assignment identities."
        )

    def _require_active(self) -> None:
        """拒绝已关闭工具继续访问运行时。"""
        if not self.active:
            raise RuntimeError("WorkerTools 已关闭")

    async def start(self, worker_type: str) -> dict:
        """初始化指定批准类型，不接受模型声明工具或身份。"""
        self._require_active()
        return await self.runtime.start_worker(self.session, worker_type)

    async def assign(self, worker_id: str, task: str, deliver_list: list[str]) -> dict:
        """向直属空闲 Worker 下发一项有明确交付清单的任务。"""
        self._require_active()
        return await self.runtime.assign_task(self.session, worker_id, task, deliver_list)

    def status(self) -> dict:
        """返回当前会话直接创建的全部 Worker。"""
        self._require_active()
        return self.runtime.check_worker_status(self.session)

    async def wait(self) -> dict:
        """等待结果，保持当前任务活跃并释放执行槽位。"""
        self._require_active()
        return await self.runtime.wait_workers(self.session)

    async def collect(self, worker_id: str) -> dict:
        """收集完整交付记录，保留固定的幂等回执。"""
        self._require_active()
        return await self.runtime.collect_worker_result(self.session, worker_id)
