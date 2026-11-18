"""以五个 Worker 工具驱动可复用会话、任务及显式结果收集。"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self
from uuid import uuid4

from ..database import WorkerStore
from ..tools.call_context import execution_id
from ..tools.policy import AllowTools, ToolPolicy
from ..tools.role_policy import role_tool_names
from .delivery import terminal_result, validate_delivery
from .typing import ChildRequest, RuntimeLimits

if TYPE_CHECKING:
    from ..agent import Session, SessionState

ChildFactory = Callable[["Session", ChildRequest], "Session | Awaitable[Session]"]


def error(code: str, message: str, **details: Any) -> dict:
    """生成有界、可由模型处理的统一错误响应。"""
    return {"ok": False, "error": {"code": code, "message": message, **details}}


class AgentRuntime:
    """保持单会话驱动、内层沙箱及宿主所有权边界。"""

    def __init__(self, storage_root: Path, child_factory: ChildFactory | None = None, *,
                 limits: RuntimeLimits | None = None, worker_tools: AllowTools | None = None,
                 worker_skills: tuple[str, ...] = ()) -> None:
        """登记宿主批准的 Worker 能力，不接受模型扩权。"""
        from .factory import SessionFactory

        self.root = storage_root.expanduser().resolve()
        self.storage_root = self.root
        self.limits = limits or RuntimeLimits()
        self.store = WorkerStore(self.root)
        self.child_factory = child_factory or SessionFactory()
        self.worker_tools = worker_tools
        self.worker_skills = worker_skills
        self._sessions: dict[str, Session] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._drivers: dict[str, asyncio.Task] = {}
        self._signals: dict[str, asyncio.Event] = {}
        self._waiting: set[str] = set()
        self._interrupted: set[str] = set()
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(self.limits.max_concurrent)
        self._slot_holders: set[str] = set()
        self._closed = False

    def _require_open(self) -> None:
        """拒绝关闭后调用，避免使用已经释放的数据库。"""
        if self._closed:
            raise RuntimeError("AgentRuntime 已关闭")

    def attach(self, session: Session) -> None:
        """登记五工具和请求边界钩子，不启动模型。"""
        from ..tools.builtin_tools.workers import WorkerTools

        self._require_open()
        identity = session.state.agent_id
        if identity in self._sessions:
            if self._sessions[identity] is session:
                return
            raise ValueError("同一 Worker 身份不能绑定多个会话")
        if self.root.is_relative_to(session.state.workdir.expanduser().resolve()):
            raise ValueError("运行时记录必须位于模型可写工作目录之外")
        if session.runtime not in (None, self):
            raise ValueError("会话已经属于其他运行时")
        if session.state.parent_agent_id is None:
            with self.store.connection:
                if self.store.get("root", identity) is None:
                    self.store.put("root", identity, {
                        "root_run_id": uuid4().hex, "status": "running",
                        "checkpoint": session.state.model_dump_json(),
                    })
        elif self.store.get("worker", identity) is None:
            raise ValueError("Worker 必须由宿主先登记")
        session.register_stateful_tool(WorkerTools)
        session.runtime = self
        self._sessions[identity] = session
        self._signals.setdefault(identity, asyncio.Event())
        previous = session.loop.before_completion_hook

        async def before_completion() -> None:
            """只在完整工具批次后投递宿主通知并保存历史。"""
            await previous()
            await self.deliver(session)
            self.save_checkpoint(session)

        session.loop.before_completion_hook = before_completion
        session.loop.finish_hook = lambda content: self.finish(session, content)
        session.loop.context_providers.append(lambda: self._assignment_context(session))

    def _root_id(self, session: Session) -> str:
        """取得会话所属根运行，阻止跨根运行复用。"""
        identity = session.state.agent_id
        record = self.store.get("worker", identity) if session.state.parent_agent_id else None
        return record["root_run_id"] if record else self.store.get("root", identity)["root_run_id"]

    def _caller_running(self, session: Session) -> bool:
        """只有当前正在处理任务的会话可以创建或分配工作。"""
        if session.state.parent_agent_id is None:
            return self.store.get("root", session.state.agent_id)["status"] == "running"
        worker = self.store.get("worker", session.state.agent_id)
        return worker["status"] == "busy" and worker["active_assignment_id"] == (
            session.state.assignment_id
        )

    def _children(self, session: Session) -> list[dict]:
        """只返回当前根运行内、当前会话直接创建的 Worker。"""
        root_id = self._root_id(session)
        return [worker for worker in self.store.records("worker")
                if worker["owner_session_id"] == session.state.agent_id
                and worker["root_run_id"] == root_id]

    def _authorized(self, session: Session, worker_id: str) -> dict:
        """拒绝跨父会话查询、分配或收集。"""
        worker = self.store.get("worker", worker_id)
        if worker is None or worker["owner_session_id"] != session.state.agent_id or (
            worker["root_run_id"] != self._root_id(session)
        ):
            raise PermissionError("该 Worker 不属于当前会话的直接委派")
        return worker

    def _receipt_key(self, session: Session) -> str | None:
        """把幂等键绑定到会话及当前 Assignment，复用时不串回执。"""
        call_id = execution_id.get()
        scope = session.state.assignment_id or self._root_id(session)
        return f"{session.state.session_id}:{scope}:{call_id}" if call_id else None

    def _replay(self, session: Session, request: dict) -> dict | None:
        """返回同一次调用的固定响应，拒绝改变请求。"""
        key = self._receipt_key(session)
        saved = self.store.get("receipt", key) if key else None
        if saved and saved["request"] != request:
            return error("execution_conflict", "同一次工具调用不能改变参数")
        return saved["response"] if saved else None

    def _save_receipt(self, session: Session, request: dict, response: dict) -> dict:
        """在当前状态事务中一起保存响应快照。"""
        key = self._receipt_key(session)
        if key:
            self.store.put("receipt", key, {"request": request, "response": response})
        return response

    def _worker_names(self, parent: Session) -> tuple[str, ...]:
        """从宿主登记目录选择 Worker 能力，不继承 Router 的执行开关。"""
        from ..tools.builtin_tools.workers import WorkerTools

        available = set(parent.tool_registry.names()) | set(parent.loop.tools.names())
        for tool_type in parent.stateful_tool_types():
            available.update(parent.tool_registry.tool_names(tool_type))
        available.update(WorkerTools.tool_names)
        selection = self.worker_tools
        if selection is None:
            selection = role_tool_names("worker", available)
        policy = ToolPolicy(selection, available, parent.tool_registry.providers())
        if not policy.allows("finish"):
            raise ValueError("Worker 必须允许 finish")
        return tuple(sorted(policy.allowed_names))

    async def start_worker(self, parent: Session, worker_type: str) -> dict:
        """初始化空闲会话；失败时清理资源，不生成可用 Worker。"""
        from ..agent import SessionState

        self._require_open()
        request = {"tool": "StartWorker", "worker_type": worker_type}
        async with self._lock:
            if replay := self._replay(parent, request):
                return replay
            if not self._caller_running(parent):
                return error("caller_not_running", "当前会话没有正在执行的任务")
            if worker_type != "general":
                return error("unknown_worker_type", "批准的 Worker 类型只有 general")
            root_id = self._root_id(parent)
            existing = self.store.get("worker", parent.state.agent_id)
            depth = existing["depth"] + 1 if existing else 1
            if depth > self.limits.max_depth:
                return error("depth_limit", "Worker 委派层级已达到上限")
            if sum(w["root_run_id"] == root_id for w in self.store.records("worker")) >= (
                self.limits.max_workers
            ):
                return error("worker_limit", "Worker 数量已达到上限")
            identity = uuid4().hex
            state = SessionState(
                session_id=uuid4().hex, agent_id=identity,
                parent_agent_id=parent.state.agent_id, task_id=parent.state.task_id,
                workdir=parent.state.workdir,
            )
            names = self._worker_names(parent)
            if self.worker_skills:
                from ..skills import SkillContext

                if parent.skill_registry is None:
                    return error("skill_registry_missing", "缺少宿主批准的技能目录")
                context = SkillContext(parent.skill_registry, available_tools=names, role="worker")
                context.preload(self.worker_skills)
                state.skill_snapshots = context.snapshots()
            worker = {
                "worker_id": identity, "worker_type": worker_type, "type_version": 1, "depth": depth,
                "owner_session_id": parent.state.agent_id, "root_run_id": root_id,
                "status": "idle", "active_assignment_id": None,
                "last_collected_assignment_id": None, "checkpoint": state.model_dump_json(),
                "allow_tools": names, "skills": self.worker_skills,
            }
            try:
                self.store.put("worker", identity, worker)
                await self._ensure_session(worker, parent)
                response = {"ok": True, "worker_id": identity,
                            "worker_type": worker_type, "status": "idle"}
                self._save_receipt(parent, request, response)
                self.store.connection.commit()
                return response
            except asyncio.CancelledError:
                self.store.connection.rollback()
                await self._discard_session(identity)
                raise
            except Exception as exc:  # noqa: BLE001 - 初始化失败须清理全部资源。
                self.store.connection.rollback()
                await self._discard_session(identity)
                return error("initialization_failed", f"{type(exc).__name__}: {str(exc)[:500]}")

    async def _ensure_session(self, worker: dict, parent: Session) -> Session:
        """恢复稳定会话身份和历史，不重用父会话或活执行资源。"""
        from ..agent import SessionState

        identity = worker["worker_id"]
        if identity in self._sessions:
            return self._sessions[identity]
        # 恢复时只收紧旧快照，不能用过去的授权重新开放已经关闭的工具。
        permitted = set(self._worker_names(parent))
        worker["allow_tools"] = [name for name in worker["allow_tools"] if name in permitted]
        if "finish" not in worker["allow_tools"]:
            raise PermissionError("恢复后的 Worker 缺少结束工具授权")
        if parent.skill_registry is None:
            worker["skills"] = []
        self.store.put("worker", identity, worker)
        state = SessionState.model_validate_json(worker["checkpoint"])
        if parent.skill_registry is None:
            state.skill_snapshots.clear()
        self._repair_history(state)
        assignment = self.store.get("assignment", worker["active_assignment_id"]) if (
            worker["active_assignment_id"]
        ) else None
        request = ChildRequest(
            state=state, task=assignment["task"] if assignment else "",
            skills=tuple(worker["skills"]), allow_tools=tuple(worker["allow_tools"]),
            parent=parent, assignment_id=state.assignment_id or "",
        )
        expected = state.model_copy(deep=True)
        child = self.child_factory(parent, request)
        if inspect.isawaitable(child):
            child = await child
        if child is parent or any(getattr(child.state, field) != getattr(expected, field) for field in (
            "session_id", "agent_id", "parent_agent_id", "task_id", "workdir", "assignment_id",
        )):
            if child is not parent:
                await child.aclose()
            raise ValueError("Worker 工厂改变了宿主绑定的会话身份或工作目录")
        if child.loop.system_prompt != parent.loop.system_prompt or (
            child.skill_registry is not parent.skill_registry
            or tuple(child.initial_skills) != request.skills
            or child.state.skill_snapshots != expected.skill_snapshots
        ):
            await child.aclose()
            raise ValueError("Worker 工厂改变了基础规则或批准的技能配置")
        try:
            self.attach(child)
            await child.start()
            if not set(child.tool_policy.allowed_names) <= set(worker["allow_tools"]):
                raise PermissionError("Worker 工厂扩大了宿主授权")
            self.save_checkpoint(child, commit=False)
            return child
        except BaseException:
            await self._discard_session(identity)
            raise

    async def _discard_session(self, worker_id: str) -> None:
        """释放初始化失败或根运行结束的会话资源。"""
        child = self._sessions.pop(worker_id, None)
        self._signals.pop(worker_id, None)
        if child is not None:
            await child.aclose()

    async def assign_task(self, parent: Session, worker_id: str, task: str,
                          deliver_list: list[str]) -> dict:
        """原子分配一项工作并立即返回；同一 Worker 没有隐式任务队列。"""
        self._require_open()
        request = {"tool": "AssignTask", "worker_id": worker_id,
                   "task": task, "deliver_list": deliver_list}
        async with self._lock:
            if replay := self._replay(parent, request):
                return replay
            if not self._caller_running(parent):
                return error("caller_not_running", "当前会话没有正在执行的任务")
            worker = self._authorized(parent, worker_id)
            if worker["status"] != "idle":
                code = "worker_busy" if worker["status"] == "busy" else "result_not_collected"
                return error(code, "先等待并收集当前结果", next_action="CollectWorkerResult")
            if not task.strip() or len(task) > self.limits.max_task_chars or (
                not deliver_list or len(deliver_list) > 32
                or any(not item.strip() or len(item) > 2000 for item in deliver_list)
            ):
                return error("invalid_arguments", "任务与交付清单不能为空，且必须在长度限制内")
            identity = uuid4().hex
            assignment = {
                "assignment_id": identity, "worker_id": worker_id,
                "caller_assignment_id": parent.state.assignment_id,
                "task": task, "deliver_list": [
                    {"id": f"D{index}", "description": item}
                    for index, item in enumerate(deliver_list, 1)
                ],
                "artifact_root": f"artifacts/workers/{worker_id}/{identity}",
                "phase": "queued", "result_id": None, "collected": False,
                "finish_errors": 0,
            }
            worker.update(status="busy", active_assignment_id=identity)
            child = self._sessions.get(worker_id)
            if child:
                self._repair_history(child.state)
                child.state.assignment_id = identity
                worker["checkpoint"] = child.state.model_dump_json()
            with self.store.connection:
                self.store.put("assignment", identity, assignment)
                self.store.put("worker", worker_id, worker)
                response = self._save_receipt(parent, request, {
                    "ok": True, "worker_id": worker_id, "assignment_id": identity,
                    "status": "busy", "phase": "queued",
                    "deliverable_ids": [item["id"] for item in assignment["deliver_list"]],
                })
            self._schedule(worker_id, identity)
            return response

    def _schedule(self, worker_id: str, assignment_id: str) -> None:
        """只调度尚未启动的 Assignment，拒绝同一会话的并发驱动。"""
        if worker_id not in self._workers or self._workers[worker_id].done():
            self._workers[worker_id] = asyncio.create_task(self._execute(worker_id, assignment_id))

    @staticmethod
    def _repair_history(state: SessionState) -> None:
        """为中断调用补齐回执，复用会话时保持工具消息协议完整。"""
        repaired = []
        pending: dict[str, dict] = {}

        def flush() -> None:
            """只说明中断，不推断命令或写入是否实际成功。"""
            for identity in pending:
                repaired.append({
                    "role": "tool", "tool_call_id": identity,
                    "content": "上次调用中断，结果未确认；继续前先核对已有产物和执行记录。",
                })
            pending.clear()

        for message in state.messages:
            if message["role"] != "tool":
                flush()
            if message["role"] == "assistant":
                pending = {item["id"]: item for item in (message.get("tool_calls") or [])}
            elif message["role"] == "tool":
                pending.pop(message.get("tool_call_id"), None)
            repaired.append(message)
        flush()
        state.messages = repaired

    async def _execute(self, worker_id: str, assignment_id: str) -> None:
        """执行任务，等待不占槽位，异常生成可收集的终止记录。"""
        try:
            await self._slots.acquire()
            self._slot_holders.add(worker_id)
            worker = self.store.get("worker", worker_id)
            if worker["active_assignment_id"] != assignment_id or worker["status"] != "busy":
                return
            parent = self._sessions[worker["owner_session_id"]]
            child = await self._ensure_session(worker, parent)
            child.state.assignment_id = assignment_id
            assignment = self.store.get("assignment", assignment_id)
            assignment["phase"] = "running"
            with self.store.connection:
                self.store.put("assignment", assignment_id, assignment)
            if (
                child.loop.history_config.compact_on_worker_reassignment
                and worker["last_collected_assignment_id"] is not None
            ):
                await child.compact(keep_recent_turns=0)
                self.save_checkpoint(child)
            await child.run(self._assignment_context(child))
            if self.store.get("assignment", assignment_id)["phase"] != "ended":
                raise RuntimeError("Worker 退出但没有提交有效 finish 结果")
        except asyncio.CancelledError:
            if not self._closed:
                await self._terminate(worker_id, "cancelled", "任务已由宿主取消")
            raise
        except Exception as exc:  # noqa: BLE001 - 执行异常需保留为可收集结果。
            await self._terminate(worker_id, "failed", f"{type(exc).__name__}: {str(exc)[:1000]}")
        finally:
            if worker_id in self._slot_holders:
                self._slot_holders.remove(worker_id)
                self._slots.release()
            child = self._sessions.get(worker_id)
            if child:
                self.save_checkpoint(child)

    def _assignment_context(self, session: Session) -> str:
        """以宿主绑定的当前清单覆盖旧任务背景，所有 Worker 都能看到交付协议。"""
        if session.state.assignment_id is None:
            return ""
        assignment = self.store.get("assignment", session.state.assignment_id)
        if assignment is None:
            return ""
        return (
            "Current assignment (previous assignments are background only):\n"
            + json.dumps({key: assignment[key] for key in (
                "assignment_id", "task", "deliver_list", "artifact_root",
            )}, ensure_ascii=False)
            + "\nFinish with a JSON content string containing outcome, summary, deliverables, "
            "and limitations. Every assigned ID must appear exactly once. Each deliverable has "
            "id, status (provided or unavailable), content, and artifact_refs (real workspace files). "
            "outcome is completed, failed, blocked, or cancelled. completed requires every item "
            "provided. Keep the record under 12000 UTF-8 bytes; put long content in artifacts."
        )

    def _pending(self, session: Session) -> list[dict]:
        """只阻止当前 Assignment 尚未收尾的直属工作。"""
        return [worker for worker in self._children(session) if worker["status"] != "idle"
                and self.store.get("assignment", worker["active_assignment_id"])[
                    "caller_assignment_id"] == session.state.assignment_id]

    async def finish(self, session: Session, content: str) -> bool | str:
        """普通 Worker 校验交付，Router 返回用户内容；等待不能冒充结束。"""
        async with self._lock:
            if self._pending(session):
                return json.dumps(error("children_pending", "先等待并收集全部直属子结果"))
            if session.state.parent_agent_id is None:
                root = self.store.get("root", session.state.agent_id)
                root.update(status="completed", checkpoint=session.state.model_dump_json())
                with self.store.connection:
                    self.store.put("root", session.state.agent_id, root)
                return True
            assignment = self.store.get("assignment", session.state.assignment_id)
            try:
                result = validate_delivery(content, assignment, session.state.workdir)
            except (ValueError, OSError) as exc:
                assignment["finish_errors"] += 1
                if assignment["finish_errors"] >= self.limits.max_finish_repairs:
                    result = terminal_result(assignment, "failed", f"交付格式修复次数超限：{str(exc)[:500]}")
                else:
                    with self.store.connection:
                        self.store.put("assignment", assignment["assignment_id"], assignment)
                    return json.dumps(error("invalid_delivery", str(exc)[:1000]), ensure_ascii=False)
            self._complete(session.state.agent_id, assignment["assignment_id"], result)
            return True

    def _complete(self, worker_id: str, assignment_id: str, result: dict) -> None:
        """一起保存不可覆盖结果、终态与通知，迟到回调不能改变新任务。"""
        worker = self.store.get("worker", worker_id)
        assignment = self.store.get("assignment", assignment_id)
        if worker["active_assignment_id"] != assignment_id or assignment["result_id"]:
            return
        result_id, event_id = uuid4().hex, uuid4().hex
        assignment.update(phase="ended", result_id=result_id)
        worker["status"] = "completed"
        child = self._sessions.get(worker_id)
        if child:
            worker["checkpoint"] = child.state.model_dump_json()
        with self.store.connection:
            self.store.put("result", result_id, result)
            self.store.put("assignment", assignment_id, assignment)
            self.store.put("worker", worker_id, worker)
            self.store.put("event", event_id, {
                "event_id": event_id, "worker_id": worker_id, "assignment_id": assignment_id,
                "result_id": result_id, "outcome": result["outcome"],
                "receiver_session_id": worker["owner_session_id"],
                "root_run_id": worker["root_run_id"], "notified": False,
            })
        if worker["owner_session_id"] in self._signals:
            self._signals[worker["owner_session_id"]].set()

    def check_worker_status(self, session: Session) -> dict:
        """查询不消费事件或结果，completed 不等于成功。"""
        self._require_open()
        rows = []
        for worker in self._children(session):
            identity = worker["active_assignment_id"] or worker["last_collected_assignment_id"]
            assignment = self.store.get("assignment", identity) if identity else None
            result = self.store.get("result", assignment["result_id"]) if (
                assignment and assignment["result_id"]
            ) else None
            rows.append({
                "worker_id": worker["worker_id"], "worker_type": worker["worker_type"],
                "status": worker["status"], "assignment_id": identity,
                "task_summary": assignment["task"][:120] if assignment else "",
                "result_available": worker["status"] == "completed",
                "outcome": result["outcome"] if result else None,
            })
        return {"ok": True, "workers": rows}

    def _ready_response(self, session: Session, reason: str) -> dict:
        """从持久化状态构建就绪快照，与通知是否送达无关。"""
        ready, busy = [], []
        for worker in self._pending(session):
            if worker["status"] == "completed":
                assignment = self.store.get("assignment", worker["active_assignment_id"])
                result = self.store.get("result", assignment["result_id"])
                ready.append({"worker_id": worker["worker_id"],
                              "assignment_id": assignment["assignment_id"],
                              "result_id": assignment["result_id"], "outcome": result["outcome"]})
            else:
                busy.append(worker["worker_id"])
        return {"ok": True, "reason": reason, "ready_workers": ready, "busy_workers": busy}

    async def wait_workers(self, session: Session) -> dict:
        """单独等待任一结果，挂起时释放槽位，超时不改变任务状态。"""
        self._require_open()
        identity = session.state.agent_id
        async with self._lock:
            response = self._ready_response(session, "worker_ready")
            if response["ready_workers"]:
                self._mark_notified(session, {w["worker_id"] for w in response["ready_workers"]})
                return response
            if not response["busy_workers"]:
                return self._ready_response(session, "no_pending_work")
            signal = self._signals[identity]
            signal.clear()
            self._waiting.add(identity)
            self._set_phase(session, "waiting")
        released = identity in self._slot_holders
        if released:
            self._slot_holders.remove(identity)
            self._slots.release()
        reason = "worker_ready"
        try:
            async with asyncio.timeout(self.limits.wait_timeout_s):
                await signal.wait()
            if identity in self._interrupted:
                self._interrupted.discard(identity)
                reason = "interrupted"
        except TimeoutError:
            reason = "timeout"
        finally:
            self._waiting.discard(identity)
            current = asyncio.current_task()
            if released and not self._closed and current and not current.cancelling():
                await self._slots.acquire()
                self._slot_holders.add(identity)
            self._set_phase(session, "running")
        response = self._ready_response(session, reason)
        self._mark_notified(session, {w["worker_id"] for w in response["ready_workers"]})
        if reason == "interrupted":
            response["cause"] = "user_message"
        return response

    def _set_phase(self, session: Session, phase: str) -> None:
        """保持 busy 公共状态，只更新当前执行的内部阶段。"""
        if session.state.assignment_id and not self._closed:
            assignment = self.store.get("assignment", session.state.assignment_id)
            if assignment and assignment["phase"] != "ended":
                assignment["phase"] = phase
                with self.store.connection:
                    self.store.put("assignment", assignment["assignment_id"], assignment)

    def interrupt_wait(self, session: Session) -> None:
        """由宿主新用户输入唤醒等待，不自动取消已分配工作。"""
        identity = session.state.agent_id
        if identity in self._waiting:
            self._interrupted.add(identity)
            self._signals[identity].set()

    def is_waiting(self, session: Session) -> bool:
        """供宿主输入通道判断是否需要唤醒，不开放模型工具。"""
        return session.state.agent_id in self._waiting

    def _mark_notified(self, session: Session, worker_ids: set[str], *, commit: bool = True) -> None:
        """Wait 或 Collect 已覆盖的通知不再插入历史。"""
        with self.store.connection if commit else nullcontext():
            for event in self._events(session):
                if event["worker_id"] in worker_ids:
                    event["notified"] = True
                    self.store.put("event", event["event_id"], event)

    def _events(self, session: Session) -> list[dict]:
        """返回当前根运行发送给该会话的完成事件。"""
        return [event for event in self.store.records("event")
                if event["receiver_session_id"] == session.state.agent_id
                and event["root_run_id"] == self._root_id(session)]

    async def deliver(self, session: Session) -> None:
        """只注入宿主状态通知，完整交付必须显式收集。"""
        with self.store.connection:
            for event in self._events(session):
                if event["notified"]:
                    continue
                assignment = self.store.get("assignment", event["assignment_id"])
                event["notified"] = True
                if not assignment["collected"] and event["event_id"] not in (
                    session.state.consumed_event_ids
                ):
                    session.state.messages.append({
                        "role": "user",
                        "content": "Worker runtime notification (status only):\n" + json.dumps({
                            "source": "worker_runtime", **{key: event[key] for key in (
                                "event_id", "worker_id", "assignment_id", "result_id", "outcome",
                            )}, "next_action": "CollectWorkerResult",
                        }, ensure_ascii=False),
                    })
                    session.state.consumed_event_ids.append(event["event_id"])
                self.store.put("event", event["event_id"], event)
            self.save_checkpoint(session, commit=False)

    async def collect_worker_result(self, session: Session, worker_id: str) -> dict:
        """原子保存收集回执并释放 Worker，不把收集当作验收。"""
        self._require_open()
        request = {"tool": "CollectWorkerResult", "worker_id": worker_id}
        async with self._lock:
            if replay := self._replay(session, request):
                return replay
            worker = self._authorized(session, worker_id)
            if worker["status"] == "busy":
                return error("not_ready", "Worker 仍在执行；此工具不隐式等待")
            replayed = worker["status"] == "idle"
            identity = worker["last_collected_assignment_id"] if replayed else (
                worker["active_assignment_id"]
            )
            if not identity:
                return error("no_result", "Worker 尚未执行任务")
            assignment = self.store.get("assignment", identity)
            # finish 已发布结果后，等本次驱动者退出，防止新任务覆盖仍在退出的会话。
            driver = self._workers.get(worker_id)
            if driver is not None and not driver.done():
                await asyncio.shield(driver)
            response = {
                "ok": True, "worker_id": worker_id, "assignment_id": identity,
                "result_id": assignment["result_id"], "status": "idle", "replayed": replayed,
                "result": self.store.get("result", assignment["result_id"]),
            }
            assignment["collected"] = True
            worker.update(status="idle", active_assignment_id=None,
                          last_collected_assignment_id=identity)
            child = self._sessions.get(worker_id)
            if child:
                child.state.assignment_id = None
                worker["checkpoint"] = child.state.model_dump_json()
            with self.store.connection:
                self.store.put("worker", worker_id, worker)
                self.store.put("assignment", identity, assignment)
                self._mark_notified(session, {worker_id}, commit=False)
                self._save_receipt(session, request, response)
            return response

    def save_checkpoint(self, session: Session, *, commit: bool = True) -> None:
        """保存完整会话，和通知回执共用事务时由外层提交。"""
        kind = "worker" if session.state.parent_agent_id else "root"
        record = self.store.get(kind, session.state.agent_id)
        if record is None:
            return
        record["checkpoint"] = session.state.model_dump_json()
        self.store.put(kind, session.state.agent_id, record)
        if commit:
            self.store.connection.commit()

    def restore_state(self, agent_id: str) -> SessionState | None:
        """恢复宿主保存的会话，不声称恢复 Notebook 内存。"""
        from ..agent import SessionState

        record = self.store.get("worker", agent_id) or self.store.get("root", agent_id)
        return SessionState.model_validate_json(record["checkpoint"]) if record else None

    async def _terminate(self, worker_id: str, outcome: str, reason: str) -> None:
        """宿主终止递归清理后代，失败也必须生成可收集结果。"""
        for worker in self.store.records("worker"):
            if worker["owner_session_id"] == worker_id and worker["status"] == "busy":
                await self._cancel_worker(worker["worker_id"])
        worker = self.store.get("worker", worker_id)
        if worker and worker["status"] == "busy":
            assignment = self.store.get("assignment", worker["active_assignment_id"])
            self._complete(worker_id, assignment["assignment_id"],
                           terminal_result(assignment, outcome, reason))

    async def _cancel_worker(self, worker_id: str) -> None:
        """取消宿主选定执行并等待清理，无模型可见的取消工具。"""
        driver = self._workers.get(worker_id)
        if driver is not None and not driver.done() and driver is not asyncio.current_task():
            driver.cancel()
            await asyncio.gather(driver, return_exceptions=True)
        await self._terminate(worker_id, "cancelled", "任务已由宿主取消")

    async def cancel_children(self, parent_id: str) -> None:
        """由用户停止或宿主控制取消当前根运行的直属工作树。"""
        for worker in self.store.records("worker"):
            if worker["owner_session_id"] == parent_id and worker["status"] == "busy":
                await self._cancel_worker(worker["worker_id"])

    async def run(self, session: Session, content: str) -> str:
        """驱动根运行；恢复已排队任务，对不确定中断不静默重跑。"""
        self._require_open()
        self.attach(session)
        identity = session.state.agent_id
        if session.state.parent_agent_id is not None or identity in self._drivers:
            raise RuntimeError("根会话必须只有一个运行驱动者")
        root = self.store.get("root", identity)
        if root["status"] != "running":
            root.update(status="running", root_run_id=uuid4().hex)
            with self.store.connection:
                self.store.put("root", identity, root)
        self._drivers[identity] = asyncio.current_task()
        try:
            await session.start()
            for worker in self.store.records("worker"):
                if worker["root_run_id"] != root["root_run_id"] or worker["status"] != "busy":
                    continue
                assignment = self.store.get("assignment", worker["active_assignment_id"])
                if assignment["phase"] == "queued" and worker["owner_session_id"] == identity:
                    self._schedule(worker["worker_id"], assignment["assignment_id"])
                else:
                    await self._terminate(worker["worker_id"], "failed",
                                          "执行进程已中断，无法确认副作用；保留检查点，不自动重跑")
            return await session.run(content)
        except asyncio.CancelledError:
            if not self._closed:
                await self.cancel_children(identity)
                root = self.store.get("root", identity)
                root["status"] = "cancelled"
                with self.store.connection:
                    self.store.put("root", identity, root)
            raise
        finally:
            self.save_checkpoint(session)
            self._drivers.pop(identity, None)

    async def aclose(self) -> None:
        """停止活跃驱动，保存检查点，再关闭 Worker 资源与宿主存储。"""
        if self._closed:
            return
        current = asyncio.current_task()
        tasks = [task for task in (*self._drivers.values(), *self._workers.values())
                 if task is not current and not task.done()]
        self._closed = True
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for identity, session in list(self._sessions.items()):
            self.save_checkpoint(session)
            if session.state.parent_agent_id is not None:
                await self._discard_session(identity)
        self.store.close()

    async def __aenter__(self) -> Self:
        """允许宿主以异步上下文管理根运行资源。"""
        return self

    async def __aexit__(self, *_: object) -> None:
        """关闭执行与存储，不删除已有结果。"""
        await self.aclose()
