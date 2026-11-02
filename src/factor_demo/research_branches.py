"""Durable branch identities with native NOOA channels and isolated host processes."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

import psutil
from nooa import QueueManager

from .autoresearch_agent_v2.database import create_branch_service
from .io_utils import write_json
from .live_events import LiveEvents, artifact_index, branch_path, public_artifact_path

TERMINAL = {"completed", "cancelled", "failed", "interrupted"}


class BranchStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self._storage = create_branch_service(self.root)

    def list(self):
        return self._storage.list()

    def get(self, identity):
        return self._storage.get(identity)

    def create(self, identity, parent, mission, title, task):
        self._storage.create(identity, parent, mission, title, task)

    def update(self, identity, **values):
        self._storage.update(identity, values)

    def process(self, row):
        if not row.get("pid"):
            return None
        try:
            proc = psutil.Process(row["pid"])
            if abs(proc.create_time() - row["process_created"]) > 0.01:
                return None
            args = proc.cmdline()
            child_match = (
                "factor_demo.branch_worker" in args and str(self.root) in args and row["id"] in args
            )
            controller_match = False
            marker = self.root / "active-mission.json"
            if "factor_demo.web_worker" in args and marker.exists():
                data = json.loads(marker.read_text())
                controller_match = (
                    data.get("pid") == row["pid"]
                    and data.get("agent_id") == row["id"]
                    and str(self.root) in args
                )
            if not (child_match or controller_match):
                return None
            return proc if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE else None
        except (psutil.Error, ValueError, OSError):
            return None

    def reconcile(self):
        for row in self.list():
            if row["state"] not in TERMINAL and not self.process(row):
                run = branch_path(self.root, row["id"])
                status = (
                    json.loads((run / "status.json").read_text())
                    if (run / "status.json").exists()
                    else {}
                )
                cancelled = (run / "requests" / f"{row['mission']}.cancel").exists()
                state = (
                    "cancelled"
                    if cancelled
                    else "completed"
                    if status.get("mission") == row["mission"]
                    and status.get("status") == "checkpoint"
                    else "interrupted"
                )
                self.update(
                    row["id"],
                    state=state,
                    review_status=status.get("review_status") if state == "completed" else None,
                    error=None
                    if state == "completed"
                    else "Host process ended; resume explicitly from saved history/files",
                )


def prepare_branch(root, identity, parent, task, title, evidence_paths, dataset_ref=None):
    run = root / "branches" / identity
    run.mkdir(parents=True)
    (run / "input").symlink_to((root / "input").resolve(), target_is_directory=True)
    (run / "workspace/artifacts").mkdir(parents=True)
    config = json.loads((root / "session.json").read_text())
    write_json(run / "session.json", config)
    write_json(
        run / "branch.json",
        {
            "agent_id": identity,
            "parent_agent_id": parent,
            "title": title,
            "task": task,
            "dataset_ref": dataset_ref,
        },
    )
    if dataset_ref:
        from .datasets import DatasetRegistry

        DatasetRegistry(root).bind(identity, dataset_ref)
    seeds = []
    for index, name in enumerate(evidence_paths):
        source = public_artifact_path(root, name)
        target = run / "workspace/artifacts/seed" / f"{index:02d}-{source.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        seeds.append(
            {
                "source": name,
                "path": str(target.relative_to(run / "workspace")),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    return run, seeds


class ResearchManager:
    def __init__(self, agent, root, mission, identity, policy, budget, provider, settings_path):
        self.agent = agent
        self.root = root.resolve()
        self.mission = mission
        self.identity = identity
        self.policy = policy
        self.budget = budget
        self.provider = provider
        self.settings_path = settings_path.resolve()
        self.events = agent._workspace.events
        self.store = BranchStore(self.root)
        if identity == "main":
            self.store.reconcile()
        self.queue = QueueManager(event_manager=agent.event_manager)
        self.guidance = self.queue.event(
            "expert_guidance", preview=lambda value: str(value)[:12000]
        )
        self.gates = asyncio.Semaphore(policy.max_concurrent_branches)
        self.handles = {}
        self.processes = {}
        self.closing = False

    def list(self):
        return [
            {k: v for k, v in r.items() if k not in {"pid", "process_created"}}
            for r in self.store.list()
        ]

    def context_summary(self):
        rows = self.store.list()
        chosen = {r["id"]: r for r in rows[-24:]}
        chosen.update({r["id"]: r for r in rows if r["state"] not in TERMINAL})
        return json.dumps(
            {
                "total_branches": len(rows),
                "recent_and_active": [
                    {
                        "agent_id": r["id"],
                        "title": r["title"],
                        "task": r["task"][:400],
                        "state": r["state"],
                        "review_status": r["review_status"],
                    }
                    for r in chosen.values()
                ],
                "history_tool": "Use list_research/read_branch_result for older saved branches",
            },
            ensure_ascii=False,
        )

    def _own(self, identity):
        row = self.store.get(identity)
        if self.identity != "main" and row["parent_id"] != self.identity:
            raise PermissionError("Cannot mutate another researcher branch")
        return row

    def _can_spawn(self):
        if self.closing:
            raise RuntimeError("Research coordinator is closing")
        if self.identity != "main":
            raise PermissionError(
                "This mission permits root-directed research branches only; request further decomposition from the parent"
            )

    async def spawn(self, task, title="", evidence_paths=None, dataset_ref=None):
        self._can_spawn()
        if not task.strip() or len(task) > 30000:
            raise ValueError("Provide a focused research question (1-30000 characters)")
        existing = [r for r in self.store.list() if r["mission"] == self.mission]
        for row in existing:
            if row["task"] == task and row["state"] not in TERMINAL:
                return {**self.result(row["id"]), "duplicate": True}
        if len(existing) >= self.policy.max_branches_per_turn:
            raise RuntimeError("Mission branch count budget exhausted")
        for name in evidence_paths or []:
            public_artifact_path(self.root, name)
        identity = "research-" + uuid.uuid4().hex[:10]
        datasets = getattr(self.agent, "_datasets", None)
        dataset_ref = dataset_ref or (datasets.current()["id"] if datasets else None)
        title = (title or task[:60])[:80]
        _run, seeds = prepare_branch(
            self.root, identity, self.identity, task, title, evidence_paths or [], dataset_ref
        )
        self.store.create(identity, self.identity, self.mission, title, task)
        self.events.emit("branch.created", branch=self.store.get(identity))
        message = task + "\n\n这是独立研究分支。返回可复核EvidenceRef，保留反证，不假设读过父会话。"
        if dataset_ref:
            message += (
                "\n绑定DatasetRef="
                + dataset_ref
                + "。使用worker内DATASET路径；保持你的独立数据范围。"
            )
        if seeds:
            message += "\n父任务提供的待复核材料（不是隐藏真值）：" + json.dumps(
                seeds, ensure_ascii=False
            )
        self._start(identity, message)
        return self.result(identity)

    def _start(self, identity, message):
        folder = branch_path(self.root, identity) / "requests"
        folder.mkdir(exist_ok=True)
        (folder / f"{self.mission}.cancel").unlink(missing_ok=True)
        self.queue.queue(identity, replace=True)
        handle = self.queue.spawn(
            self._execute(identity, message),
            channel=identity,
            buffer=1,
            label=self.store.get(identity)["title"],
        )
        self.handles[identity] = handle

    async def _execute(self, identity, message):
        run = branch_path(self.root, identity)
        proc = None
        try:
            async with self.gates:
                if (
                    self.closing
                    or self.store.get(identity)["state"] == "cancelled"
                    or (run / "requests" / f"{self.mission}.cancel").exists()
                ):
                    raise asyncio.CancelledError
                self.store.update(identity, state="starting", mission=self.mission, error=None)
                request = run / "requests" / f"{self.mission}.json"
                request.parent.mkdir(exist_ok=True)
                write_json(
                    request,
                    {
                        "message": message,
                        "provider": self.provider,
                        "settings_path": str(self.settings_path),
                        "admission_path": str(self.budget.path),
                        "policy": self.policy.model_dump(),
                    },
                )
                cancel = run / "requests" / f"{self.mission}.cancel"
                with (request.with_suffix(".log")).open("ab") as log:
                    launch = asyncio.create_task(
                        asyncio.create_subprocess_exec(
                            sys.executable,
                            "-m",
                            "factor_demo.branch_worker",
                            "--root",
                            str(self.root),
                            "--agent",
                            identity,
                            "--mission",
                            self.mission,
                            cwd=Path(__file__).resolve().parents[2],
                            stdout=log,
                            stderr=log,
                            start_new_session=True,
                        )
                    )
                    try:
                        proc = await asyncio.shield(launch)
                    except asyncio.CancelledError:
                        proc = await launch
                        raise
                    self.processes[identity] = proc
                    self.store.update(
                        identity,
                        state="running",
                        pid=proc.pid,
                        process_created=psutil.Process(proc.pid).create_time(),
                    )
                    self.events.emit("branch.state", branch=self.store.get(identity))
                    waiter = asyncio.create_task(proc.wait())
                    while not waiter.done():
                        _progress, last = self.progress(identity)
                        if (
                            time.time() - max(last, self.store.get(identity)["updated"])
                            > self.policy.wait_no_progress_s
                        ):
                            await self._stop_process(identity, proc)
                            raise TimeoutError("Branch made no observable progress")
                        await asyncio.wait({waiter}, timeout=1)
                    code = await waiter
                status = (
                    json.loads((run / "status.json").read_text())
                    if (run / "status.json").exists()
                    else {}
                )
                review = (
                    json.loads((run / "latest_review.json").read_text())
                    if (run / "latest_review.json").exists()
                    else {}
                )
                state = (
                    "cancelled"
                    if cancel.exists()
                    else "completed"
                    if code == 0 and status.get("status") == "checkpoint"
                    else "failed"
                )
                self.store.update(
                    identity,
                    state=state,
                    review_status=review.get("status"),
                    error=None
                    if state == "completed"
                    else status.get("error_type", "Research process ended"),
                )
        except asyncio.CancelledError:
            await self._stop_process(identity, proc)
            self.store.update(identity, state="cancelled", error="Cancelled by coordinator")
            raise
        except Exception as error:  # noqa: BLE001 - persist one failed branch without aborting siblings
            await self._stop_process(identity, proc)
            self.store.update(identity, state="failed", error=type(error).__name__)
        finally:
            self.budget.ledger.abandon_agent(identity)
            self.processes.pop(identity, None)
            self.events.emit("branch.state", branch=self.store.get(identity))
        value = self.result(identity)
        self.queue.event(
            "result-" + identity,
            replace=True,
            preview=lambda item: json.dumps(item, ensure_ascii=False)[:4000],
        ).put(
            {
                "agent_id": identity,
                "state": value["state"],
                "review_status": value.get("review_status"),
                "summary": value.get("report", {}).get("summary", "")[:2000],
            }
        )
        return value

    async def _stop_process(self, identity, proc):
        run = branch_path(self.root, identity)
        folder = run / "requests"
        folder.mkdir(exist_ok=True)
        (folder / f"{self.mission}.cancel").touch()
        if proc and proc.returncode is None:
            children = []
            with contextlib.suppress(psutil.Error):
                children = psutil.Process(proc.pid).children(recursive=True)
            try:
                await asyncio.wait_for(asyncio.shield(proc.wait()), 5)
            except TimeoutError:
                for child in reversed(children):
                    with contextlib.suppress(psutil.Error):
                        child.kill()
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()

    def result(self, identity):
        row = self.store.get(identity)
        run = branch_path(self.root, identity)
        value = {k: v for k, v in row.items() if k not in {"pid", "process_created"}}
        value["agent_id"] = identity
        if row["state"] in TERMINAL:
            if (run / "latest_reply.json").exists():
                payload = json.loads((run / "latest_reply.json").read_text())
                if payload.get("mission") == row["mission"]:
                    value["report"] = payload
            if (run / "latest_review.json").exists():
                value["review"] = json.loads((run / "latest_review.json").read_text())
            value["artifacts"] = [
                {**f, "path": f"branches/{identity}/{f['path']}"} for f in artifact_index(run)
            ]
            if "report" in value:
                report = value["report"]
                for h in report.get("hypotheses", []):
                    h["evidence_files"] = [f"branches/{identity}/" + p for p in h["evidence_files"]]
                for e in report.get("evidence", []):
                    e["artifact"] = f"branches/{identity}/" + e["artifact"]
                report["artifacts"] = [
                    f"branches/{identity}/" + p for p in report.get("artifacts", [])
                ]
        return value

    def progress(self, identity):
        return self.events.progress(identity, self.mission)

    async def wait(self, identities=None, timeout_s=30):
        identities = identities or list(self.handles)
        for identity in identities:
            self.store.get(identity)
        if not identities:
            return []
        try:
            async with asyncio.timeout(min(60, max(1, timeout_s))):
                while any(self.store.get(i)["state"] not in TERMINAL for i in identities):
                    if not self.queue.channels():
                        await asyncio.sleep(0.2)
                    else:
                        try:
                            await asyncio.wait_for(self.queue.race(), 0.25)
                        except TimeoutError:
                            pass
        except TimeoutError:
            pass
        results = [self.result(i) for i in identities]
        for row in results:
            row["progress_revision"] = self.progress(row["agent_id"])[0]
        return results

    def guide(self, identity, message):
        row = self._own(identity)
        if row["state"] in TERMINAL:
            raise ValueError("Branch is inactive; resume it with the new guidance")
        e = LiveEvents(
            branch_path(self.root, identity),
            row["mission"],
            root=self.root,
            agent_id=identity,
            parent_agent_id=row["parent_id"],
        )
        try:
            e.add_guidance(message, source="parent:" + self.identity)
            e.emit("branch.guidance", message=message, source_agent=self.identity)
        finally:
            e.close()
        return {"agent_id": identity, "queued": True}

    async def cancel(self, identity):
        self._own(identity)
        if identity in self.handles:
            await asyncio.sleep(0)
            await self.handles[identity].cancel()
        else:
            row = self.store.get(identity)
            proc = self.store.process(row)
            (branch_path(self.root, identity) / "requests" / f"{row['mission']}.cancel").touch()
            if proc:
                children = proc.children(recursive=True)
                await asyncio.sleep(0.3)
                if self.store.process(row):
                    for child in reversed(children):
                        with contextlib.suppress(psutil.Error):
                            child.kill()
                    with contextlib.suppress(psutil.Error):
                        proc.kill()
            self.store.update(identity, state="cancelled", error="Cancelled explicitly")
        self.events.emit("branch.state", branch=self.store.get(identity))
        return self.result(identity)

    async def resume(self, identity, message):
        self._can_spawn()
        row = self._own(identity)
        if row["state"] not in TERMINAL:
            raise ValueError("Branch is already active")
        self.store.update(
            identity, state="queued", mission=self.mission, review_status=None, error=None
        )
        self._start(identity, message)
        return self.result(identity)

    async def shutdown(self, wait=True):
        if wait:
            for identity, handle in list(self.handles.items()):
                if handle.state == "running":
                    try:
                        await self.wait([identity], 60)
                    except Exception as error:  # noqa: BLE001 - cleanup must still run
                        self.events.emit("branch.cleanup_error", error_type=type(error).__name__)
        self.closing = True
        await self.queue.shutdown()
