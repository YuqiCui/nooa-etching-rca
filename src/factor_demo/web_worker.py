"""One isolated NOOA host process per active browser turn."""

import argparse
import asyncio
import json
import os
from pathlib import Path

import psutil

from .autoresearch import research_turn
from .live_events import LiveEvents, branch_path
from .research_branches import BranchStore


async def main(args):
    request = json.loads((args.run / "requests" / f"{args.task}.json").read_text())

    identity = request.get("agent_id", "main")
    local = args.run if identity == "main" else branch_path(args.run, identity)
    store = BranchStore(args.run)
    row = store.get(identity) if identity != "main" else None
    events = LiveEvents(
        local,
        args.task,
        root=args.run,
        agent_id=identity,
        parent_agent_id=row["parent_id"] if row else "",
    )
    if row:
        store.update(
            identity,
            mission=args.task,
            state="running",
            pid=os.getpid(),
            process_created=psutil.Process().create_time(),
            error=None,
        )
    try:
        ui_path = args.run / "ui.json"
        ui = json.loads(ui_path.read_text()) if ui_path.exists() else {}
        runner = research_turn
        if identity == "main" and ui.get("engine") == "autoresearch_v2":
            from .autoresearch_web import research_turn as core_research_turn

            runner = core_research_turn
        await runner(
            local,
            request["provider"],
            request["message"],
            request["mode"],
            events=events,
            live=True,
            timeout=request.get("timeout", 3600),
        )
    except asyncio.CancelledError:
        events.emit("task.cancelled")
    except Exception as error:
        events.emit(
            "task.error", error_type=type(error).__name__, message="本轮中断，已有记录已保存。"
        )
        raise
    finally:
        if row:
            status = (
                json.loads((local / "status.json").read_text())
                if (local / "status.json").exists()
                else {}
            )
            cancelled = (args.run / "requests" / f"{args.task}.cancel").exists() or (
                local / "requests" / f"{args.task}.cancel"
            ).exists()
            store.update(
                identity,
                state="cancelled"
                if cancelled
                else "completed"
                if status.get("status") == "checkpoint"
                else "interrupted",
                review_status=status.get("review_status"),
            )
            events.emit("branch.state", branch=store.get(identity))
        events.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--task", required=True)
    asyncio.run(main(p.parse_args()))
