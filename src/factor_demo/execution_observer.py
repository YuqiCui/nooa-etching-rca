"""Observe upstream execution through public middleware without replacing its executor."""

import json

from .execution_observations import parent_execution


def install_execution_observer(agent):
    workspace = agent._workspace
    if workspace is None:
        return

    from .native_sandbox import sandbox_status

    workspace.emit("sandbox.selected", **sandbox_status())

    async def observe(ctx, nxt):
        journal = workspace.observations
        identity = journal.record(
            "orchestration", "running", code=ctx.code, executor="NOOA SandboxedExecutor"
        )
        token = parent_execution.set(identity)
        error = None
        try:
            return await nxt(ctx)
        except BaseException as caught:
            error = caught
            raise
        finally:
            parent_execution.reset(token)
            result = ctx.result
            children = journal.children(identity)
            if result is not None and children:
                evidence = [
                    {
                        k: row[k]
                        for k in (
                            "observation_id",
                            "kind",
                            "status",
                            "code",
                            "output",
                            "error",
                            "experiment_id",
                            "execution_ref",
                        )
                        if k in row
                    }
                    for row in children[-6:]
                ]
                for row in evidence:
                    for key in ("code", "output", "error"):
                        if key in row:
                            row[key] = workspace.preview(str(row[key]), 4000)
                result.stdout += "\n[Host execution observations]\n" + json.dumps(
                    evidence, ensure_ascii=False
                )
            doc = journal.get(identity)
            doc.update(
                status="failed" if error or (result and result.error) else "passed",
                output=(result.stdout + "\n" + result.stderr) if result else "",
                error=str(error or (result.error if result else "")),
            )
            journal.save(doc)

    agent.event_manager.intercept("execute_python", observe)
