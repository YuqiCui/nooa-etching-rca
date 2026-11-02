"""Expose existing business tools over MCP without granting filesystem tools."""

import argparse
import inspect
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from nooa.unifiedllm import FakeLLMClient

from factor_demo.autoresearch_agent import AutoresearchAgent
from factor_demo.research_workspace import PythonWorkspace

from .constants import ISOLATED_PREFIXES, SECRET_MARKERS


def clean_environment(environment: dict[str, str]) -> dict[str, str]:
    """Remove provider configuration and credentials before starting tool processes.

    Args:
        environment: Parent process environment.

    Returns:
        A copy without model settings or credential-like variable names.
    """
    return {
        key: value
        for key, value in environment.items()
        if not key.startswith(ISOLATED_PREFIXES)
        and not any(marker in key.upper() for marker in SECRET_MARKERS)
    }


def business_tools(agent: AutoresearchAgent) -> dict:
    """Select deterministic research operations; exclude all NOOA model methods.

    Args:
        agent: Existing tool owner with an isolated scientific workspace.

    Returns:
        An explicit registry of the methods available to Claude Code.
    """
    return {
        "inspect_dataset": agent.inspect_dataset,
        "list_datasets": agent.list_datasets,
        "read_dataset_lineage": agent.read_dataset_lineage,
        "run_code": agent.run_code,
        "code_write": agent.code_write,
        "code_edit": agent.code_edit,
        "code_read": agent.code_read,
        "code_list": agent.code_list,
        "code_run": agent.code_run,
        "write_file": agent.write_file,
        "run_script": agent.run_script,
        "list_executions": agent.list_executions,
        "read_execution": agent.read_execution,
        "read_observation": agent.read_observation,
        "list_artifacts": agent.list_artifacts,
        "read_artifact": agent.read_artifact,
    }


def create_server(run: Path, journal: Path | None = None) -> FastMCP:
    """Create an MCP server with the application's original guarded tools.

    Args:
        run: Experiment directory containing only public inputs and fresh outputs.
        journal: Separate probe or formal tool journal; defaults to the run root.

    Returns:
        A stdio server whose lifespan closes the scientific worker.
    """
    workspace = PythonWorkspace(run)
    # NOOA owns the tools, but generation must never use a second model backend.
    agent = AutoresearchAgent(workspace, llm=FakeLLMClient(strict_exhaustion=True))

    @asynccontextmanager
    async def lifespan(server):
        try:
            yield {}
        finally:
            await workspace.close()
            await agent.aclose()

    server = FastMCP("research", lifespan=lifespan)
    for name, method in business_tools(agent).items():
        server.add_tool(
            _journaled_tool(method, name, journal or run / "tool-calls.jsonl"),
            name=name,
            description=inspect.getdoc(method),
            structured_output=False,
        )
    return server


def _journaled_tool(method, name, journal):
    """Preserve typed tool signatures while recording timings and failures."""

    async def call(**arguments):
        started = time.time()
        record = {"tool": name, "started": started, "arguments": arguments}
        try:
            result = await method(**arguments)
            record.update(result=result, ok=not isinstance(result, dict) or result.get("ok", True))
            return result
        except Exception as error:
            record.update(ok=False, error_type=type(error).__name__, error=str(error))
            raise
        finally:
            record["elapsed_s"] = time.time() - started
            with journal.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    call.__name__ = name
    call.__signature__ = inspect.signature(method)
    call.__annotations__ = inspect.get_annotations(method, eval_str=True)
    return call


def main() -> None:
    """Start the credential-free business tool subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--journal", type=Path)
    args = parser.parse_args()
    safe = clean_environment(dict(os.environ))
    os.environ.clear()
    os.environ.update(safe)
    create_server(args.run.resolve(), args.journal).run(transport="stdio")


if __name__ == "__main__":
    main()
