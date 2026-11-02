"""Run a fresh Claude Code comparison against one existing public-data session."""

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import uvicorn
from dotenv import dotenv_values

from factor_demo.io_utils import write_json
from factor_demo.providers import Provider
from factor_demo.research_workspace import PythonWorkspace
from factor_demo.runtime import RuntimeSettings

from .constants import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    ANTHROPIC_MODEL,
    CLAUDE_CONFIG_DIR,
    CLAUDE_SUBAGENT_MODEL,
    DISABLE_TOOL_SEARCH,
    DISABLE_TRAFFIC,
    FACTOR_API_BASE,
    FACTOR_API_KEY,
    HOME_ENV,
    MAX_OUTPUT_TOKENS,
    MAX_THINKING_TOKENS,
    MCP_CONNECTION_NONBLOCKING,
    XDG_CONFIG_HOME,
)
from .gateway import create_gateway
from .tools import clean_environment


@contextmanager
def _gateway(provider, settings, directory):
    """Serve a private protocol adapter for the duration of one CLI invocation."""
    directory.mkdir()
    key = "sk-" + secrets.token_hex(24)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    app = create_gateway(provider, settings, key, directory)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("Experiment gateway did not start")
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}", key
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()


def _invoke_claude(provider, settings, run, directory, prompt, system_prompt, timeout, tools):
    """Stream CLI events to disk and kill its process group on timeout."""
    with (
        _gateway(provider, settings, directory) as (url, key),
        tempfile.TemporaryDirectory(prefix="nooa-claude-") as temporary,
    ):
        profile = Path(temporary)
        home, config = profile / "home", profile / "config"
        home.mkdir()
        config.mkdir()
        environment = clean_environment(dict(os.environ))
        environment.update(
            {
                HOME_ENV: str(home),
                XDG_CONFIG_HOME: str(home / ".config"),
                CLAUDE_CONFIG_DIR: str(config),
                DISABLE_TRAFFIC: "1",
                ANTHROPIC_BASE_URL: url,
                ANTHROPIC_API_KEY: key,
                ANTHROPIC_MODEL: provider.model,
                CLAUDE_SUBAGENT_MODEL: provider.model,
                MAX_OUTPUT_TOKENS: str(settings.max_output_tokens),
                MAX_THINKING_TOKENS: str(settings.anthropic_thinking_budget),
                DISABLE_TOOL_SEARCH: "false",
                MCP_CONNECTION_NONBLOCKING: "false",
            }
        )
        mcp = {"mcpServers": {}}
        if tools:
            mcp["mcpServers"]["research"] = {
                "command": sys.executable,
                "args": [
                    "-m",
                    "factor_demo.claude_experiment.tools",
                    str(run),
                    "--journal",
                    str(directory / "tool-calls.jsonl"),
                ],
                "alwaysLoad": True,
            }
        mcp_path = config / "mcp.json"
        write_json(mcp_path, mcp)
        settings_path = config / "settings.json"
        write_json(settings_path, {"disableAllHooks": True})
        command = [
            shutil.which("claude") or "claude",
            "--bare",
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--model",
            provider.model,
            "--effort",
            settings.reasoning_effort,
            "--setting-sources",
            "",
            "--settings",
            str(settings_path),
            "--strict-mcp-config",
            "--mcp-config",
            str(mcp_path),
            "--tools",
            "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--max-turns",
            "64" if directory.name == "formal" else "4",
            "--system-prompt",
            system_prompt,
        ]
        if tools:
            command.extend(["--allowedTools", "mcp__research__*"])
        started = time.time()
        with (
            (directory / "claude.jsonl").open("w") as output,
            (directory / "stderr.log").open("w") as errors,
        ):
            process = subprocess.Popen(
                command,
                cwd=run / "workspace",
                env=environment,
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=errors,
                text=True,
                start_new_session=True,
            )
            write_json(directory / "process.json", {"pid": process.pid, "started": started})
            try:
                process.communicate(input=prompt, timeout=timeout)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        events = [
            json.loads(line) for line in (directory / "claude.jsonl").read_text().splitlines()
        ]
        result = next((event for event in reversed(events) if event.get("type") == "result"), None)
        write_json(
            directory / "summary.json",
            {
                "elapsed_s": time.time() - started,
                "returncode": process.returncode,
                "result": result,
            },
        )
        if process.returncode or not result or result.get("is_error"):
            raise RuntimeError(f"Claude invocation failed; see {directory}")
        return result


def _prepare_run(baseline, run):
    """Copy public inputs only and verify every byte against the source session."""
    if run.exists():
        raise ValueError("Choose a new experiment directory")
    source = json.loads((baseline / "session.json").read_text())
    run.mkdir(parents=True)
    shutil.copytree(baseline / "input", run / "input")
    (run / "workspace/artifacts").mkdir(parents=True)
    hashes = {
        str(path.relative_to(run / "input")): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((run / "input").rglob("*"))
        if path.is_file()
    }
    if hashes != source["public_sha256"]:
        raise ValueError("Copied public inputs do not match the baseline hashes")
    source.pop("data", None)
    source["public_sha256"] = hashes
    write_json(run / "session.json", source)
    return len(hashes)


async def _preflight(run, baseline):
    """Reproduce the original host preflight without using a model."""
    initial = json.loads((baseline / "execution.jsonl").read_text().splitlines()[0])
    worker = PythonWorkspace(run)
    try:
        result = await worker.execute(initial["code"])
        if not result["ok"]:
            raise RuntimeError(f"Original public-data preflight failed: {result['output']}")
    finally:
        await worker.close()


def main() -> None:
    """Verify configuration, probe the real model, and run one serial experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    baseline, run = args.baseline.resolve(), args.run.resolve()
    saved = json.loads((baseline / "provider.json").read_text())
    environment = dotenv_values(Path.cwd() / ".env")
    if environment[FACTOR_API_BASE] != saved["base_url"]:
        raise ValueError("Current credential endpoint differs from the original run")
    provider = Provider(
        "custom", saved["model"], saved["base_url"], "openai", environment[FACTOR_API_KEY]
    )
    raw_settings = json.loads((baseline / "runtime.json").read_text())
    settings = RuntimeSettings.model_validate(
        {name: raw_settings[name] for name in RuntimeSettings.model_fields}
    )
    count = _prepare_run(baseline, run)
    message = (baseline / "rounds/round_001/message.md").read_text()
    (run / "message.md").write_text(message)
    manifest = {
        "baseline": str(baseline),
        "run": str(run),
        "model": provider.model,
        "upstream_base_url_matches": True,
        "credential_source": ".env:FACTOR_API_KEY",
        "historical_key_equality": "Original run did not store a credential fingerprint",
        "data_hashes_verified": count,
        "timeout_s": args.timeout,
        "runtime": settings.model_dump(),
        "execution_environment": "escalated",
        "concurrency": {
            "experiments": 1,
            "main_agents": 1,
            "model_in_flight": 1,
            "subagents": 0,
            "reviewers": 0,
        },
        "budget": {"max_turns": 64, "max_model_calls": 256, "max_total_tokens": 4_000_000},
        "status": "preflight",
        "started": time.time(),
    }
    write_json(run / "manifest.json", manifest)
    try:
        asyncio.run(_preflight(run, baseline))
        print("Public-data preflight passed.", flush=True)
        _invoke_claude(
            provider,
            settings,
            run,
            run / "probe",
            "Use run_code to execute print('TOOL_PROBE_OK'), then reply OK.",
            "Use the research MCP tools to fulfill the user request. Reply briefly.",
            180,
            True,
        )
        probe_tools = run / "probe/tool-calls.jsonl"
        if not probe_tools.exists() or not any(
            row.get("tool") == "run_code" and row.get("ok")
            for row in map(json.loads, probe_tools.read_text().splitlines())
        ):
            raise RuntimeError("The real model probe did not execute the scientific worker")
        manifest.update(status="running", probe_verified=True)
        write_json(run / "manifest.json", manifest)
        print("Real Claude Code / DeepSeek probe passed; starting the research task.", flush=True)
        knowledge = Path(__file__).parents[1] / "knowledge"
        system = (
            "You are a scientific research agent. Use the provided research MCP tools. "
            "Inspect the dataset schema first. All Python execution must use run_code or "
            "code_write/code_edit/code_run or write_file/run_script in the guarded worker. "
            "DATASET is the selected read-only pathlib.Path. The worker cwd is workspace. "
            "Variables persist between run_code calls; scripts have a separate namespace. "
            "Use installed scientific libraries, not guessed APIs. Inspect failed tool output "
            "before repair. Save analysis code, numeric evidence, plots and a final report "
            "under artifacts/. Only claim findings backed by computed evidence. Distinguish "
            "association from causality. Do not access hidden truth or other experiments. "
            "This is a single-agent experiment; no subagents or additional model calls. "
            "Return a concise Chinese final handoff with verified artifact paths.\n\n"
        )
        for name in ("physics", "statistics"):
            system += (knowledge / name / "SKILL.md").read_text() + "\n\n"
        (run / "system-prompt.txt").write_text(system)
        result = _invoke_claude(
            provider, settings, run, run / "formal", message, system, args.timeout, True
        )
        if not (run / "formal/tool-calls.jsonl").exists():
            raise RuntimeError("Claude returned without executing any business tools")
        (run / "final-response.md").write_text(result.get("result", ""))
        manifest.update(status="completed", finished=time.time())
    except BaseException as error:
        manifest.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error).replace(provider.api_key, "[REDACTED]"),
            finished=time.time(),
        )
        raise
    finally:
        write_json(run / "manifest.json", manifest)


if __name__ == "__main__":
    main()
