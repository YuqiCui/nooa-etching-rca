"""使用基线冻结代码和临时提示词，以纯 Notebook 原生重跑 case_03。"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import uuid
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

PROJECT = Path(__file__).resolve().parents[2]
BASELINE = PROJECT / "runs/case03-native-rerun-20261102-173817"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--preflight", action="store_true")
parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high", "max"])
parser.add_argument("--timeout", type=float, default=7200, help="整次研究时限，单位秒")
parser.add_argument("--with-verifier", action="store_true", help="使用当前代码和提示词联动验证器")
ARGS = parser.parse_args()
ROOT = ARGS.run_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
if not (ROOT / "source").exists():
    if ARGS.with_verifier:
        shutil.copytree(PROJECT / "src/factor_demo", ROOT / "source/factor_demo",
                        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    else:
        shutil.copytree(
            BASELINE / "source", ROOT / "source", ignore=shutil.ignore_patterns("__pycache__")
        )
sys.path.insert(0, str(ROOT / "source"))

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.agent.prompts import SYSTEM_PROMPT
from factor_demo.autoresearch_agent_v2.config import HistoryConfig, ModelConfig
from factor_demo.autoresearch_agent_v2.python_tools import PythonTools
from factor_demo.autoresearch_agent_v2.sandboxing import Sandbox, SandboxConfig
from factor_demo.autoresearch_agent_v2.tools import FileTools, SandboxTools, TaskTools
from factor_demo.providers import load_provider

if ARGS.with_verifier:
    from factor_demo.autoresearch_agent_v2.feature_verifier import FeatureAnalyzeVerifier
    from factor_demo.autoresearch_agent_v2.feature_verifier.prompts import VERIFIER_PROMPT

WORK = ROOT / "workspace"
PROMPT = (SYSTEM_PROMPT if ARGS.with_verifier else
          (PROJECT / "configs/experiments/notebook_only_system_prompt.txt").read_text())


def now():
    """返回可核对的 UTC 时间。"""
    return datetime.now(UTC).isoformat()


def save(name, value):
    """原子保存无凭据的状态。"""
    target = ROOT / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)


def append(name, value):
    """追加调用或用量记录。"""
    with (ROOT / name).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def register(session, workdir, tasks=True):
    """配置当前平台内部沙箱，不授予网络或额外业务目录权限。"""
    session.register_resource(
        Sandbox,
        partial(
            SandboxTools.create_sandbox,
            config=SandboxConfig(workdir=workdir, backend="macos", timeout_s=120),
        ),
    )
    if tasks:
        TaskTools.register(session, storage_root=ROOT / "task_storage")
    session.register_stateful_tool(FileTools)
    session.register_stateful_tool(PythonTools)
    if ARGS.with_verifier:
        session.register_stateful_tool(FeatureAnalyzeVerifier)


async def preflight():
    """先验证原生 Notebook 和科学库，避免在不可用环境中消耗模型调用。"""
    workdir = ROOT / "preflight"
    workdir.mkdir(exist_ok=True)
    state = SessionState(
        session_id="preflight", agent_id="preflight", task_id="preflight", workdir=workdir
    )
    model = Model(ModelConfig(model="openai/fixture", api_key="fixture"))
    session = Session(model, state, system_prompt=PROMPT)
    register(session, workdir, tasks=False)
    async with session:
        schemas = session.loop.tools.schemas()
        names = [item["function"]["name"] for item in schemas]
        assert not any("run_shell" in name or "run_python" in name for name in names)
        save("enabled_tools.json", schemas)
        (ROOT / "effective_system_prompt.txt").write_text(session.loop.build_system_prompt())
        created = await session.call(
            "notebook__create_notebook", {"notebook_name": "native_preflight"}
        )
        save("preflight_created.json", created)
        if not created.get("connected"):
            raise RuntimeError("原生 Notebook 启动失败，未调用研究模型")
        cell = await session.call(
            "notebook__write_new_cell",
            {
                "notebook_name": "native_preflight",
                "source": "import pandas as pd, numpy as np, pyarrow, scipy\nfrom pathlib import Path\nPath('native_probe.txt').write_text('ok')\nprint('native_notebook_science_ok', pd.__version__)",
            },
        )
        result = await session.call(
            "notebook__exec_cell",
            {
                "notebook_name": "native_preflight",
                "cell_id": cell["cell_id"],
                "revision": cell["revision"],
            },
        )
        save("preflight_result.json", result)
        if result.get("status") != "completed":
            raise RuntimeError("原生科学计算预检失败，未调用研究模型")
    print("原生 Notebook、科学库和内部沙箱预检通过", flush=True)


async def run():
    """按原任务启动新会话，不注入旧结果、真值或运行中提示。"""
    if (ROOT / "status.json").exists():
        raise RuntimeError("已有运行状态，拒绝覆盖实验")
    spec = importlib.util.spec_from_file_location("native_trace_helper", ROOT / "trace_helper.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    provider = load_provider("custom")
    records = []

    def usage(value):
        """逐次记录服务端报告用量，不估算未报告费用。"""
        row = {"received_at": now(), "usage": value}
        records.append(row)
        append("token_usage.jsonl", row)
        summary = helper._summarize_usage(records)
        save("token_summary.json", summary)
        print("response", len(records), "reported_tokens", summary["total_tokens"], flush=True)

    config = ModelConfig(
        model="openai/glm-5.3-flash",
        api_key=provider.api_key,
        base_url=provider.base_url,
        max_tokens=None,
        num_retries=10,
        timeout_s=180,
        thinking={"type": "enabled", "clear_thinking": False},
        reasoning_effort=ARGS.reasoning_effort,
        tool_choice="required",
    )
    state = SessionState(
        session_id=uuid.uuid4().hex, agent_id="new", task_id="case_03", workdir=WORK
    )
    model = helper._TracingModel(config, state, ROOT, usage)
    session = Session(
        model,
        state,
        system_prompt=PROMPT,
        history_config=HistoryConfig(context_window_tokens=272000),
    )
    register(session, WORK)
    session.register_dataset(
        "DATASET",
        str(WORK / "input"),
        "固定配方加工的三个业务时段、同期参考腔、原始时序与统计特征；先检查数据结构。",
    )
    if ARGS.with_verifier:
        (ROOT / "effective_verifier_prompt.txt").write_text(VERIFIER_PROMPT)
    save("run_config.json", config.model_dump(mode="json", exclude={"api_key", "base_url"}))
    status = {
        "status": "running",
        "started_at": now(),
        "pid": os.getpid(),
        "platform": "native macOS",
    }
    save("status.json", status)
    started = time.monotonic()
    try:
        async with asyncio.timeout(ARGS.timeout), session:
            (ROOT / "effective_system_prompt.txt").write_text(session.loop.build_system_prompt())
            result = await session.run((ROOT / "task.txt").read_text())
        (ROOT / "final_response.md").write_text(result)
        status["status"] = "completed"
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        status.update(finished_at=now(), elapsed_s=time.monotonic() - started)
        save("session.json", state.model_dump(mode="json"))
        save("status.json", status)


if __name__ == "__main__":
    if not (WORK / "input").exists():
        WORK.mkdir(exist_ok=True)
        shutil.copytree(BASELINE / "workspace/input", WORK / "input")
        for name in ("task.txt", "trace_helper.py"):
            shutil.copy2(BASELINE / name, ROOT / name)
        hashes = {}
        for path in sorted((WORK / "input").rglob("*")):
            if path.is_file():
                relative = path.relative_to(WORK / "input")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                assert (
                    digest
                    == hashlib.sha256(
                        (BASELINE / "workspace/input" / relative).read_bytes()
                    ).hexdigest()
                )
                hashes[str(relative)] = digest
        save("input_hashes.json", hashes)
        save(
            "experiment.json",
            {
                "baseline": str(BASELINE),
                "case": "case_03",
                "source": "current source snapshot" if ARGS.with_verifier else "baseline frozen snapshot",
                "with_verifier": ARGS.with_verifier,
                "outer_docker": False,
                "execution_backend": "macos",
                "disabled_tools": ["sandbox__run_shell"],
                "prompt": ("current SYSTEM_PROMPT + verifier instructions" if ARGS.with_verifier else
                           str(PROJECT / "configs/experiments/notebook_only_system_prompt.txt")),
                "input_file_count": len(hashes),
                "all_input_hashes_match": True,
                "limitations": ["单次联动测试，不作为与历史运行的受控比较。"] if ARGS.with_verifier
                else ["单次运行；提示词和工具可用性共同变化，不能单独归因。"],
            },
        )
    asyncio.run(preflight() if ARGS.preflight else run())
