"""用核心 Agent 完成 A/B 数据的 Top 10 特征分析，并保存服务端 token 用量。

运行示例：
    uv run --env-file .env python scripts/experiments/run_sensor_similarity.py
    uv run --env-file .env python scripts/experiments/run_sensor_similarity.py --model glm-5.3-flash
    uv run python scripts/experiments/run_sensor_similarity.py --dry-run

默认读取项目模型配置和数据集原有 task.txt，不启动前端或 Web 服务。
不发送输出或 thinking token 上限，模型级重试为 10 次。
"""

import argparse
import asyncio
import json
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.autoresearch_agent_v2.feature_verifier import FeatureAnalyzeVerifier
from factor_demo.autoresearch_agent_v2.python_tools import PythonTools
from factor_demo.autoresearch_agent_v2.sandboxing import SandboxConfig
from factor_demo.autoresearch_agent_v2.tools import FileTools, SandboxTools, TaskTools
from factor_demo.dataset_adapter import discovery_copy
from factor_demo.io_utils import write_json
from factor_demo.providers import load_provider
from factor_demo.runtime import RuntimeSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _TracingModel(Model):
    """逐请求保存模型输入、原始回复和最新会话，便于中断后分析。"""

    def __init__(
        self,
        config: ModelConfig,
        state: SessionState,
        run: Path,
        usage_callback: Callable[[dict[str, Any] | None], None],
    ) -> None:
        super().__init__(config, usage_callback=usage_callback)
        self.state = state
        self.run = run
        self.request_index = 0

    def _record_event(self, event: dict[str, Any]) -> None:
        """追加不含连接凭据的模型事件。"""
        with (self.run / "model_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    async def completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """在真实模型调用前后记录完整消息和时间。

        Args:
            messages: 本次请求的完整有序历史。
            tools: 本次请求允许调用的工具定义；None 表示不提供原生工具。

        Returns:
            保留推理内容和工具调用的完整 assistant 消息。
        """
        self.request_index += 1
        metadata = {"request_index": self.request_index, "model": self.config.model}
        write_json(self.run / "session.json", self.state.model_dump(mode="json"))
        self._record_event(
            {
                **metadata,
                "event": "request",
                "at": datetime.now(UTC).isoformat(),
                "messages": messages,
                "tools": tools,
            }
        )
        try:
            message = await super().completion(messages, tools=tools)
        except BaseException as error:
            self._record_event(
                {
                    **metadata,
                    "event": "error",
                    "at": datetime.now(UTC).isoformat(),
                    "error_type": type(error).__name__,
                }
            )
            raise
        self._record_event(
            {
                **metadata,
                "event": "response",
                "at": datetime.now(UTC).isoformat(),
                "message": message,
            }
        )
        return message


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析独立运行入口参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/sensor-similarity-demo-20261030",
        help="包含 public/ 和原始 task.txt 的数据集目录",
    )
    parser.add_argument("--run-dir", type=Path, help="新的输出目录；已有目录会被拒绝")
    parser.add_argument("--provider", default="custom", help="沿用项目 provider 配置")
    parser.add_argument("--model", help="覆盖 provider 中的模型名称，不修改凭据或服务地址")
    parser.add_argument(
        "--resume-from", type=Path, help="从失败运行复制输入、产物和历史到新目录续跑"
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=PROJECT_ROOT / "configs/runtime.json",
        help="沿用推理开关、推理强度和单次请求超时；不设置输出或 thinking token 上限",
    )
    parser.add_argument("--timeout", type=float, default=3600, help="整次任务超时，单位秒")
    parser.add_argument("--tool-timeout", type=float, default=120, help="单次代码执行上限，单位秒")
    parser.add_argument("--dry-run", action="store_true", help="只显示计划，不创建目录或调用模型")
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.tool_timeout <= 0:
        parser.error("timeouts must be positive")
    args.dataset = args.dataset.expanduser().resolve(strict=True)
    args.run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir
        else PROJECT_ROOT
        / "runs"
        / (
            "sensor-similarity-v2-"
            + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )
    if args.run_dir.is_relative_to(args.dataset) or args.dataset.is_relative_to(args.run_dir):
        parser.error("dataset and run directories must be separate")
    if args.resume_from:
        args.resume_from = args.resume_from.expanduser().resolve(strict=True)
        if args.run_dir.is_relative_to(args.resume_from) or args.resume_from.is_relative_to(
            args.run_dir
        ):
            parser.error("source and recovery directories must be separate")
    return args


def _summarize_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总服务端已报告的用量，缓存和推理 token 不重复计入总数。"""
    summary: dict[str, Any] = {
        "received_responses": len(records),
        "responses_with_usage": 0,
        "responses_without_usage": 0,
        "responses_with_incomplete_usage": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "scope": "provider-reported responses; failed or cancelled requests may have unreported usage",
    }
    for record in records:
        usage = record["usage"]
        if usage is None:
            summary["responses_without_usage"] += 1
            continue
        summary["responses_with_usage"] += 1
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        if input_tokens is None or output_tokens is None or total_tokens is None:
            summary["responses_with_incomplete_usage"] += 1
        summary["input_tokens"] += input_tokens or 0
        summary["output_tokens"] += output_tokens or 0
        summary["total_tokens"] += total_tokens or 0
        summary["cached_input_tokens"] += (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        ) or 0
        summary["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens", 0
        ) or 0
    return summary


def _build_model_config(args: argparse.Namespace) -> ModelConfig:
    """从项目配置和命令行覆盖项构造正式运行与探针共用的模型配置。"""
    settings = RuntimeSettings.load(args.settings)
    provider = load_provider(args.provider)
    model_name = args.model or provider.model
    thinking: dict[str, Any] = {"type": "enabled" if settings.thinking else "disabled"}
    if provider.api_style == "openai" and model_name.lower().startswith("glm-"):
        thinking["clear_thinking"] = False
    return ModelConfig(
        model=f"{provider.api_style}/{model_name}",
        api_key=provider.api_key,
        base_url=provider.base_url,
        max_tokens=None,
        timeout_s=settings.request_timeout,
        num_retries=10,
        thinking=thinking,
        reasoning_effort=settings.reasoning_effort if provider.api_style == "openai" else None,
        # DeepSeek 的 Chat Completions thinking 模式不接受 required。
        tool_choice=(
            "auto"
            if settings.thinking and model_name.lower().startswith("deepseek")
            else "required"
        ),
    )


def _restore_failed_run(
    source: Path, run: Path, state: SessionState, config: ModelConfig, dataset: Path
) -> str:
    """校验失败源并复制产物，保留全部历史，为未完成调用补充明确的中断结果。"""
    source_status = json.loads((source / "status.json").read_text(encoding="utf-8"))
    if source_status["status"] not in {"failed", "timeout", "cancelled"}:
        raise ValueError("Only failed, timed out or cancelled runs can be recovered")
    source_config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    # 旧版脚本固定使用 required，历史记录中没有这个显式字段。
    source_config.setdefault("tool_choice", "required")
    public_config = config.model_dump(mode="json", exclude={"api_key", "base_url"})
    if any(source_config[key] != value for key, value in public_config.items()):
        raise ValueError("Recovery model configuration differs from the source run")
    if source_config["dataset"] != str(dataset):
        raise ValueError("Recovery dataset differs from the source run")
    if any(
        path.is_symlink() for name in ("input", "workspace") for path in (source / name).rglob("*")
    ):
        raise ValueError("Recovery inputs and workspace cannot contain symbolic links")
    source_state = SessionState.model_validate_json((source / "session.json").read_text())
    shutil.copytree(source / "input", run / "input")
    shutil.copytree(source / "workspace", state.workdir, dirs_exist_ok=True)
    state.messages = source_state.messages
    state.compaction = source_state.compaction
    last_assistant_index = next(
        (
            index
            for index in range(len(state.messages) - 1, -1, -1)
            if state.messages[index]["role"] == "assistant"
        ),
        None,
    )
    interrupted_messages = (
        state.messages[last_assistant_index:] if last_assistant_index is not None else []
    )
    completed_ids = {
        message["tool_call_id"] for message in interrupted_messages if message["role"] == "tool"
    }
    pending_calls = [
        call
        for message in interrupted_messages
        for call in message.get("tool_calls") or []
        if call["id"] not in completed_ids
    ]
    for call in pending_calls:
        state.messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": (
                    f"上次运行在 {source_status.get('error_type', source_status['status'])} "
                    "后中断，此调用未确认成功，可能已有部分副作用；请先检查产物再修正参数继续。"
                ),
            }
        )
    write_json(
        run / "recovery.json",
        {
            "source_run": str(source),
            "source_status": source_status,
            "source_token_summary": json.loads((source / "token_summary.json").read_text()),
            "interrupted_tool_call_ids": [call["id"] for call in pending_calls],
            "python_kernel_restored": False,
        },
    )
    return (
        "从已保存的失败会话继续完成原任务。旧运行保持原样，输入和产物已复制。"
        f"当前可写工作目录为 {state.workdir}，只读输入目录为 {run / 'input'}。"
        "历史中的旧路径应替换成当前对应路径。Python 内存未恢复；"
        "Notebook 文件保留源码与输出，新建 Notebook 重新加载所需变量。"
        "若 artifacts/ 中已有分析脚本，请显式传入当前输入和输出路径复跑。"
        "请通过新 Notebook 中的 runpy 和 sys.argv 验证脚本，不用 Shell 绕过 Notebook。"
        "Shell 超时不得超过宿主配置；参数错误应修正，不能扩大权限或预算。"
        "继续检查已有产物，补全报告和代码验证，并通过 finish 提交最终结果。"
        "优先复用已保存的脚本、统计表和图表，避免重复探索已完成的内容。"
    )


async def _run_agent(args: argparse.Namespace) -> None:
    """创建隔离数据快照、运行一次会话，并在退出时保存状态和用量。"""
    task = ((args.resume_from or args.dataset) / "task.txt").read_text(encoding="utf-8")
    config = _build_model_config(args)
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=False)
    workdir = run / "workspace"
    (workdir / "artifacts").mkdir(parents=True)
    records: list[dict[str, Any]] = []
    status = {
        "status": "preparing",
        "started_at": datetime.now(UTC).isoformat(),
        "model": config.model,
        "tool_call_format": "native",
    }
    state = SessionState(
        session_id=uuid.uuid4().hex,
        agent_id="sensor-similarity",
        task_id=run.name,
        workdir=workdir,
    )

    def record_usage(usage: dict[str, Any] | None) -> None:
        """逐响应落盘原始用量，并原子更新累计统计。"""
        record = {
            "response_index": len(records) + 1,
            "received_at": datetime.now(UTC).isoformat(),
            "usage": usage,
        }
        with (run / "token_usage.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        records.append(record)
        summary = _summarize_usage(records)
        write_json(run / "token_summary.json", summary)
        print(
            f"[usage] responses={summary['received_responses']} "
            f"input={summary['input_tokens']} output={summary['output_tokens']} "
            f"total={summary['total_tokens']} missing={summary['responses_without_usage']}",
            flush=True,
        )

    try:
        write_json(run / "status.json", status)
        write_json(
            run / "run_config.json",
            {
                **config.model_dump(mode="json", exclude={"api_key", "base_url"}),
                "provider": args.provider,
                "dataset": str(args.dataset),
                "settings": str(args.settings),
                "task_timeout_s": args.timeout,
                "tool_timeout_s": args.tool_timeout,
                "tool_call_format": "native",
                "session_concurrency": 1,
                "api_concurrency": 1,
                "tool_concurrency": 1,
            },
        )
        if args.resume_from:
            recovery_prompt = _restore_failed_run(
                args.resume_from, run, state, config, args.dataset
            )
            (run / "recovery_prompt.txt").write_text(recovery_prompt, encoding="utf-8")
        else:
            # 与旧 Agent 使用相同的导入器，不复制整个 public，也不授权读取其父目录。
            discovery_copy(args.dataset / "public", run / "input")
        write_json(
            run / "input_manifest.json",
            sorted(
                str(path.relative_to(run / "input"))
                for path in (run / "input").rglob("*")
                if path.is_file()
            ),
        )
        (run / "task.txt").write_text(task, encoding="utf-8")
        model = _TracingModel(config, state, run, record_usage)
        session = Session(model, state)
        session.register_dataset(
            "DATASET",
            str(run / "input"),
            "A、B 两组传感器统计特征及原始曲线；",
        )
        SandboxTools.register(
            session,
            config=SandboxConfig(
                workdir=workdir,
                read_paths=(run / "input",),
                timeout_s=args.tool_timeout,
            ),
        )
        TaskTools.register(session, storage_root=run / "task_storage")
        session.register_stateful_tool(FileTools)
        session.register_stateful_tool(PythonTools)
        session.register_stateful_tool(FeatureAnalyzeVerifier)
        status["status"] = "running"
        write_json(run / "status.json", status)
        print(f"Run directory: {run}", flush=True)
        async with asyncio.timeout(args.timeout):
            async with session:
                result = await session.run(recovery_prompt if args.resume_from else task)
        (run / "final_response.md").write_text(result, encoding="utf-8")
        status["status"] = "completed"
    except TimeoutError:
        status["status"] = "timeout"
        raise
    except asyncio.CancelledError:
        status["status"] = "cancelled"
        raise
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        status["finished_at"] = datetime.now(UTC).isoformat()
        write_json(run / "session.json", state.model_dump(mode="json"))
        write_json(run / "token_summary.json", _summarize_usage(records))
        write_json(run / "status.json", status)


def main(argv: list[str] | None = None) -> None:
    """读取参数后运行任务，dry-run 只输出计划。

    Args:
        argv: 命令行参数；None 使用当前进程参数。
    """
    args = _parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "run_dir": str(args.run_dir),
                    "task": ((args.resume_from or args.dataset) / "task.txt").read_text(
                        encoding="utf-8"
                    ),
                    "input_policy": "discovery_copy(public, input), same as the previous agent",
                    "excluded_files": ["features.parquet", "group_A.parquet", "group_B.parquet"],
                    "provider": args.provider,
                    "model_override": args.model,
                    "resume_from": str(args.resume_from) if args.resume_from else None,
                    "settings": str(args.settings),
                    "tool_call_format": "native",
                    "timeout_s": args.timeout,
                    "tool_timeout_s": args.tool_timeout,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    asyncio.run(_run_agent(args))


if __name__ == "__main__":
    main()
