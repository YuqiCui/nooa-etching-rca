"""使用真实模型验证 LiteLLM 统一工具调用及结果回传，不发送业务数据。"""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from factor_demo.autoresearch_agent_v2.agent import Session, SessionState
from factor_demo.autoresearch_agent_v2.agent.model import Model
from factor_demo.autoresearch_agent_v2.config import ModelConfig
from factor_demo.providers import load_provider


async def probe(provider_name: str, model_name: str | None, output: Path) -> bool:
    """请求随机标记并核对模型读取工具结果，保存不含凭据的响应。

    Args:
        provider_name: 项目中已有的 provider 配置名。
        model_name: 可选模型覆盖值，不改变服务地址和凭据。
        output: 尚不存在的本地结果目录。

    Returns:
        完整工具往返成功时返回 True。
    """
    provider = load_provider(provider_name)
    config = ModelConfig(
        model=f"{provider.api_style}/{model_name or provider.model}",
        api_key=provider.api_key,
        base_url=provider.base_url,
        max_tokens=2048,
        timeout_s=90,
        num_retries=0,
        tool_choice="auto",
    )
    output.mkdir(parents=True, exist_ok=False)
    marker = uuid4().hex
    received = []
    usage = []
    calls = []

    class ProbeModel(Model):
        """限制请求数量并保留 LiteLLM 返回的完整 message。"""

        async def completion(self, messages, *, tools=None):
            """最多请求四次，响应只包含本次人工构造的测试内容。"""
            if len(received) >= 4:
                raise ValueError("探针已达到四次请求上限")
            message = await super().completion(messages, tools=tools)
            received.append(message)
            return message

    def read_sample(path: str) -> str:
        """返回人工样本，不读取磁盘文件。"""
        if path != "sample.json":
            raise ValueError("只允许读取 sample.json 人工样本")
        calls.append(path)
        return json.dumps({"marker": marker, "value": 123})

    state = SessionState(
        session_id=uuid4().hex, agent_id="native-probe", task_id="native-probe", workdir=output
    )
    model = ProbeModel(config, usage_callback=usage.append)
    session = Session(model, state)
    session.loop.tools.register_callable(
        "read_sample", read_sample, description="读取 sample.json 人工样本，返回随机 marker。"
    )
    report = {"model": config.model, "litellm_version": version("litellm"), "passed": False}
    try:
        async with asyncio.timeout(180), session:
            result = await session.run(
                "这是原生工具调用测试。先调用 read_sample(path='sample.json')，"
                "读取返回结果后，单独调用 finish，其 content 必须只包含返回的 marker 原文。"
                "请使用 API 工具调用，不要在正文里编写工具调用 JSON。"
            )
        report["passed"] = calls == ["sample.json"] and result == marker
        report["result"] = result
    except Exception as error:  # noqa: BLE001 - 探针保存失败类型，避免泄露服务凭据。
        # 服务异常可能带 URL 或请求头，只保留异常类型。
        report["error_type"] = type(error).__name__
    finally:
        report.update(messages=received, history=state.messages, usage=usage, tool_executions=calls)
        (output / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({key: report[key] for key in ("model", "passed")}, ensure_ascii=False))
    print(output / "result.json")
    return report["passed"]


def main() -> None:
    """读取已有连接配置并运行单个真实模型探针。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="custom")
    parser.add_argument("--model")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs") / f"native-tools-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}",
    )
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(probe(args.provider, args.model, args.output)) else 1)


if __name__ == "__main__":
    main()
