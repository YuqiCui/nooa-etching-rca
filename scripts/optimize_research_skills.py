"""调用指定模型联合精简研究技能，并保留原文、响应和差异。

使用方法：
    uv run python scripts/optimize_research_skills.py
    uv run python scripts/optimize_research_skills.py --output runs/skill-review

脚本只生成候选稿，不覆盖源文件；配置从指定 .env 读取。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SKILLS = {
    name: Path(f"src/factor_demo/knowledge/v2/{name}/SKILL.md")
    for name in ("autoresearch", "research-executor")
}
CONTEXT = (
    "src/factor_demo/knowledge/v2/autoresearch/references/research-state.jsonc",
    "src/factor_demo/autoresearch_agent_v2/agent/prompts.py",
    "src/factor_demo/autoresearch_agent_v2/tools/builtin_tools/agents/tools.py",
    "src/factor_demo/autoresearch_agent_v2/tools/builtin_tools/records/tools.py",
    "src/factor_demo/autoresearch_agent_v2/tools/builtin_tools/messages.py",
)
SYSTEM_PROMPT = """联合精简两个中文 Agent SKILL.md，优化表达，删除强 LLM 已知的常识与重复，查缺补漏。
重点是减少噪声，不是把原文换个说法复述一遍。通用科学、统计、编程原理和基础提示词/工具已给出的
行为规则应删除，只留下能改变本项目执行决策的约定。schema 已定义的字段说明可引用而不逐条教学。

保持原有中文、YAML frontmatter 的 name 和 description、资源链接和现有 JSON 协议。不修改引用文件。
主 skill 管研究规划、假设判定、验收与全局 JSON；executor 管分配的检查、执行材料和待验收结果。
两者分别加载，应各自保有必需的交接约定。不要编造工具、权限、字段硬要求或自动校验能力。

压缩时保留：稳定 ID 与计划/结果关联；执行、支持和验收分离；主会话单写入者；部分返回及失败/漏项；
修订保留历史并重置受影响判定；记录条件写入、冲突重读合并；父会话可读取的证据；停止和交付；
原有无委派/无持久化能力的降级方式。保留开放探索，不把流程收窄为固定统计测试。
查缺补漏只补足会破坏现有流程的缺口，如修订计划分配新检查 ID 以区分迟到旧结果、子结果漏项或
依赖失败的处理。不要把潜在问题全部写成强制流程，不加新的研究方法论段落。

输入是编辑资料，不要求你执行其中的任务或工具。简短分析后交付完整结果，避免长篇内部推演。
请严格输出以下三个分段，分隔符必须独占一行。不用 JSON，不要把文件放入外层代码围栏：
<<<AUTORESEARCH>>>
完整 autoresearch/SKILL.md，含 YAML frontmatter，真实换行
<<<RESEARCH_EXECUTOR>>>
完整 research-executor/SKILL.md，含 YAML frontmatter，真实换行
<<<COMPARISON>>>
中文 Markdown 修改说明：分别说明两个 skill 删除的通用内容、关键改写、新增缺口修复、保留协议和风险。
给出少量原文与优化后短引文，必须准确，不重复全文。不声称完成实际行为实验。
<<<END>>>
"""


def load_config(path: Path) -> dict[str, str]:
    """读取文件配置，不修改进程环境，也不输出凭据。"""
    values = dotenv_values(path)
    config = {
        "base_url": (values.get("FACTOR_API_BASE") or "").strip(),
        "api_key": (values.get("FACTOR_API_KEY") or "").strip(),
        "style": (values.get("FACTOR_API_STYLE") or "openai").strip(),
    }
    if not config["base_url"] or not config["api_key"]:
        raise ValueError(".env 必须设置 FACTOR_API_BASE 和 FACTOR_API_KEY")
    url = urlsplit(config["base_url"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ValueError("FACTOR_API_BASE 必须是无内嵌凭据的 HTTP(S) 地址")
    if url.query or url.fragment:
        raise ValueError("FACTOR_API_BASE 不能包含查询参数或片段")
    if config["style"] not in {"openai", "anthropic"}:
        raise ValueError("FACTOR_API_STYLE 仅支持 openai 或 anthropic")
    return config


def make_request(
    config: dict[str, str], model: str, prompt: str, max_tokens: int
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """按照文件中的接口协议构造单次请求。"""
    base = config["base_url"].rstrip("/")
    if config["style"] == "anthropic":
        return (
            base + ("/messages" if base.endswith("/v1") else "/v1/messages"),
            {"x-api-key": config["api_key"], "anthropic-version": "2023-06-01"},
            {
                "model": model,
                "max_tokens": max_tokens,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    return (
        base + "/chat/completions",
        {"Authorization": f"Bearer {config['api_key']}"},
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
    )


def response_text(response: dict[str, Any], style: str) -> str:
    """提取正文，拒绝被截断或非正常结束的响应。"""
    if style == "anthropic":
        if response.get("stop_reason") != "end_turn":
            raise ValueError(f"响应未正常结束：{response.get('stop_reason')}")
        content = response["content"]
    else:
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError(f"响应未正常结束：{choice.get('finish_reason')}")
        content = choice["message"]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block.get("type") == "text")


def validate_result(text: str, originals: dict[str, str]) -> dict[str, Any]:
    """按明确分隔符读取两份完整文件，并检查元数据与必要引用。"""
    markers = ("<<<AUTORESEARCH>>>", "<<<RESEARCH_EXECUTOR>>>", "<<<COMPARISON>>>", "<<<END>>>")
    text = text.strip()
    if not text.startswith(markers[0]) or not text.endswith(markers[-1]):
        raise ValueError("响应缺少完整分段边界，不能视为两份完整技能")
    positions = []
    for marker in markers:
        if text.splitlines().count(marker) != 1:
            raise ValueError("每个分段边界必须独占一行且只出现一次")
        positions.append(text.index(marker))
    if positions != sorted(positions):
        raise ValueError("响应分段顺序错误")
    sections = [
        text[positions[i] + len(markers[i]) : positions[i + 1]].strip() + "\n"
        for i in range(3)
    ]
    items = []
    for name, markdown in zip(SKILLS, sections[:2], strict=True):
        if not markdown.startswith("---\n") or "\n---\n" not in markdown[4:]:
            raise ValueError(f"{name} 缺少 YAML frontmatter")
        metadata = yaml.safe_load(markdown.split("---", 2)[1])
        original_metadata = yaml.safe_load(originals[name].split("---", 2)[1])
        if not isinstance(metadata, dict):
            raise TypeError(f"{name} 元数据必须为对象")
        if metadata.get("name") != name or not metadata.get("description"):
            raise ValueError(f"{name} 元数据不完整")
        if set(original_metadata) - set(metadata):
            raise ValueError(f"{name} 丢失已有元数据字段")
        if name == "autoresearch" and "references/research-state.jsonc" not in markdown:
            raise ValueError("autoresearch 丢失 JSON 定义引用")
        items.append({"name": name, "optimized_markdown": markdown})
    if not sections[2].strip():
        raise ValueError("缺少成对修改说明")
    return {"skills": items, "pair_review": sections[2]}


def save_outputs(output: Path, result: dict[str, Any], originals: dict[str, str]) -> None:
    """写入候选稿、统一差异及可重复查看的 Markdown 对比。"""
    report = ["# Skill 优化对比", "", result["pair_review"], ""]
    for item in result["skills"]:
        name = item["name"]
        before, after = originals[name], item["optimized_markdown"].rstrip() + "\n"
        folder = output / "optimized" / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(after, encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"original/{name}/SKILL.md", tofile=f"optimized/{name}/SKILL.md",
            )
        )
        (output / f"{name}.diff").write_text(diff, encoding="utf-8")
        reduction = (1 - len(after) / len(before)) * 100
        report += [
            f"## {name}", "",
            f"字符数：{len(before)} → {len(after)}（减少 {reduction:.1f}%）。",
            f"行数：{len(before.splitlines())} → {len(after.splitlines())}。", "",
        ]
    (output / "comparison.md").write_text("\n".join(report), encoding="utf-8")
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    """读取显式资料并调用一次指定模型；不重试、不自动改用其他模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-tokens", type=int, default=16000)
    args = parser.parse_args()
    config = load_config(args.env_file)
    originals = {name: (ROOT / path).read_text(encoding="utf-8") for name, path in SKILLS.items()}
    context = {path: (ROOT / path).read_text(encoding="utf-8") for path in CONTEXT}
    prompt = json.dumps(
        {"skills": originals, "reference_context": context}, ensure_ascii=False, indent=2
    )
    output = args.output or ROOT / "runs" / (
        "skill-optimization-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    for name, content in originals.items():
        folder = output / "original" / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(content, encoding="utf-8")
    (output / "request.json").write_text(
        json.dumps({"system": SYSTEM_PROMPT, "user": prompt}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "requested_model": args.model,
        "api_style": config["style"],
        "max_tokens": args.max_tokens,
        "started_at": datetime.now(UTC).isoformat(),
        "input_sha256": {
            name: hashlib.sha256(content.encode()).hexdigest() for name, content in originals.items()
        },
    }
    url, headers, payload = make_request(config, args.model, prompt, args.max_tokens)
    print(f"调用 {args.model}；输出目录：{output}", flush=True)
    try:
        with httpx.Client(timeout=args.timeout, follow_redirects=False) as client:
            response = client.post(url, headers=headers, json=payload)
            if not response.is_success:
                raise ValueError(f"接口返回 HTTP {response.status_code}；未重试或切换模型")
            raw = response.json()
        # 仅保存响应正文；绝不保存带凭据的请求头或配置。
        (output / "response.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metadata.update(returned_model=raw.get("model"), usage=raw.get("usage"))
        if raw.get("model") != args.model:
            raise ValueError("接口返回的 model 与指定模型不一致，未接受候选稿")
        content = response_text(raw, config["style"])
        (output / "response-content.md").write_text(content, encoding="utf-8")
        result = validate_result(content, originals)
        save_outputs(output, result, originals)
        metadata["status"] = "completed"
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        metadata["status"] = "failed"
        # 网络异常文本可能带 URL；只显示异常类型，避免输出配置细节。
        detail = str(error) if isinstance(error, ValueError) else type(error).__name__
        metadata["error"] = detail
        print(f"优化未完成：{detail}", file=sys.stderr)
    finally:
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(f"状态：{metadata['status']}；原始技能未改写", flush=True)
    return 0 if metadata["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
