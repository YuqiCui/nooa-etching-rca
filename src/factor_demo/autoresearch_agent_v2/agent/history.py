"""保存完整历史，并为模型构建可压缩的续接上下文。"""

import json
from typing import Any

from ..config import HistoryConfig
from .model import Model
from .typing import HistoryCompaction, SessionState

# 英文指令与引导语沿用 DeepSeek Harness 39163dd，版权及 MIT 许可见 COMPACTION_LICENSE。
COMPACT_PROMPT = """You are now acting as a compaction engine for this AI coding assistant. Condense the conversation ABOVE into a structured checkpoint that lets another model resume the work with no loss of essential context.

Output EXACTLY the Markdown structure below: keep every section, in order. Use terse bullets, not prose paragraphs. Write "(none)" for an empty section — never drop a section.

## Primary Request and Intent
- [the user's original and evolving goals; quote verbatim where the exact wording matters]

## Key Technical Concepts
- [technologies, frameworks, patterns, and conventions in play]

## Files and Code
- [exact path: why it matters, key changes or snippets]

## Errors and Fixes
- [error: how it was resolved, plus any related user feedback]

## Pending Jobs
- [explicitly requested work not yet completed]

## Current Work
- [precisely what was in progress at this checkpoint]

## Next Step
- [the single next action, directly in line with the most recent request, or "(none)"]

## Critical Context
- [decisions and their rationale, constraints, user preferences, open questions, data needed to continue]

Rules:
- Write concise English engineering prose. Preserve exact file paths, commands, error strings, identifiers, numeric values, function signatures, and syntax fragments.
- Capture user feedback and explicit instructions faithfully, especially corrections.
- Do NOT mention this summarization request or that the context was compacted.
- Output only the checkpoint text: do not call any tool or take any other action.
- If the conversation already contains a <compacted-summary> block, it is a PRIOR checkpoint. Do not copy it forward verbatim: preserve still-true facts, drop stale ones, and merge newer information into a single consolidated summary under the same structure."""

SUMMARY_PREFIX = (
    "This is an automatically generated checkpoint condensing an earlier span of the "
    "conversation to free up context. Treat the captured context as established background "
    "and build on it without restating it. Continue the task directly from the messages "
    "that follow, without acknowledging this checkpoint.\n\n<compacted-summary>\n"
)
SUMMARY_SUFFIX = "\n</compacted-summary>"


def get_context_messages(state: SessionState) -> list[dict[str, Any]]:
    """构建模型使用的历史副本，完整原始历史仍由 state 保存。

    Args:
        state: 含原始消息和可选压缩检查点的会话状态。

    Returns:
        没有检查点时返回完整历史；否则返回摘要及未被摘要覆盖的消息。
        前缀变化影响的旧 thinking 只从请求副本移除，新生成的 thinking 原样保留。
    """
    messages = state.get_messages()
    checkpoint = state.compaction
    if checkpoint is None:
        return messages
    retained = messages[checkpoint.end_index :]
    for message in retained[: checkpoint.source_message_count - checkpoint.end_index]:
        if message["role"] != "assistant":
            continue
        for key in ("reasoning_content", "reasoning", "thinking_blocks"):
            message.pop(key, None)
        if isinstance(message.get("content"), list):
            message["content"] = [
                block
                for block in message["content"]
                if block.get("type") not in ("thinking", "redacted_thinking", "reasoning")
            ] or None
    return [
        {"role": "user", "content": SUMMARY_PREFIX + checkpoint.summary + SUMMARY_SUFFIX},
        *retained,
    ]


def _find_compaction_end(messages: list[dict[str, Any]], keep_recent_turns: int) -> int:
    """检查工具批次完整性，并按模型回复及其整批结果确定轮次边界。"""
    pending: set[str] = set()
    turn_starts = []
    next_turn_start: int | None = None
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("工具结果缺少对应调用，不能压缩历史")
            pending.remove(call_id)
            continue
        if pending:
            raise ValueError("工具调用尚未全部返回，不能压缩历史")
        if role == "user":
            if next_turn_start is None:
                next_turn_start = index
        elif role == "assistant":
            turn_starts.append(next_turn_start if next_turn_start is not None else index)
            next_turn_start = None
        calls = message.get("tool_calls") or []
        pending = {call["id"] for call in calls}
        if len(pending) != len(calls):
            raise ValueError("工具调用 ID 重复，不能压缩历史")
    if pending:
        raise ValueError("工具调用尚未全部返回，不能压缩历史")
    if next_turn_start is not None:
        # 尚未收到模型回复的最新输入也占一轮，避免压缩时丢掉当前请求。
        turn_starts.append(next_turn_start)
    if keep_recent_turns == 0:
        return len(messages)
    return turn_starts[-keep_recent_turns] if len(turn_starts) > keep_recent_turns else 0


async def compact_history(
    state: SessionState,
    model: Model,
    *,
    system_prompt: str,
    history_config: HistoryConfig | None = None,
    keep_recent_turns: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> bool:
    """总结较早的完整轮次，成功后原子切换模型上下文。

    Args:
        state: 原始消息只追加、不被压缩删除的会话状态。
        model: 使用现有 completion 接口生成摘要；返回的工具调用不会执行。
        system_prompt: 沿用当前 Agent 的系统说明，保留压缩请求的历史前缀。
        history_config: 压缩策略；未提供时使用 HistoryConfig 的默认值。
        keep_recent_turns: 本次覆盖配置中的保留轮数；None 使用配置，0 总结全部历史。
        tools: 当前工具定义；只用于保持请求前缀，不进入工具执行循环。

    Returns:
        成功保存新摘要时返回 True；没有可新增压缩的历史时返回 False。

    Raises:
        ValueError: 保留轮数非法、工具协议不完整、摘要为空或包含调用、压缩未缩短上下文。
        RuntimeError: 请求期间原始历史或检查点发生变化。
        Exception: 模型错误或取消向上传播，原检查点保持不变。
    """
    config = history_config if history_config is not None else HistoryConfig()
    if keep_recent_turns is None:
        keep_recent_turns = config.keep_recent_turns
    if type(keep_recent_turns) is not int or keep_recent_turns < 0:
        raise ValueError("保留轮数必须是非负整数")
    snapshot = state.get_messages()
    previous = state.compaction
    end_index = _find_compaction_end(snapshot, keep_recent_turns)
    previous_end = previous.end_index if previous is not None else 0
    if end_index <= previous_end:
        return False
    context = get_context_messages(state)
    prefix_count = end_index - previous_end + (previous is not None)
    message = await model.completion(
        [
            {"role": "system", "content": system_prompt},
            *context[:prefix_count],
            {"role": "user", "content": COMPACT_PROMPT},
        ],
        tools=tools,
        tool_choice="auto",
    )
    summary = message.get("content")
    if message.get("tool_calls") or not isinstance(summary, str) or not summary.strip():
        raise ValueError("压缩必须返回非空摘要文本，不能包含工具调用")
    checkpoint = HistoryCompaction(
        summary=summary.strip(), end_index=end_index, source_message_count=len(snapshot)
    )
    candidate = state.model_copy(update={"compaction": checkpoint})
    if len(json.dumps(get_context_messages(candidate), ensure_ascii=False)) >= len(
        json.dumps(context, ensure_ascii=False)
    ):
        raise ValueError("压缩结果未缩短上下文，原历史保持不变")
    if state.messages != snapshot or state.compaction != previous:
        raise RuntimeError("压缩期间会话历史已改变，请重新压缩")
    state.compaction = checkpoint
    return True
