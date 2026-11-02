"""模型请求、工具执行和消息回传组成的基础 Agent 循环。"""

import json
import logging

from ..config import HistoryConfig
from ..tools import create_tool_factory
from ..tools.result_guard import protect_tool_result
from .history import compact_history, get_context_messages
from .model import Model
from .prompts import SYSTEM_PROMPT, dataset_prompt
from .typing import SessionState

logger = logging.getLogger(__name__)


class AgentLoop:
    """执行一轮用户请求，直到模型调用 finish。"""

    def __init__(
        self,
        model: Model,
        session: SessionState,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        history_config: HistoryConfig | None = None,
    ) -> None:
        """绑定模型和会话，注册 finish 工具。

        Args:
            model: 用于生成下一条消息的模型。
            session: 保存完整对话历史的会话。
            system_prompt: 每次模型请求使用的系统提示词。
            history_config: 上下文预算、自动压缩百分比及默认保留轮数。

        """
        self.model = model
        self.session = session
        self.system_prompt = system_prompt
        self.tools = create_tool_factory()
        self.history_config = history_config if history_config is not None else HistoryConfig()

    async def before_run_hook(self, content: str) -> str:
        """在写入用户输入前执行钩子，默认原样返回。

        Args:
            content: 本次用户输入。

        Returns:
            交给本次循环的用户输入。
        """
        return content

    async def after_run_hook(self) -> None:
        """在本次运行结束时执行钩子，默认不做处理。"""
        pass

    def build_system_prompt(self) -> str:
        """拼接当前会话实际使用的完整系统提示词。

        Returns:
            包含数据集及工具说明的系统提示词。
        """
        return "\n\n".join(
            part
            for part in (
                self.system_prompt,
                dataset_prompt(self.session),
                self.tools.instructions(),
            )
            if part
        )

    async def run(self, content: str) -> str:
        """追加本轮输入并执行循环，返回 finish 内容，不清空会话历史。

        工具参数解析和执行中的普通异常均作为工具结果返回，供模型决定后续操作。
        外部取消、进程退出等控制信号不转换为工具结果。

        Args:
            content: 本次用户输入。

        Returns:
            finish 工具向用户输出的最终内容。

        Raises:
            ValueError: 当前请求达到上下文 token 预算。
            Exception: 模型请求、运行钩子或工具结果保存失败，原样向上传播。

        """
        try:
            content = await self.before_run_hook(content)
            self.session.messages.append({"role": "user", "content": content})
            while True:
                schemas = self.tools.schemas()
                system_prompt = self.build_system_prompt()
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    *get_context_messages(self.session),
                ]
                input_tokens = self.model.count_tokens(messages, tools=schemas)
                config = self.history_config
                if (
                    input_tokens * 100
                    >= config.context_window_tokens * config.compact_trigger_percent
                ):
                    changed = await compact_history(
                        self.session,
                        self.model,
                        system_prompt=system_prompt,
                        history_config=config,
                    )
                    if changed:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            *get_context_messages(self.session),
                        ]
                        compacted_tokens = self.model.count_tokens(messages, tools=schemas)
                        logger.info(
                            "自动压缩上下文：输入 token %s → %s", input_tokens, compacted_tokens
                        )
                        input_tokens = compacted_tokens
                if input_tokens >= config.context_window_tokens:
                    raise ValueError(
                        "当前请求已达到上下文 token 预算；请减少保留轮数、缩短输入或工具说明，"
                        "或按实际模型能力调整 context_window_tokens"
                    )
                message = await self.model.completion(
                    messages,
                    tools=schemas,
                )
                self.session.messages.append(message)
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    self.session.messages.append(
                        {
                            "role": "user",
                            "content": "请调用工具继续；如需结束本轮，请调用 finish(content) 提交最终回复。",
                        }
                    )
                    continue
                for index, tool_call in enumerate(tool_calls):
                    function = tool_call["function"]
                    failed = False
                    try:
                        result = await self.tools.call(
                            function["name"],
                            json.loads(function["arguments"]),
                            call_id=tool_call["id"],
                        )
                    except Exception as error:  # noqa: BLE001 - 工具错误统一回传，控制信号继续传播。
                        result = f"{type(error).__name__}: {error}"
                        failed = True
                    self.session.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": protect_tool_result(
                                result,
                                self.session.workdir / "tool_results",
                                workspace_root=self.session.workdir,
                            ),
                        }
                    )
                    if function["name"] == "finish" and not failed:
                        for skipped_call in tool_calls[index + 1 :]:
                            self.session.messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": skipped_call["id"],
                                    "content": "未执行：finish 已结束本轮。",
                                }
                            )
                        return result
        finally:
            await self.after_run_hook()
