"""统计 Router 每轮任务的耗时和累计 token 用量。"""

import math
import time
from datetime import datetime, timedelta


class TimeBudget:
    """记录本轮预算，并将同次检查达到的相同阈值合并提醒。"""

    thresholds = (50, 60, 70, 80, 90, 95)

    def __init__(self, seconds: float, tokens: int | None = None) -> None:
        """记录时间与 token 上限，以及带时区的开始时间。"""
        self.seconds = seconds
        self.tokens = tokens
        self.started = time.monotonic()
        self.started_at = datetime.now().astimezone()
        self.time_notified: set[int] = set()
        self.token_notified: set[int] = set()

    def _token_limit(self) -> str:
        """返回提示词中显示的 token 上限。"""
        return "无限" if self.tokens is None else str(self.tokens)

    def initial_prompt(self) -> str:
        """生成本轮固定的初始预算说明，避免逐次改变系统前缀。"""
        deadline = self.started_at + timedelta(seconds=self.seconds)
        return (
            f"时间预算：当前时间 {self.started_at.isoformat(timespec='seconds')}；"
            f"本轮 Router 时间预算 {self.seconds / 60:g} 分钟；"
            f"预算截止时间 {deadline.isoformat(timespec='seconds')}。"
            f"本轮 Router token 预算：{self._token_limit()}。"
            "请据此安排研究与收尾；预算提醒不代表强制终止。"
        )

    def reminders(self, used_tokens: int = 0) -> list[dict[str, str]]:
        """返回所有新跨越阈值的系统消息，工具批次完成后才投递。"""
        elapsed = max(0.0, time.monotonic() - self.started)
        time_reached = {
            percent for percent in self.thresholds
            if percent not in self.time_notified and elapsed >= self.seconds * percent / 100
        }
        token_reached = {
            percent for percent in self.thresholds
            if self.tokens is not None
            and percent not in self.token_notified
            and used_tokens * 100 >= self.tokens * percent
        }
        reached = sorted(time_reached | token_reached)
        if not reached:
            return []
        current = datetime.now().astimezone().isoformat(timespec="seconds")
        remaining = math.ceil(max(0.0, self.seconds - elapsed))
        token_remaining = "无限" if self.tokens is None else str(max(0, self.tokens - used_tokens))
        self.time_notified.update(time_reached)
        self.token_notified.update(token_reached)
        status = (
            f"当前时间 {current}。"
            f"时间预算 {self.seconds / 60:g} 分钟，已用 {elapsed / 60:.1f} 分钟，"
            f"剩余 {remaining // 60} 分 {remaining % 60} 秒。"
            f"token 预算 {self._token_limit()}，服务端报告已用 {used_tokens}，"
            f"按已报告用量计算的剩余 token {token_remaining}。"
            "请根据剩余预算安排后续工作与收尾。"
        )
        messages = []
        for percent in reached:
            trigger = "时间" if percent in time_reached else "token"
            if percent in time_reached and percent in token_reached:
                trigger = "时间、token"
            messages.append({
                "role": "system",
                "content": f"预算提醒（{trigger}达到 {percent}% 阈值）。{status}",
            })
        return messages
