"""提供可显式注册的统计业务工具。"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from statistics import fmean, median, pstdev
from typing import Any

from factor_demo.autoresearch_agent_v2.python_tools.base import PythonTool


def _validate_values(values: list[float], minimum_count: int) -> None:
    """检查样本数量和有限性，保持公开签名使用普通 Python 类型。"""
    if len(values) < minimum_count or not all(isfinite(value) for value in values):
        raise ValueError(f"至少需要 {minimum_count} 个有限数值，不能包含 NaN 或无穷值。")


class Statistics(PythonTool):
    """对数值观测进行描述性统计和相关分析。

    方法：
        summarize：计算数量、均值、中位数、极值和标准差。
        pearson_correlation：计算配对观测的 Pearson 相关系数及检验结果。
    """

    tool_methods = ("summarize", "pearson_correlation")
    """静态声明公开方法，注册时不构造实例。"""

    def methods(self) -> dict[str, Callable[..., Any]]:
        """声明允许模型使用的统计方法。

        Returns:
            方法名到绑定方法的显式映射。
        """
        return {
            "summarize": self.summarize,
            "pearson_correlation": self.pearson_correlation,
        }

    def summarize(self, values: list[float]) -> dict[str, int | float]:
        """计算有限数值观测的描述性统计。

        Args:
            values: 至少一个有限数值；拒绝空列表、缺失值、NaN 和无穷值。
                不自动丢弃观测，不修改输入顺序。

        Returns:
            count、mean、median、minimum、maximum 和 std。
            std 使用总体标准差，即 ddof=0；单个观测的标准差为零。
            统计量沿用输入数值的单位。

        Raises:
            ValueError: 输入为空或包含非有限数值。
        """
        _validate_values(values, 1)
        return {
            "count": len(values),
            "mean": fmean(values),
            "median": median(values),
            "minimum": min(values),
            "maximum": max(values),
            "std": pstdev(values),
        }

    def pearson_correlation(self, x: list[float], y: list[float]) -> dict[str, int | float]:
        """计算 Pearson 线性相关，不将相关关系解释为物理因果。

        Args:
            x: 至少三个有限数值，按位置与 y 配对。
            y: 与 x 等长的有限数值序列；不自动对齐、插值或过滤缺失值。

        Returns:
            count 是配对观测数量，coefficient 是 -1 到 1 之间的相关系数，
            p_value 是 scipy.stats.pearsonr 默认双侧检验的 p 值。
            解释时需要核对抽样和分布假设。

        Raises:
            ValueError: 少于三组观测、包含非有限数值、两组长度不同或任一组为常量。
        """
        from scipy.stats import pearsonr

        _validate_values(x, 3)
        _validate_values(y, 3)
        if len(x) != len(y):
            raise ValueError("配对观测必须等长。")
        if min(x) == max(x) or min(y) == max(y):
            raise ValueError("常量序列不能计算 Pearson 相关系数。")
        result = pearsonr(x, y)
        return {
            "count": len(x),
            "coefficient": float(result.statistic),
            "p_value": float(result.pvalue),
        }
