"""成功率统计：Wilson 95% 置信区间（评估报告用）。

小样本（n=10~50 回合）下正态近似误差大，机器人学习论文（LIBERO/lerobot-eval、
π0 系列评估）通行做法是报 Wilson 区间。
"""

from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """k 成功 / n 试验 的 Wilson 置信区间 (lo, hi)，返回比例。n=0 时返回 (0, 0)。"""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)
