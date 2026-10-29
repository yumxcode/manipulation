"""RECAP（π*0.6: RL with Experience & Corrections via Advantage-conditioned Policies）的公共部分。

论文: "π*0.6: a VLA That Learns From Experience", arXiv:2511.14759（2025）。

回报定义（与论文一致的“到达成功所需步数”形式）:
    每步奖励 -1；成功回合终止奖励 0；失败回合终止额外 -C_fail（默认 = 最长回合步数）。
    G_t = -(T-1-t) - [失败] * C_fail，再除以 R_max = L_max + C_fail 归一化到 [-1, 0]。
价值模型 V(o_t) 用分布式（分 bin）交叉熵回归 G_t。
N 步优势:
    A_t = (G_t - G_{t+N}) + V(o_{t+N}) - V(o_t)      (t+N < T)
    A_t = G_t - V(o_t)                                (否则，用到回合结束的蒙特卡洛回报)
二值化（论文 III/IV 节）：I_t = 1(A_t > ε_ℓ)，ε_ℓ 取该任务优势值的 30% 分位数，
即约 70% 的帧为 positive；人工接管帧（人工纠正）强制 positive。
"""

from __future__ import annotations

import numpy as np

from .config import max_episode_steps


def return_scale(cfg: dict) -> tuple[int, int]:
    l_max = max_episode_steps(cfg)
    c_fail = cfg["task"].get("fail_penalty_steps") or l_max
    return l_max, int(c_fail)


def normalized_returns(num_frames: int, success: bool, cfg: dict) -> np.ndarray:
    l_max, c_fail = return_scale(cfg)
    t = np.arange(num_frames)
    g = -(num_frames - 1 - t).astype(np.float64)
    if not success:
        g -= c_fail
    return np.clip(g / (l_max + c_fail), -1.0, 0.0).astype(np.float32)


def n_step_advantage(returns: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    T = len(returns)
    adv = np.empty(T, dtype=np.float32)
    for t in range(T):
        if t + n < T:
            adv[t] = (returns[t] - returns[t + n]) + values[t + n] - values[t]
        else:
            adv[t] = returns[t] - values[t]
    return adv


def positive_threshold(advantages: np.ndarray, positive_fraction: float) -> float:
    """ε_ℓ：取分位数使约 positive_fraction 比例的帧满足 A > ε 而被标为 positive。

    论文取优势值的 30% 分位数（positive_fraction=0.7 → quantile=0.3）。
    """
    a = np.asarray(advantages, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return 0.0
    return float(np.quantile(a, 1.0 - positive_fraction))


class Bins:
    def __init__(self, num_bins: int = 201, lo: float = -1.0, hi: float = 0.0):
        self.num_bins, self.lo, self.hi = num_bins, lo, hi
        self.centers = np.linspace(lo, hi, num_bins, dtype=np.float32)

    def two_hot(self, y) -> np.ndarray:
        """(B,) 标量 -> (B, num_bins)：落在相邻两个 bin 上的线性插值分布。"""
        y = np.clip(np.atleast_1d(np.asarray(y, dtype=np.float32)), self.lo, self.hi)
        pos = (y - self.lo) / (self.hi - self.lo) * (self.num_bins - 1)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, self.num_bins - 1)
        w1 = (pos - i0).astype(np.float32)
        out = np.zeros((len(y), self.num_bins), dtype=np.float32)
        r = np.arange(len(y))
        np.add.at(out, (r, i0), 1.0 - w1)
        np.add.at(out, (r, i1), w1)
        return out

    def expectation(self, probs: np.ndarray) -> np.ndarray:
        return probs @ self.centers
