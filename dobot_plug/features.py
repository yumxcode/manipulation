"""状态/动作向量的唯一定义处。数据转换（训练）与部署（推理）都调用这里，保证完全一致。

state  = [q(6, rad), gripper(1, 0开~1闭), (可选) F/T(6: N, Nm)]
action = [目标关节角 q*(6, rad), 夹爪指令(1, 0~1)]

openpi 训练配置里对前 6 维做 DeltaActions（相对当前关节角的增量），夹爪保持绝对值。
"""

from __future__ import annotations

import numpy as np

IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
ACTION_DIM = 7


def state_dim(include_ft: bool) -> int:
    return 13 if include_ft else 7


def build_state(q_deg, gripper_pos, ft, include_ft: bool) -> np.ndarray:
    q_deg = np.asarray(q_deg, dtype=np.float64)
    gripper_pos = np.asarray(gripper_pos, dtype=np.float64)
    parts = [np.deg2rad(q_deg), gripper_pos[..., None] if gripper_pos.ndim == q_deg.ndim - 1 else gripper_pos]
    if include_ft:
        parts.append(np.asarray(ft, dtype=np.float64))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def build_action(action_q_deg, action_gripper) -> np.ndarray:
    action_q_deg = np.asarray(action_q_deg, dtype=np.float64)
    g = np.asarray(action_gripper, dtype=np.float64)
    if g.ndim == action_q_deg.ndim - 1:
        g = g[..., None]
    return np.concatenate([np.deg2rad(action_q_deg), g], axis=-1).astype(np.float32)


def action_to_command(action: np.ndarray) -> tuple[np.ndarray, float]:
    """模型输出(绝对值) -> (目标关节角 deg, 夹爪 0~1)。"""
    action = np.asarray(action, dtype=np.float64)
    return np.rad2deg(action[:6]), float(np.clip(action[6], 0.0, 1.0))


def policy_observation(images_by_slot: dict[str, np.ndarray], state: np.ndarray, prompt: str) -> dict:
    """组装发给 openpi 策略服务器的 observation（键名与训练配置 DobotInputs 一致）。"""
    obs = {f"observation/{slot}": img for slot, img in images_by_slot.items()}
    obs["observation/state"] = state
    obs["prompt"] = prompt
    return obs


RECAP_SUFFIX_POS = "\nAdvantage: positive"
RECAP_SUFFIX_NEG = "\nAdvantage: negative"
