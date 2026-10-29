"""3D 鼠标（SpaceMouse）遥操作：速度指令 -> 积分目标 TCP 位姿 -> 控制器逆解 -> 目标关节角。"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

log = logging.getLogger(__name__)


class SpaceMouse:
    """读取 pyspacemouse；后台线程持续读，避免阻塞控制循环。"""

    def __init__(self):
        import pyspacemouse

        self._sm = pyspacemouse
        dev = pyspacemouse.open()
        if not dev:
            raise RuntimeError("找不到 3D 鼠标（检查 USB / udev 权限: /etc/udev/rules.d 中给 hidraw 读权限）")
        self._dev = dev if dev is not True else None
        self._lock = threading.Lock()
        self._axes = np.zeros(6)
        self._buttons = [0, 0]
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True, name="spacemouse")
        self._t.start()

    def _loop(self):
        while self._running:
            st = self._dev.read() if self._dev is not None and hasattr(self._dev, "read") else self._sm.read()
            if st is not None:
                with self._lock:
                    self._axes = np.array([st.x, st.y, st.z, st.roll, st.pitch, st.yaw], dtype=np.float64)
                    self._buttons = list(st.buttons)
            time.sleep(0.002)

    def read(self) -> tuple[np.ndarray, list[int]]:
        with self._lock:
            return self._axes.copy(), list(self._buttons)

    def close(self):
        self._running = False


class CartesianTeleop:
    """把 3D 鼠标输入变为目标关节角。

    目标位姿在“指令空间”积分（不读回实际位姿），避免接触时位置误差导致漂移；
    接触时实际位姿跟不上，目标与实际之差就是“推”的意图，会体现在记录的动作里。
    """

    def __init__(self, cfg: dict, arm, mouse: SpaceMouse | None = None):
        tc = cfg["teleop"]
        self.cfg = tc
        self.arm = arm
        self.mouse = mouse or SpaceMouse()
        self.dt = 1.0 / cfg["task"]["fps"]
        self.axis_map = np.asarray(tc["axis_map"], dtype=int)
        self.axis_sign = np.asarray(tc["axis_sign"], dtype=np.float64)
        self.gripper_closed = False
        self._prev_buttons = [0, 0]
        self.target_pose: np.ndarray | None = None
        self.target_q: np.ndarray | None = None

    def sync_to_robot(self, q_deg: np.ndarray | None = None):
        """从机械臂当前状态（或给定关节角）重新初始化目标，切换控制权时必须调用。"""
        st = self.arm.state()
        q = st.q_deg if q_deg is None else np.asarray(q_deg)
        pose = self.arm.forward_kin(q) if q_deg is not None else st.tcp_pose
        self.target_q = q.copy()
        self.target_pose = (pose if pose is not None else st.tcp_pose).copy()

    def raw_input(self) -> tuple[np.ndarray, list[int]]:
        axes, buttons = self.mouse.read()
        axes = axes[self.axis_map] * self.axis_sign
        dz = self.cfg["deadzone"]
        axes = np.where(np.abs(axes) < dz, 0.0, (np.abs(axes) - dz) / (1 - dz) * np.sign(axes))
        return axes, buttons

    def activity(self) -> float:
        axes, _ = self.raw_input()
        return float(np.max(np.abs(axes)))

    def step(self) -> tuple[np.ndarray, float]:
        """返回 (目标关节角 deg, 夹爪指令 0/1)。"""
        if self.target_pose is None:
            self.sync_to_robot()
        axes, buttons = self.raw_input()

        gb, fb = self.cfg["gripper_button"], self.cfg["fine_button"]
        if len(buttons) > gb and buttons[gb] and not self._prev_buttons[gb]:
            self.gripper_closed = not self.gripper_closed
        fine = len(buttons) > fb and bool(buttons[fb])
        self._prev_buttons = list(buttons) + [0] * (2 - len(buttons))

        scale = self.cfg["fine_scale"] if fine else 1.0
        v = axes[:3] * self.cfg["max_lin_vel_mm_s"] * scale * self.dt
        w = np.deg2rad(axes[3:] * self.cfg["max_rot_vel_deg_s"] * scale * self.dt)

        pose = self.target_pose.copy()
        rot = self.arm.pose_to_rot(pose)
        if self.cfg["trans_frame"] == "tool":
            v = rot.apply(v)
        pose[:3] += v
        if np.any(w != 0):
            dr = Rotation.from_rotvec(w)
            rot = rot * dr if self.cfg["rot_frame"] == "tool" else dr * rot
            pose[3:] = self.arm.rot_to_euler(rot)

        if np.any(pose != self.target_pose):
            q = self.arm.inverse_kin(pose, self.target_q)
            if q is not None and np.max(np.abs(q - self.target_q)) < 20.0:  # 拒绝构型跳变
                self.target_pose, self.target_q = pose, q
            else:
                log.debug("逆解失败或关节跳变，忽略本次输入")
        return self.target_q.copy(), 1.0 if self.gripper_closed else 0.0
