"""把机械臂、相机、夹爪组合成一个环境：get_obs() / command() / 安全检查 / 回零位。"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

from .cameras import CameraRig
from .config import camera_slots
from .gripper import make_gripper
from .robot import CR5AF, RobotState

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Obs:
    robot: RobotState
    gripper_pos: float
    images: dict[str, np.ndarray]          # 相机名 -> RGB

    def images_by_slot(self, slots: dict[str, str]) -> dict[str, np.ndarray]:
        return {slots[name]: img for name, img in self.images.items()}


class DobotEnv:
    def __init__(self, cfg: dict, enable_cameras: bool = True):
        self.cfg = cfg
        self.arm = CR5AF(cfg)
        self.cams = CameraRig(cfg) if enable_cameras else None
        self.gripper = None
        self.slots = camera_slots(cfg)

    def start(self):
        self.arm.connect(enable=True)
        self.gripper = make_gripper(self.cfg, self.arm)
        self.gripper.start()
        if self.cams:
            self.cams.start()
        self.arm.start_servo()

    def close(self):
        if self.cams:
            self.cams.stop()
        if self.gripper:
            self.gripper.stop()
        self.arm.close()

    def get_obs(self) -> Obs:
        return Obs(
            robot=self.arm.state(),
            gripper_pos=self.gripper.position(),
            images=self.cams.get() if self.cams else {},
        )

    def command(self, q_deg: np.ndarray, gripper: float):
        self.arm.set_target(q_deg)
        if self.arm.hold_reason is None:
            self.gripper.command(gripper)

    def safety(self, obs: Obs) -> str | None:
        reason = self.arm.check_safety(obs.robot)
        if reason:
            self.arm.enter_hold(reason)
        return self.arm.hold_reason

    def go_home(self):
        home = self.cfg["robot"].get("home_joints_deg")
        if home is None:
            log.warning("未配置 robot.home_joints_deg，跳过回零位")
            return
        self.arm.move_joints_blocking(home)  # 内部会停止并以新位置重启伺服线程
