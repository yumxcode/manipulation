"""用官方 dobot_sdk + 假控制器跑通 CR5AF 驱动层：python tests/test_with_fake_controller.py"""

from __future__ import annotations

import logging
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, "/root/TCP-IP-Python-V4")

from fake_dobot import FakeDobot  # noqa: E402

from dobot_plug.config import load_config, merged  # noqa: E402
from dobot_plug.robot import CR5AF  # noqa: E402
from dobot_plug.teleop import CartesianTeleop  # noqa: E402


def main():
    logging.basicConfig(level=logging.WARNING)
    fake = FakeDobot()
    cfg = merged(load_config(), **{"robot.ip": "127.0.0.1", "robot.workspace_mm": {"x": [-1e4, 1e4], "y": [-1e4, 1e4], "z": [-1e4, 1e4]}})
    arm = CR5AF(cfg)
    arm.connect()
    try:
        assert {"RequestControl", "EnableRobot", "EnableFTSensor", "SixForceHome"} <= set(fake.log), fake.log
        st = arm.state()
        assert np.allclose(st.q_deg, fake.q), (st.q_deg, fake.q)
        assert np.allclose(st.tcp_pose, fake.q * 10)
        print("PASS connect/state")

        q = arm.inverse_kin(st.tcp_pose, st.q_deg)
        assert q is not None and np.allclose(q, st.q_deg, atol=1e-4)
        assert np.allclose(arm.forward_kin(st.q_deg), st.tcp_pose)
        print("PASS InverseKin/PositiveKin 解析")

        arm.start_servo()
        q0 = st.q_deg.copy()
        goal = q0 + np.array([0, 0, 0, 0, 0, 2.0])
        for _ in range(15):
            arm.set_target(goal)
            time.sleep(1 / 15)
        time.sleep(0.3)
        assert np.allclose(arm.state().q_deg, goal, atol=1e-3), arm.state().q_deg
        hz = fake.servo_count / 1.3
        print(f"PASS ServoJ 流式控制（~{hz:.0f} Hz）")

        # 安全：力超限 -> HOLD，目标被忽略
        fake.ft = np.array([0, 0, 80.0, 0, 0, 0])
        time.sleep(0.05)
        reason = arm.check_safety(arm.state())
        assert reason and "合力" in reason
        arm.enter_hold(reason)
        held = fake.q.copy()
        arm.set_target(held + 5)
        time.sleep(0.3)
        assert np.allclose(fake.q, held, atol=1e-3)
        fake.ft = np.zeros(6)
        arm.clear_hold()
        assert arm.hold_reason is None
        print("PASS 保护停")

        class _Mouse:
            def read(self):
                return np.array([0, 0, 1.0, 0, 0, 0]), [0, 0]

        tel = CartesianTeleop(cfg, arm, mouse=_Mouse())
        tel.sync_to_robot()
        z0 = arm.state().tcp_pose[2]
        for _ in range(15):
            qq, _ = tel.step()
            arm.set_target(qq)
            time.sleep(1 / 15)
        time.sleep(0.4)
        dz = arm.state().tcp_pose[2] - z0
        assert abs(dz - cfg["teleop"]["max_lin_vel_mm_s"]) < 1.0, dz
        print(f"PASS 遥操作 -> InverseKin -> ServoJ（1 秒上移 {dz:.1f} mm）")

        home = q0.tolist()
        arm.move_joints_blocking(home)
        assert np.allclose(arm.state().q_deg, home, atol=1e-3) and arm._servo is not None  # noqa: SLF001
        print("PASS 回零位 + 伺服线程重启")
    finally:
        arm.close()
        fake.close()


if __name__ == "__main__":
    main()
