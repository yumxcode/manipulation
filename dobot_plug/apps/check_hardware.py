"""硬件自检。逐项运行，确认无误后再采数据。

    python scripts/check_hardware.py --cams            # 列出 RealSense 序列号并各存一张图到 data/check/
    python scripts/check_hardware.py --robot           # 连接 CR5AF，打印关节角/位姿/六维力，校验逆解
    python scripts/check_hardware.py --robot --servo   # ServoJ 流式小幅运动测试（J6 ±3°），确认伺服链路
    python scripts/check_hardware.py --gripper         # 夹爪开/合
    python scripts/check_hardware.py --robot --teleop  # 3D 鼠标遥操作 20 秒，确认轴向映射（不录制）
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

import numpy as np

from dobot_plug.config import load_config


def check_cams(cfg):
    from PIL import Image

    from dobot_plug.cameras import CameraRig, list_realsense

    devs = list_realsense()
    print("检测到的 RealSense:")
    for d in devs:
        print(f"  {d['name']:<28} serial={d['serial']}  USB{d['usb']}")
        if not str(d["usb"]).startswith("3"):
            print("    !! 不是 USB3 连接，多相机时会掉帧")
    configured = {str(v["serial"]) for v in cfg["cameras"]["devices"].values()}
    missing = configured - {d["serial"] for d in devs}
    if missing:
        print(f"配置里的序列号未找到: {missing} —— 请把上面的序列号填进 configs/hardware.yaml")
        return
    rig = CameraRig(cfg)
    rig.start()
    out = pathlib.Path("data/check")
    out.mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 0
    while time.time() - t0 < 3:
        imgs = rig.get()
        n += 1
        time.sleep(1 / 30)
    for name, img in imgs.items():
        Image.fromarray(img).save(out / f"{name}.jpg")
    rig.stop()
    print(f"相机正常，~{n / 3:.0f} 次/秒取图；截图已存到 {out}/ ，检查视角是否覆盖插座与插头")


def check_robot(cfg, servo: bool, teleop: bool):
    from dobot_plug.robot import CR5AF

    arm = CR5AF(cfg)
    arm.connect(enable=servo or teleop)
    try:
        for _ in range(3):
            st = arm.state()
            print(f"mode={st.mode} err={st.error}")
            print(f"  q(deg)   = {np.round(st.q_deg, 2).tolist()}")
            print(f"  tcp      = {np.round(st.tcp_pose, 2).tolist()}")
            print(f"  F/T      = {np.round(st.ft, 2).tolist()}")
            time.sleep(0.3)
        print("\n把当前关节角作为 home 时，填入 configs/hardware.yaml -> robot.home_joints_deg:")
        print(f"  home_joints_deg: {np.round(st.q_deg, 2).tolist()}")

        q_ik = arm.inverse_kin(st.tcp_pose, st.q_deg)
        if q_ik is None:
            print("!! InverseKin 调用失败")
        else:
            print(f"\n逆解校验: max|IK(FK(q)) - q| = {np.max(np.abs(q_ik - st.q_deg)):.4f} deg（应 < 0.01）")
        # 欧拉角约定校验：用 scipy 按配置约定把位姿转旋转矩阵再转回，确认与 FK 一致
        fk = arm.forward_kin(st.q_deg)
        if fk is not None:
            r = arm.pose_to_rot(fk)
            print(f"FK 位姿 = {np.round(fk, 2).tolist()}，往返误差 {np.max(np.abs(arm.rot_to_euler(r) - fk[3:])):.4f} deg")
        sens = np.linalg.norm(st.ft[:3])
        print(f"静止时 |F| = {sens:.2f} N（清零后应 < 2N；偏大检查负载设置 payload_kg/payload_center_mm）")

        if servo:
            print("\nServoJ 测试：J6 +3° 再回原位 ...")
            arm.start_servo()
            q0 = st.q_deg.copy()
            dt = 1.0 / cfg["task"]["fps"]
            for k in range(60):
                q = q0.copy()
                q[5] += 3.0 * np.sin(np.pi * k / 59)
                arm.set_target(q)
                time.sleep(dt)
            time.sleep(0.5)
            arm.stop_servo()
            err = np.max(np.abs(arm.state().q_deg - q0))
            print(f"回到原位误差 {err:.3f} deg {'OK' if err < 0.2 else '!! 偏大'}")

        if teleop:
            from dobot_plug.teleop import CartesianTeleop

            arm.start_servo()
            tel = CartesianTeleop(cfg, arm)
            tel.sync_to_robot()
            print("\n遥操作 20 秒：依次推/拉/扭 3D 鼠标，观察 TCP 运动方向是否符合直觉；不对就改 teleop.axis_map/axis_sign")
            t_end = time.time() + 20
            dt = 1.0 / cfg["task"]["fps"]
            while time.time() < t_end:
                stt = arm.state()
                reason = arm.check_safety(stt)
                if reason:
                    print("\n保护停:", reason)
                    break
                axes, _ = tel.raw_input()
                q, _ = tel.step()
                arm.set_target(q)
                sys.stdout.write(f"\rinput={np.round(axes, 2).tolist()}  tcp={np.round(stt.tcp_pose[:3], 1).tolist()}   ")
                sys.stdout.flush()
                time.sleep(dt)
            arm.stop_servo()
            print()
    finally:
        arm.close()


def check_gripper(cfg):
    from dobot_plug.gripper import make_gripper

    arm = None
    if cfg["gripper"]["type"] == "tool_do":
        from dobot_plug.robot import CR5AF

        arm = CR5AF(cfg)
        arm.connect(enable=False)
    g = make_gripper(cfg, arm)
    g.start()
    for target in (1.0, 0.0, 0.5, 0.0):
        g.command(target)
        time.sleep(1.5)
        print(f"指令 {target:.1f} -> 位置 {g.position():.2f}")
    g.stop()
    if arm:
        arm.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--cams", action="store_true")
    ap.add_argument("--robot", action="store_true")
    ap.add_argument("--servo", action="store_true")
    ap.add_argument("--teleop", action="store_true")
    ap.add_argument("--gripper", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    if args.cams:
        check_cams(cfg)
    if args.robot:
        check_robot(cfg, args.servo, args.teleop)
    if args.gripper:
        check_gripper(cfg)
    if not (args.cams or args.robot or args.gripper):
        print(__doc__)


if __name__ == "__main__":
    main()
