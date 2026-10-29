"""遥操作采集示范数据（3D 鼠标）。

用法:
    python scripts/record.py --task insert
    python scripts/record.py --task unplug

按键（在运行脚本的终端里按，不需要回车）:
    空格  开始录制当前任务
    s     结束并标记【成功】保存
    f     结束并标记【失败】保存（失败的示范对 RECAP 价值模型同样有用）
    d     丢弃当前回合
    t     切换任务 insert <-> unplug（插完直接录拔，自然复位）
    h     回零位（仅未录制时）
    c     清除保护停（力超限/越界/报警后）
    q     退出
3D 鼠标: 左键=夹爪开/合，按住右键=精细模式
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from dobot_plug.config import load_config, max_episode_steps
from dobot_plug.env import DobotEnv
from dobot_plug.episode_io import EpisodeWriter
from dobot_plug.keyboard import KeyReader
from dobot_plug.teleop import CartesianTeleop

log = logging.getLogger("record")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--task", default="insert", help="configs 里 task.prompts 的键")
    ap.add_argument("--operator", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    tasks = list(cfg["task"]["prompts"].keys())
    task = args.task
    fps = cfg["task"]["fps"]
    dt = 1.0 / fps
    max_steps = max_episode_steps(cfg)

    env = DobotEnv(cfg)
    env.start()
    teleop = CartesianTeleop(cfg, env.arm)
    teleop.sync_to_robot()

    writer: EpisodeWriter | None = None
    n_saved = 0
    last_print = 0.0
    print(__doc__)
    print(f"当前任务: [{task}] {cfg['task']['prompts'][task]}")

    with KeyReader() as keys:
        next_t = time.time()
        try:
            while True:
                key = keys.poll()
                if key == "q":
                    if writer:
                        writer.discard()
                    break
                if key == "t" and writer is None:
                    task = tasks[(tasks.index(task) + 1) % len(tasks)]
                    print(f"\n切换任务 -> [{task}] {cfg['task']['prompts'][task]}")
                if key == "h" and writer is None:
                    env.go_home()
                    teleop.sync_to_robot()
                if key == "c":
                    env.arm.clear_hold()
                    teleop.sync_to_robot()
                    print("\n已清除保护停")
                if key == " " and writer is None:
                    writer = EpisodeWriter(
                        cfg["data"]["raw_dir"], task, cfg["task"]["prompts"][task], fps,
                        list(env.cams.cams.keys()), source="demo", jpeg_quality=cfg["cameras"]["jpeg_quality"],
                        extra_meta={"operator": args.operator},
                    )
                    print(f"\n● 开始录制 [{task}]")
                if writer is not None and key in ("s", "f", "d"):
                    if key == "d" or writer.n < 5:
                        writer.discard()
                        print("\n已丢弃")
                    else:
                        path = writer.save(success=(key == "s"))
                        n_saved += 1
                        print(f"\n已保存 ({'成功' if key == 's' else '失败'}, {writer.n} 帧) -> {path}   本次共 {n_saved} 条")
                    writer = None

                # ---------------- 控制 ----------------
                obs = env.get_obs()
                hold = env.safety(obs)
                q_cmd, g_cmd = teleop.step()
                if hold:
                    teleop.sync_to_robot()
                    q_cmd = obs.robot.q_deg
                else:
                    env.command(q_cmd, g_cmd)

                if writer is not None:
                    writer.add(
                        obs.images,
                        t=obs.robot.t, q_deg=obs.robot.q_deg, q_target_deg=obs.robot.q_target_deg,
                        tcp_pose=obs.robot.tcp_pose, ft=obs.robot.ft, gripper_pos=obs.gripper_pos,
                        action_q_deg=q_cmd, action_gripper=g_cmd, intervention=1, hold=int(bool(hold)),
                    )
                    if writer.n >= max_steps:
                        print(f"\n达到最长 {max_steps} 帧，请按 s / f / d 结束本回合")

                if time.time() - last_print > 0.25:
                    f = np.linalg.norm(obs.robot.ft[:3])
                    rec = f"● REC {writer.n:4d}" if writer else "○ idle    "
                    hold_s = f"  HOLD: {hold}" if hold else ""
                    sys.stdout.write(f"\r{rec} [{task}] |F|={f:5.1f}N  grip={obs.gripper_pos:.2f}{hold_s}      ")
                    sys.stdout.flush()
                    last_print = time.time()

                next_t += dt
                sleep = next_t - time.time()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    if writer is not None and -sleep > dt:
                        log.warning("控制循环超时 %.0fms（相机/串口阻塞？）", -sleep * 1000)
                    next_t = time.time()
        finally:
            env.close()


if __name__ == "__main__":
    main()
