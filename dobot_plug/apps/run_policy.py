"""部署 π0.5 策略（openpi 策略服务器）+ 人工接管 + 自动录制自主执行数据（供 RECAP 使用）。

1) GPU 机器上启动策略服务器（openpi 仓库里）:
   uv run scripts/serve_policy.py policy:checkpoint \\
       --policy.config=pi05_dobot_plug_lora --policy.dir=checkpoints/pi05_dobot_plug_lora/sft_v1/29999

2) 机械臂电脑上:
   python scripts/run_policy.py --task insert --host <GPU机器IP>
   RECAP 阶段训练出的策略加 --recap（prompt 末尾带 "Advantage: positive"）

按键:
    空格  开始一个回合（策略开始控制）
    i     人工接管 / 交还策略（推动 3D 鼠标超过阈值也会自动接管）
    p     暂停/继续（暂停时保持当前位置）
    s / f 回合结束并标记 成功 / 失败（默认保存录像，加 --no-record 不保存）
    d     丢弃当前回合
    t     切换任务
    h     回零位（回合外）
    c     清除保护停
    q     退出

RECAP 的数据闭环：策略自主执行 -> 失败前人工接管纠正(intervention=1) -> 标记成功/失败。
这些回合都存在 data/raw/<task>/ 下，source=rollout。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from dobot_plug.config import load_config, max_episode_steps
from dobot_plug.env import DobotEnv, Obs
from dobot_plug.episode_io import EpisodeWriter
from dobot_plug.features import RECAP_SUFFIX_POS, build_state, policy_observation
from dobot_plug.keyboard import KeyReader
from dobot_plug.policy_client import ActionChunkExecutor, AsyncPolicy
from dobot_plug.teleop import CartesianTeleop

log = logging.getLogger("run_policy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--task", default="insert")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--recap", action="store_true", help="prompt 追加 'Advantage: positive'（RECAP 训练后的策略）")
    ap.add_argument("--no-ft", action="store_true", help="与训练数据转换时的 --no-ft 保持一致")
    ap.add_argument("--replan-steps", type=int, default=8, help="执行多少步后请求下一个动作块（≤ action_horizon=15）")
    ap.add_argument("--max-joint-dev-deg", type=float, default=8.0, help="策略目标与当前关节角的最大允许偏差")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--policy-tag", default="", help="记录到 meta.json，便于区分不同 checkpoint 的数据")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    tasks = list(cfg["task"]["prompts"].keys())
    task = args.task
    fps = cfg["task"]["fps"]
    dt = 1.0 / fps
    max_steps = max_episode_steps(cfg)
    include_ft = not args.no_ft

    env = DobotEnv(cfg)
    env.start()
    teleop = CartesianTeleop(cfg, env.arm)
    teleop.sync_to_robot()
    policy = AsyncPolicy(args.host, args.port)
    executor = ActionChunkExecutor(policy, args.max_joint_dev_deg)

    def prompt_of(task_key: str) -> str:
        p = cfg["task"]["prompts"][task_key]
        return p + RECAP_SUFFIX_POS if args.recap else p

    def make_policy_obs(obs: Obs) -> dict:
        state = build_state(obs.robot.q_deg, obs.gripper_pos, obs.robot.ft, include_ft)
        return policy_observation(policy.preprocess(obs.images_by_slot(env.slots)), state, prompt_of(task))

    running = False     # 回合进行中
    paused = False
    human = False       # 人工接管中
    writer: EpisodeWriter | None = None
    tick = 0
    last_cmd = (env.arm.state().q_deg, 0.0)
    last_print = 0.0
    stats = {"ok": 0, "fail": 0}
    print(__doc__)
    print(f"当前任务: [{task}] {prompt_of(task)!r}")

    def to_human():
        nonlocal human
        human = True
        teleop.sync_to_robot(env.arm.commanded_q)
        teleop.gripper_closed = last_cmd[1] >= 0.5
        print("\n>>> 人工接管")

    def to_policy():
        nonlocal human
        human = False
        policy.invalidate()
        print("\n>>> 交还策略")

    def end_episode(key: str):
        nonlocal running, writer, paused, human
        running, paused, human = False, False, False
        policy.invalidate()
        if key in ("s", "f"):
            stats["ok" if key == "s" else "fail"] += 1
        if writer is not None:
            if key == "d" or writer.n < 5:
                writer.discard()
                print("\n已丢弃")
            else:
                path = writer.save(success=(key == "s"))
                print(f"\n已保存 ({'成功' if key == 's' else '失败'}, {writer.n} 帧) -> {path}")
            writer = None
        n = stats["ok"] + stats["fail"]
        if n:
            print(f"本次成功率 {stats['ok']}/{n} = {stats['ok'] / n:.0%}")
        teleop.sync_to_robot()

    with KeyReader() as keys:
        next_t = time.time()
        try:
            while True:
                key = keys.poll()
                if key == "q":
                    if writer:
                        writer.discard()
                    break
                if not running:
                    if key == "t":
                        task = tasks[(tasks.index(task) + 1) % len(tasks)]
                        print(f"\n切换任务 -> [{task}] {prompt_of(task)!r}")
                    elif key == "h":
                        env.go_home()
                        teleop.sync_to_robot()
                    elif key == " ":
                        running, paused, human, tick = True, False, False, 0
                        policy.invalidate()
                        if not args.no_record:
                            writer = EpisodeWriter(
                                cfg["data"]["raw_dir"], task, cfg["task"]["prompts"][task], fps,
                                list(env.cams.cams.keys()), source="rollout",
                                jpeg_quality=cfg["cameras"]["jpeg_quality"],
                                extra_meta={"policy": args.policy_tag, "recap_prompt": args.recap},
                            )
                        print(f"\n● 回合开始 [{task}]")
                else:
                    if key in ("s", "f", "d"):
                        end_episode(key)
                    elif key == "i":
                        to_policy() if human else to_human()
                    elif key == "p":
                        paused = not paused
                        if not paused:
                            policy.invalidate()
                        print("\n暂停" if paused else "\n继续")
                if key == "c":
                    env.arm.clear_hold()
                    teleop.sync_to_robot()
                    policy.invalidate()
                    print("\n已清除保护停")

                # ---------------- 控制 ----------------
                obs = env.get_obs()
                hold = env.safety(obs)
                q_now = obs.robot.q_deg
                q_cmd, g_cmd = env.arm.commanded_q, last_cmd[1]
                intervention = 1
                status = "idle"

                if hold:
                    teleop.sync_to_robot()
                    q_cmd = q_now
                    status = "HOLD"
                elif not running:
                    q_cmd, g_cmd = teleop.step()   # 回合外可以用 3D 鼠标摆位置
                    env.command(q_cmd, g_cmd)
                elif paused:
                    status = "paused"
                else:
                    if not human and teleop.activity() > cfg["teleop"]["takeover_threshold"]:
                        to_human()
                    if human:
                        q_cmd, g_cmd = teleop.step()
                        env.command(q_cmd, g_cmd)
                        status = "HUMAN"
                    else:
                        intervention = 0
                        executor.maybe_request(lambda obs=obs: make_policy_obs(obs), tick, args.replan_steps)
                        cmd = executor.command(tick, q_now)
                        if cmd is None:
                            status = "waiting"   # 等第一个动作块 / 动作块耗尽：保持
                        else:
                            q_cmd, g_cmd, idx = cmd
                            env.command(q_cmd, g_cmd)
                            status = f"policy[{idx:2d}]"
                    tick += 1

                last_cmd = (q_cmd, g_cmd)
                if running and writer is not None and not paused:
                    writer.add(
                        obs.images,
                        t=obs.robot.t, q_deg=q_now, q_target_deg=obs.robot.q_target_deg,
                        tcp_pose=obs.robot.tcp_pose, ft=obs.robot.ft, gripper_pos=obs.gripper_pos,
                        action_q_deg=q_cmd, action_gripper=g_cmd, intervention=intervention,
                        hold=int(bool(hold)),
                    )
                    if writer.n == max_steps:
                        paused = True
                        print(f"\n达到最长 {max_steps} 帧，已暂停：按 s / f / d 结束")

                if time.time() - last_print > 0.25:
                    f = np.linalg.norm(obs.robot.ft[:3])
                    n = writer.n if writer else tick
                    sys.stdout.write(
                        f"\r{'●' if running else '○'} [{task}] {status:<11} step={n:4d} "
                        f"|F|={f:5.1f}N grip={obs.gripper_pos:.2f} infer={policy.latency * 1000:4.0f}ms"
                        + (f" HOLD: {hold}" if hold else "") + "     "
                    )
                    sys.stdout.flush()
                    last_print = time.time()

                next_t += dt
                sleep = next_t - time.time()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.time()
        finally:
            env.close()


if __name__ == "__main__":
    main()
