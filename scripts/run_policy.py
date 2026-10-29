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
import pathlib
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dobot_plug.config import load_config, max_episode_steps  # noqa: E402
from dobot_plug.env import DobotEnv, Obs  # noqa: E402
from dobot_plug.episode_io import EpisodeWriter  # noqa: E402
from dobot_plug.features import RECAP_SUFFIX_POS, action_to_command, build_state, policy_observation  # noqa: E402
from dobot_plug.keyboard import KeyReader  # noqa: E402
from dobot_plug.teleop import CartesianTeleop  # noqa: E402

log = logging.getLogger("run_policy")


class AsyncPolicy:
    """后台线程做推理：执行当前动作块的同时请求下一个块，避免机械臂停顿。

    每个请求记录发出时的控制步号 tick；动作块的第 k 个动作对应 tick+k，
    取用时按当前步号对齐（跳过推理延迟期间已经过去的动作）。
    """

    def __init__(self, host: str, port: int, resize: int = 224):
        from openpi_client import image_tools, websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.image_tools = image_tools
        self.resize = resize
        self._lock = threading.Lock()
        self._req = None
        self._busy = False
        self.chunk: np.ndarray | None = None
        self.chunk_tick = -1
        self.generation = 0
        self.latency = 0.0
        self._cv = threading.Condition(self._lock)
        threading.Thread(target=self._loop, daemon=True, name="policy").start()
        log.info("策略服务器 metadata: %s", self.client.get_server_metadata())

    def preprocess(self, images_by_slot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        it = self.image_tools
        return {k: it.convert_to_uint8(it.resize_with_pad(v, self.resize, self.resize)) for k, v in images_by_slot.items()}

    def request(self, obs: dict, tick: int):
        with self._cv:
            if self._busy or self._req is not None:
                return False
            self._req = (obs, tick, self.generation)
            self._cv.notify()
            return True

    def invalidate(self):
        with self._lock:
            self.generation += 1
            self.chunk = None
            self.chunk_tick = -1

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy or self._req is not None

    def _loop(self):
        while True:
            with self._cv:
                while self._req is None:
                    self._cv.wait()
                obs, tick, gen = self._req
                self._req = None
                self._busy = True
            t0 = time.time()
            try:
                actions = np.asarray(self.client.infer(obs)["actions"])
            except Exception as e:  # noqa: BLE001
                log.error("推理失败: %s", e)
                actions = None
            with self._lock:
                self._busy = False
                self.latency = time.time() - t0
                if actions is not None and gen == self.generation:
                    self.chunk, self.chunk_tick = actions, tick

    def action_at(self, tick: int) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self.chunk is None:
                return None, -1
            idx = tick - self.chunk_tick
            if idx >= len(self.chunk):
                return None, idx
            return self.chunk[idx], idx


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
                        action, idx = policy.action_at(tick)
                        if (action is None or idx >= args.replan_steps) and not policy.busy:
                            policy.request(make_policy_obs(obs), tick)
                        if action is None:
                            status = "waiting"   # 等第一个动作块 / 动作块耗尽：保持
                        else:
                            q_cmd, g_cmd = action_to_command(action)
                            dev = q_cmd - q_now
                            if np.max(np.abs(dev)) > args.max_joint_dev_deg:
                                log.warning("策略目标偏离当前位置 %.1f°，已限幅", np.max(np.abs(dev)))
                                q_cmd = q_now + np.clip(dev, -args.max_joint_dev_deg, args.max_joint_dev_deg)
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
