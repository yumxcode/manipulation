"""正式评估 harness：固定布局集合上的成功率 + Wilson 95% 置信区间，输出可比对的报告。

协议（对齐机器人学习通行做法：π0 论文每任务 10 回合、ALOHA 25 回合、
Dream Machines 实证 40 固定布局各 1 回合；小样本成功率一律报 Wilson 95% CI）：
  1. 准备一个固定布局清单（--layouts-file，见 configs/layouts.example.yaml；不给则按序号约定），
     每回合开始前按清单布置插座/插头位置——不同 checkpoint 用同一组布局，结果才可比较。
  2. 每回合纯策略自主执行，不提供人工接管。
  3. 操作员只做判定：s=成功，f=失败，a=中止判无效（布局摆错、外因中断等非策略因素）。
  4. 结束输出 成功数/有效回合数 + Wilson 95% CI，写 evals/<run>/report.json。
     回合同时照常存入 data/raw（source=rollout, eval=1），可并入 RECAP 数据。

    python scripts/eval.py --task insert --episodes 20 --host <GPU机IP> --policy-tag sft_v1
    python scripts/eval.py --task insert --episodes 20 --recap --policy-tag recap_r1   # RECAP 策略
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import pathlib
import sys
import time

import numpy as np
import yaml

from dobot_plug.config import load_config, max_episode_steps
from dobot_plug.env import DobotEnv
from dobot_plug.episode_io import EpisodeWriter
from dobot_plug.features import RECAP_SUFFIX_POS, build_state, policy_observation
from dobot_plug.keyboard import KeyReader
from dobot_plug.policy_client import ActionChunkExecutor, AsyncPolicy
from dobot_plug.stats import wilson_ci

log = logging.getLogger("eval")


def load_layouts(path: str | None, tasks: list[str], episodes: int) -> dict[str, list[str]]:
    """布局清单：yaml 里每个任务一个字符串列表；没给就按 “布局 i/N” 编号。"""
    layouts: dict[str, list[str]] = {t: [] for t in tasks}
    if path:
        data = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or {}
        for t in tasks:
            layouts[t] = [str(x) for x in data.get(t, [])]
    for t in tasks:
        if not layouts[t]:
            layouts[t] = [f"布局 {i + 1}/{episodes}" for i in range(episodes)]
        elif len(layouts[t]) < episodes:
            q = layouts[t]
            layouts[t] = [q[i % len(q)] for i in range(episodes)]
    return layouts


def summarize(rows: list[dict]) -> dict:
    """rows: [{'task':..., 'success': bool|None(无效), ...}] -> 汇总（含 Wilson CI）。"""
    out: dict = {"tasks": {}}
    for task in sorted({r["task"] for r in rows}):
        tr = [r for r in rows if r["task"] == task]
        valid = [r for r in tr if r["success"] is not None]
        k = sum(1 for r in valid if r["success"])
        n = len(valid)
        lo, hi = wilson_ci(k, n)
        out["tasks"][task] = {
            "success": k, "valid": n, "invalid": len(tr) - n,
            "success_rate": round(k / n, 4) if n else None,
            "wilson95": [round(lo, 4), round(hi, 4)],
            "mean_frames": round(float(np.mean([r["frames"] for r in valid])), 1) if valid else None,
        }
    valid = [r for r in rows if r["success"] is not None]
    k = sum(1 for r in valid if r["success"])
    lo, hi = wilson_ci(k, len(valid))
    out["overall"] = {"success": k, "valid": len(valid), "invalid": len(rows) - len(valid),
                      "success_rate": round(k / len(valid), 4) if valid else None,
                      "wilson95": [round(lo, 4), round(hi, 4)]}
    return out


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 64)
    print(f"{'任务':<10}{'成功/有效':>12}{'成功率':>9}{'Wilson 95% CI':>22}{'无效':>5}")
    for task, s in summary["tasks"].items():
        ci = f"[{s['wilson95'][0]:.0%}, {s['wilson95'][1]:.0%}]" if s["valid"] else "-"
        rate = f"{s['success_rate']:.0%}" if s["valid"] else "-"
        cells = f"{s['success']}/{s['valid']}"
        print(f"{task:<10}{cells:>12}{rate:>9}{ci:>22}{s['invalid']:>5}")
    o = summary["overall"]
    ci = f"[{o['wilson95'][0]:.0%}, {o['wilson95'][1]:.0%}]"
    print("-" * 64)
    print(f"{'合计':<10}{str(o['success']) + '/' + str(o['valid']):>12}{o['success_rate']:.0%}{ci:>22}{o['invalid']:>5}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--task", default="insert")
    ap.add_argument("--episodes", type=int, default=20, help="评估回合数（建议 20~40）")
    ap.add_argument("--layouts-file", default=None, help="固定布局清单 yaml（见 configs/layouts.example.yaml）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--recap", action="store_true", help="prompt 追加 'Advantage: positive'（RECAP 训练后的策略）")
    ap.add_argument("--no-ft", action="store_true", help="与训练数据转换时的 --no-ft 保持一致")
    ap.add_argument("--replan-steps", type=int, default=8)
    ap.add_argument("--max-joint-dev-deg", type=float, default=8.0)
    ap.add_argument("--no-record", action="store_true", help="不把评估回合存入 data/raw")
    ap.add_argument("--policy-tag", default="", help="被评估的 checkpoint 名，写入报告与回合 meta")
    ap.add_argument("--report-dir", default="evals")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    tasks = list(cfg["task"]["prompts"].keys())
    task = args.task if args.task in tasks else tasks[0]
    fps = cfg["task"]["fps"]
    dt = 1.0 / fps
    max_steps = max_episode_steps(cfg)
    include_ft = not args.no_ft
    layouts = load_layouts(args.layouts_file, [task], args.episodes)

    env = DobotEnv(cfg)
    env.start()
    policy = AsyncPolicy(args.host, args.port)
    executor = ActionChunkExecutor(policy, args.max_joint_dev_deg)

    def prompt_of(tk: str) -> str:
        p = cfg["task"]["prompts"][tk]
        return p + RECAP_SUFFIX_POS if args.recap else p

    def make_policy_obs(obs):
        state = build_state(obs.robot.q_deg, obs.gripper_pos, obs.robot.ft, include_ft)
        return policy_observation(policy.preprocess(obs.images_by_slot(env.slots)), state, prompt_of(task))

    rows: list[dict] = []
    print(__doc__)
    run_dir = pathlib.Path(args.report_dir) / f"{args.policy_tag or 'policy'}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with KeyReader() as keys:
        try:
            for i in range(args.episodes):
                layout = layouts[task][i]
                print(f"\n───── 回合 {i + 1}/{args.episodes} [{task}] 布局: {layout} ─────")
                print("布置好场景后按 空格 开始（策略自主执行，不许接管）；按 q 提前结束评估")
                while True:
                    key = keys.poll()
                    if key in (" ", "\n", "\r"):
                        break
                    if key == "q":
                        raise KeyboardInterrupt
                    time.sleep(0.02)

                writer = EpisodeWriter(
                    cfg["data"]["raw_dir"], task, cfg["task"]["prompts"][task], fps,
                    list(env.cams.cams.keys()), source="rollout", jpeg_quality=cfg["cameras"]["jpeg_quality"],
                    extra_meta={"policy": args.policy_tag, "recap_prompt": args.recap, "eval": True,
                                "layout": layout, "eval_index": i},
                ) if not args.no_record else None

                outcome: bool | None = None   # True/False=有效成败, None=无效
                tick, paused, last_print = 0, False, 0.0
                g_last = 0.0                    # 上一拍夹爪指令（HOLD/waiting 帧记录时延续，避免污染数据）
                policy.invalidate()
                next_t = time.time()
                while True:
                    key = keys.poll()
                    if key == "a":
                        outcome = None
                        print("\n>> 判无效")
                        break
                    if key in ("s", "f"):
                        outcome = key == "s"
                        break
                    if key == "p":
                        paused = not paused
                        if not paused:
                            policy.invalidate()
                    if key == "c":
                        env.arm.clear_hold()
                        policy.invalidate()
                        print("\n已清除保护停")

                    obs = env.get_obs()
                    hold = env.safety(obs)
                    q_now = obs.robot.q_deg
                    q_cmd, g_cmd, status, intervention = q_now, g_last, "idle", 0
                    if hold:
                        status = "HOLD"
                    elif paused:
                        status = "paused"
                    else:
                        executor.maybe_request(lambda obs=obs: make_policy_obs(obs), tick, args.replan_steps)
                        cmd = executor.command(tick, q_now)
                        if cmd is None:
                            status = "waiting"
                        else:
                            q_cmd, g_cmd, idx = cmd
                            env.command(q_cmd, g_cmd)
                            g_last = g_cmd
                            status = f"policy[{idx:2d}]"
                        tick += 1

                    if writer is not None and not paused:
                        writer.add(
                            obs.images,
                            t=obs.robot.t, q_deg=q_now, q_target_deg=obs.robot.q_target_deg,
                            tcp_pose=obs.robot.tcp_pose, ft=obs.robot.ft, gripper_pos=obs.gripper_pos,
                            action_q_deg=q_cmd, action_gripper=g_cmd, intervention=intervention,
                            hold=int(bool(hold)),
                        )
                        if writer.n >= max_steps:
                            paused = True
                            print(f"\n达到最长 {max_steps} 帧，已暂停：按 s=失败 / a=无效 结束")

                    if time.time() - last_print > 0.25:
                        n = writer.n if writer else tick
                        sys.stdout.write(f"\r  {status:<11} step={n:4d} infer={policy.latency * 1000:4.0f}ms"
                                         + (f"  HOLD: {hold}" if hold else "     "))
                        sys.stdout.flush()
                        last_print = time.time()
                    next_t += dt
                    sleep = next_t - time.time()
                    if sleep > 0:
                        time.sleep(sleep)
                    else:
                        next_t = time.time()

                row = {"task": task, "layout": layout, "success": outcome,
                       "frames": writer.n if writer else tick,
                       "policy": args.policy_tag, "recap_prompt": args.recap}
                rows.append(row)
                if writer is not None:
                    if outcome is None:
                        writer.discard()
                    else:
                        path = writer.save(success=outcome)
                        print(f"  已保存 ({'成功' if outcome else '失败'}) -> {path}")
                summary = summarize(rows)
                s = summary["tasks"][task]
                if s["valid"]:
                    ci = wilson_ci(s["success"], s["valid"])
                    print(f"  当前: {s['success']}/{s['valid']} = {s['success_rate']:.0%}"
                          f"  CI [{ci[0]:.0%}, {ci[1]:.0%}]" + (f"  无效 {s['invalid']}" if s["invalid"] else ""))
        except KeyboardInterrupt:
            print("\n中断")
        finally:
            env.close()

    if not rows:
        print("没有完成的回合，未生成报告")
        return
    summary = summarize(rows)
    print_summary(summary)
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "policy_tag": args.policy_tag, "recap_prompt": args.recap,
        "task": task, "episodes_planned": args.episodes,
        "layouts": layouts[task], "config": args.config, "summary": summary, "episodes": rows,
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
