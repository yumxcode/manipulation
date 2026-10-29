"""原始回合 -> LeRobot 数据集（openpi 训练直接读取）。

在 openpi 的环境里运行（需要 openpi 锁定版本的 lerobot）:
    cd ~/openpi
    # 阶段 1 SFT：只用成功的示范
    uv run python ~/dobot_plug/scripts/convert_to_lerobot.py --repo-id local/dobot_plug_sft

    # 阶段 2 RECAP：示范 + 自主执行(含人工接管)，成功和失败都要，带 advantage 字段
    uv run python ~/dobot_plug/scripts/convert_to_lerobot.py --repo-id local/dobot_plug_recap \\
        --sources demo rollout --include-failures --with-advantage

输出位置：$HF_LEROBOT_HOME/<repo-id>（默认 ~/.cache/huggingface/lerobot/<repo-id>）
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import numpy as np

from dobot_plug.config import camera_slots, load_config
from dobot_plug.episode_io import Episode, list_episodes
from dobot_plug.features import ACTION_DIM, build_action, build_state, state_dim


def keep_mask(ep: Episode, drop_hold: bool, trim_idle: bool) -> np.ndarray:
    ld = ep.lowdim
    n = len(ep)
    keep = np.ones(n, dtype=bool)
    if drop_hold:
        keep &= ld["hold"].astype(int) == 0
    if trim_idle and n > 2:
        dq = np.abs(np.diff(ld["action_q_deg"], axis=0)).max(axis=1)
        dg = np.abs(np.diff(ld["action_gripper"]))
        moving = np.concatenate([[False], (dq > 0.02) | (dg > 0.1)])
        idx = np.flatnonzero(moving)
        if len(idx):
            lo, hi = max(idx[0] - 2, 0), min(idx[-1] + 3, n)
            span = np.zeros(n, dtype=bool)
            span[lo:hi] = True
            keep &= span
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--tasks", nargs="*", default=None, help="默认全部任务")
    ap.add_argument("--sources", nargs="*", default=["demo"], choices=["demo", "rollout"])
    ap.add_argument("--include-failures", action="store_true", help="包含失败回合（RECAP 需要；SFT 不要）")
    ap.add_argument("--with-advantage", action="store_true", help="写入 label_advantage.py 生成的 advantage 字段")
    ap.add_argument("--no-ft", action="store_true", help="state 不包含六维力（需与部署时 --no-ft 一致）")
    ap.add_argument("--keep-hold", action="store_true", help="保留保护停期间的帧（默认丢弃）")
    ap.add_argument("--no-trim", action="store_true", help="不裁剪首尾静止帧")
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    cfg = load_config(args.config)
    raw_dir = args.raw_dir or cfg["data"]["raw_dir"]
    if not pathlib.Path(raw_dir).is_absolute():
        raw_dir = pathlib.Path(cfg["_path"]).resolve().parent.parent / raw_dir
    slots = camera_slots(cfg)
    include_ft = not args.no_ft

    eps = list_episodes(raw_dir, args.tasks, args.sources)
    if not args.include_failures:
        eps = [e for e in eps if e.success]
    if args.max_episodes:
        eps = eps[: args.max_episodes]
    if not eps:
        raise SystemExit(f"{raw_dir} 下没有符合条件的回合")

    fps = eps[0].meta["fps"]
    cams = eps[0].meta["cameras"]
    h, w = eps[0].image(cams[0], 0).shape[:2]
    for e in eps:
        if e.meta["fps"] != fps or e.meta["cameras"] != cams:
            raise SystemExit(f"回合 {e.dir} 的 fps/相机与其他回合不一致")

    out = HF_LEROBOT_HOME / args.repo_id
    if out.exists():
        shutil.rmtree(out)

    features = {
        slots[c]: {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]} for c in cams
    }
    features.update(
        {
            "state": {"dtype": "float32", "shape": (state_dim(include_ft),), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
            "advantage": {"dtype": "float32", "shape": (1,), "names": ["advantage"]},
            "intervention": {"dtype": "float32", "shape": (1,), "names": ["intervention"]},
            "success": {"dtype": "float32", "shape": (1,), "names": ["success"]},
        }
    )
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, robot_type="dobot_cr5af", fps=fps, features=features,
        image_writer_threads=10, image_writer_processes=4,
    )

    n_frames, n_pos = 0, 0
    for k, ep in enumerate(eps):
        ld = ep.lowdim
        keep = keep_mask(ep, drop_hold=not args.keep_hold, trim_idle=not args.no_trim)
        states = build_state(ld["q_deg"], ld["gripper_pos"], ld["ft"], include_ft)
        actions = build_action(ld["action_q_deg"], ld["action_gripper"])
        if args.with_advantage:
            adv = ep.load_npz("advantage.npz")
            if adv is None:
                raise SystemExit(f"{ep.dir} 没有 advantage.npz，先运行 dobot-recap-label-advantage")
            adv_label = adv["label"].astype(np.float32)
        else:
            adv_label = np.ones(len(ep), dtype=np.float32)
        for i in np.flatnonzero(keep):
            frame = {slots[c]: ep.image(c, int(i)) for c in cams}
            frame.update(
                state=states[i], actions=actions[i],
                advantage=adv_label[i : i + 1], intervention=ld["intervention"][i : i + 1].astype(np.float32),
                success=np.array([float(ep.success)], dtype=np.float32), task=ep.meta["prompt"],
            )
            ds.add_frame(frame)
        ds.save_episode()
        n_frames += int(keep.sum())
        n_pos += int(adv_label[keep].sum())
        print(f"[{k + 1}/{len(eps)}] {ep.dir.name} {ep.meta['task_key']} {ep.meta['source']} "
              f"{'成功' if ep.success else '失败'} {int(keep.sum())}/{len(ep)} 帧")

    print(f"\n完成：{len(eps)} 回合，{n_frames} 帧 -> {out}")
    if args.with_advantage:
        print(f"advantage=positive 比例: {n_pos / max(n_frames, 1):.1%}")
    print(f"state 维度 = {state_dim(include_ft)}（include_ft={include_ft}），部署时 run_policy.py 须保持一致")


if __name__ == "__main__":
    main()
