"""RECAP 第 2 步：用价值模型给每一帧算 N 步优势并二值化，写入各回合目录的 advantage.npz。

    dobot-recap-label-advantage --value checkpoints/value/v1/best.pt
    # 然后转换数据集（在 openpi 环境中）:
    uv run python ~/dobot_plug/scripts/convert_to_lerobot.py --repo-id local/dobot_plug_recap \\
        --sources demo rollout --include-failures --with-advantage

参数说明:
    --n-step            N 步优势的 N（帧）。15Hz 下 15~45 比较合适，默认 30（2 秒）
    --positive-fraction 每个任务取阈值 ε_ℓ 使约这么多比例的自主帧被标为 positive。
                        论文（arXiv:2511.14759）取优势值的 30% 分位数 => 默认 0.7
    --score-demos       示范帧也按优势打分（默认否：论文中示范阶段 I_t 全部为 True，
                        慢/犹豫的示范片段不会被标 negative）
人工接管帧（intervention=1 且 source=rollout）始终强制为 positive。
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from dobot_plug.apps.recap.train_value import MEAN, STD, ValueNet, pad_resize
from dobot_plug.config import load_config
from dobot_plug.episode_io import list_episodes
from dobot_plug.features import build_state
from dobot_plug.recap import n_step_advantage, normalized_returns, positive_threshold


@torch.no_grad()
def predict_values(model, ep, ckpt, dev, batch: int = 64) -> np.ndarray:
    cams, tasks = ckpt["cams"], ckpt["tasks"]
    states = build_state(ep.lowdim["q_deg"], ep.lowdim["gripper_pos"], ep.lowdim["ft"], ckpt["include_ft"])
    states = (states - ckpt["state_mean"]) / ckpt["state_std"]
    task = tasks.index(ep.meta["task_key"])
    centers = torch.linspace(-1.0, 0.0, ckpt["num_bins"], device=dev)
    out = []
    for s in range(0, len(ep), batch):
        idx = range(s, min(s + batch, len(ep)))
        imgs = torch.stack([torch.stack([(pad_resize(ep.image(c, i)) - MEAN) / STD for c in cams]) for i in idx])
        st = torch.from_numpy(states[s : s + len(idx)].astype(np.float32))
        tk = torch.full((len(idx),), task, dtype=torch.long)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(imgs.to(dev), st.to(dev), tk.to(dev))
        out.append((logits.float().softmax(-1) @ centers).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--value", required=True, help="train_value.py 输出的 best.pt")
    ap.add_argument("--sources", nargs="*", default=["demo", "rollout"])
    ap.add_argument("--n-step", type=int, default=30)
    ap.add_argument("--positive-fraction", type=float, default=0.7,
                    help="阈值 ε_ℓ 取 (1 - 该值) 分位数；论文取 30%% 分位数 => 0.7")
    ap.add_argument("--score-demos", action="store_true",
                    help="示范帧也按优势打分（默认示范帧全部 positive，与论文 SFT 阶段一致）")
    ap.add_argument("--reuse-values", action="store_true", help="已有 value.npz 时不重新推理")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = pathlib.Path(args.raw_dir or cfg["data"]["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = pathlib.Path(cfg["_path"]).resolve().parent.parent / raw_dir

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.value, map_location="cpu", weights_only=False)
    model = ValueNet(len(ckpt["cams"]), ckpt["state_dim"], len(ckpt["tasks"]), ckpt["num_bins"], ckpt["backbone"])
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()

    eps = list_episodes(raw_dir, ckpt["tasks"], args.sources)
    print(f"{len(eps)} 回合待标注")

    # 1) 价值 + 优势
    per_ep = []
    for k, ep in enumerate(eps):
        cached = ep.load_npz("value.npz") if args.reuse_values else None
        values = cached["value"] if cached is not None else predict_values(model, ep, ckpt, dev)
        returns = normalized_returns(len(ep), ep.success, cfg)
        np.savez(ep.dir / "value.npz", value=values, ret=returns)
        adv = n_step_advantage(returns, values, args.n_step)
        forced = np.zeros(len(ep), dtype=bool)
        if ep.meta["source"] == "rollout":
            forced |= ep.lowdim["intervention"].astype(bool)
        elif not args.score_demos:
            forced[:] = True
        per_ep.append((ep, adv, forced))
        print(f"[{k + 1}/{len(eps)}] {ep.dir.name} {'成功' if ep.success else '失败'} "
              f"V0={values[0]:+.3f} G0={returns[0]:+.3f} mean|V-G|={np.abs(values - returns).mean():.3f}")

    # 2) 每个任务一个阈值（只用非强制帧统计）
    thresholds = {}
    for task in ckpt["tasks"]:
        a = np.concatenate([adv[~forced] for ep, adv, forced in per_ep if ep.meta["task_key"] == task] or [np.zeros(0)])
        if len(a) == 0:
            continue
        thresholds[task] = positive_threshold(a, args.positive_fraction)
    print("阈值 ε:", {k: round(v, 4) for k, v in thresholds.items()})

    # 3) 写入
    stats = {}
    for ep, adv, forced in per_ep:
        eps_t = thresholds.get(ep.meta["task_key"], 0.0)
        label = (adv > eps_t) | forced
        np.savez(ep.dir / "advantage.npz", advantage=adv, label=label.astype(np.float32),
                 threshold=np.float32(eps_t), n_step=np.int32(args.n_step))
        key = (ep.meta["task_key"], ep.meta["source"], "成功" if ep.success else "失败")
        s = stats.setdefault(key, [0, 0])
        s[0] += int(label.sum())
        s[1] += len(label)
    print("\npositive 比例（按 任务/来源/结果）:")
    for key, (p, n) in sorted(stats.items()):
        print(f"  {key}: {p}/{n} = {p / max(n, 1):.1%}")
    print("\n合理的现象：失败回合的 positive 比例明显低于成功回合；如果差不多，说明价值模型没学好（数据太少/失败样本太少）。")


if __name__ == "__main__":
    main()
