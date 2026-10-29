"""RECAP 第 1 步：训练分布式价值模型 V(o_t) -> 归一化“剩余步数”回报。

π*0.6 原文用的是 VLM 价值模型；这里是单场景（插/拔两个任务），用预训练视觉骨干 + 多相机融合
+ state + 任务嵌入，足够且训练很快（单卡几十分钟）。骨干可选：
    resnet18（torchvision 预训练，默认）| dinov2_vits14（torch.hub，泛化更好，需联网下载权重）

用法:
    python scripts/recap/train_value.py --out checkpoints/value/v1
    # 只用部分数据源 / 任务
    python scripts/recap/train_value.py --sources demo rollout --tasks insert --out ...

注意：价值模型需要同时看到成功和失败的回合，否则学不出“离失败有多近”。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from dobot_plug.config import load_config  # noqa: E402
from dobot_plug.episode_io import Episode, list_episodes  # noqa: E402
from dobot_plug.features import build_state  # noqa: E402
from dobot_plug.recap import Bins, normalized_returns  # noqa: E402

IMG = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pad_resize(img: np.ndarray, size: int = IMG) -> torch.Tensor:
    """与 openpi 一致的保持比例缩放 + 补零，返回 (3,size,size) float [0,1]。"""
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    h, w = t.shape[1:]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    t = F.interpolate(t[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
    out = torch.zeros(3, size, size)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[:, top : top + nh, left : left + nw] = t
    return out


class FrameDataset(Dataset):
    def __init__(self, eps: list[Episode], cams: list[str], tasks: list[str], cfg: dict, include_ft: bool,
                 state_mean=None, state_std=None, augment=False):
        self.eps, self.cams, self.augment = eps, cams, augment
        self.task_idx = {t: i for i, t in enumerate(tasks)}
        self.index = [(k, i) for k, ep in enumerate(eps) for i in range(len(ep))]
        self.returns = [normalized_returns(len(ep), ep.success, cfg) for ep in eps]
        self.states = [build_state(ep.lowdim["q_deg"], ep.lowdim["gripper_pos"], ep.lowdim["ft"], include_ft) for ep in eps]
        if state_mean is None:
            allst = np.concatenate(self.states)
            state_mean, state_std = allst.mean(0), allst.std(0) + 1e-3
        self.state_mean, self.state_std = state_mean.astype(np.float32), state_std.astype(np.float32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, j):
        k, i = self.index[j]
        ep = self.eps[k]
        imgs = []
        for c in self.cams:
            x = pad_resize(ep.image(c, i))
            if self.augment:
                # 亮度/对比度扰动 + 小幅平移
                x = (x * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05)).clamp(0, 1)
                dx, dy = random.randint(-8, 8), random.randint(-8, 8)
                x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
            imgs.append((x - MEAN) / STD)
        state = (self.states[k][i] - self.state_mean) / self.state_std
        return {
            "images": torch.stack(imgs),
            "state": torch.from_numpy(state),
            "task": torch.tensor(self.task_idx[ep.meta["task_key"]]),
            "ret": torch.tensor(self.returns[k][i]),
        }


class ValueNet(nn.Module):
    def __init__(self, num_cams: int, state_dim: int, num_tasks: int, num_bins: int, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet18":
            import torchvision

            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            net.fc = nn.Identity()
            feat = 512
        elif backbone == "dinov2_vits14":
            net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            feat = 384
        else:
            raise ValueError(backbone)
        self.backbone = net
        self.cam_embed = nn.Parameter(torch.zeros(num_cams, feat))
        self.task_embed = nn.Embedding(num_tasks, 64)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, 128), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(num_cams * feat + 64 + 128, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 512), nn.GELU(), nn.Linear(512, num_bins),
        )

    def forward(self, images, state, task):
        b, n = images.shape[:2]
        f = self.backbone(images.flatten(0, 1)).view(b, n, -1) + self.cam_embed
        z = torch.cat([f.flatten(1), self.task_embed(task), self.state_mlp(state)], dim=-1)
        return self.head(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--sources", nargs="*", default=["demo", "rollout"])
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "dinov2_vits14"])
    ap.add_argument("--no-ft", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone-lr", type=float, default=3e-5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--num-bins", type=int, default=201)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cfg = load_config(args.config)
    raw_dir = pathlib.Path(args.raw_dir or cfg["data"]["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = pathlib.Path(cfg["_path"]).resolve().parent.parent / raw_dir
    tasks = args.tasks or list(cfg["task"]["prompts"].keys())
    eps = list_episodes(raw_dir, tasks, args.sources)
    if not eps:
        raise SystemExit("没有数据")
    n_fail = sum(not e.success for e in eps)
    print(f"{len(eps)} 回合（失败 {n_fail}），{sum(len(e) for e in eps)} 帧")
    if n_fail == 0:
        print("!! 警告：没有失败回合，价值模型无法区分好坏状态。建议先跑一批策略自主执行（含失败）再训练。")
    cams = eps[0].meta["cameras"]
    include_ft = not args.no_ft

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(eps))
    n_val = max(1, int(len(eps) * args.val_frac)) if len(eps) >= 10 else 0
    val_eps = [eps[i] for i in perm[:n_val]]
    tr_eps = [eps[i] for i in perm[n_val:]]

    tr = FrameDataset(tr_eps, cams, tasks, cfg, include_ft, augment=True)
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    va = FrameDataset(val_eps, cams, tasks, cfg, include_ft, tr.state_mean, tr.state_std) if val_eps else None
    dl_va = DataLoader(va, batch_size=args.batch_size, num_workers=args.workers) if va else None

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bins = Bins(args.num_bins)
    centers = torch.from_numpy(bins.centers).to(dev)
    model = ValueNet(len(cams), tr.states[0].shape[-1], len(tasks), args.num_bins, args.backbone).to(dev)
    bb = list(model.backbone.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.backbone_lr}, {"params": rest, "lr": args.lr}], weight_decay=1e-4)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.backbone_lr, args.lr], total_steps=total, pct_start=0.05)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for step, b in enumerate(dl):
            target = torch.from_numpy(bins.two_hot(b["ret"].numpy())).to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(b["images"].to(dev), b["state"].to(dev), b["task"].to(dev))
            loss = -(target * F.log_softmax(logits.float(), -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(dl)} loss {loss.item():.4f}")
        msg = f"== epoch {epoch} train_loss {tot / len(dl):.4f}"
        val_metric = tot / len(dl)
        if dl_va:
            model.eval()
            preds, rets = [], []
            with torch.no_grad():
                for b in dl_va:
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        logits = model(b["images"].to(dev), b["state"].to(dev), b["task"].to(dev))
                    preds.append((logits.float().softmax(-1) @ centers).cpu().numpy())
                    rets.append(b["ret"].numpy())
            p, r = np.concatenate(preds), np.concatenate(rets)
            mae = float(np.abs(p - r).mean())
            corr = float(np.corrcoef(p, r)[0, 1]) if p.std() > 0 else 0.0
            msg += f"  val_MAE {mae:.4f}  val_corr {corr:.3f}"
            val_metric = mae
        print(msg)
        ckpt = {
            "model": model.state_dict(), "cams": cams, "tasks": tasks, "include_ft": include_ft,
            "state_mean": tr.state_mean, "state_std": tr.state_std, "num_bins": args.num_bins,
            "backbone": args.backbone, "state_dim": int(tr.states[0].shape[-1]),
        }
        torch.save(ckpt, out / "last.pt")
        if val_metric < best:
            best = val_metric
            torch.save(ckpt, out / "best.pt")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print(f"完成，模型保存在 {out}/best.pt")


if __name__ == "__main__":
    main()
