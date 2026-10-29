"""数据集质量检查（无硬件，本地即可跑）：对 data/raw 里的原始回合出一份体检报告。

    python scripts/inspect_data.py                       # 汇总 + 警告
    python scripts/inspect_data.py --montage             # 另存每任务抽样的相机拼图到 data/check/
    python scripts/inspect_data.py --json report.json    # 机器可读报告

检查项（对应 LeRobot/openpi 生态的数据质量实践）：
  * 数量：按 任务/来源/成败 分组的回合数、帧数、时长
  * 节拍（cadence）：每回合实测 fps（帧时间戳差分）应接近标称 fps；偏离即提醒——
    数据集会“声称”标称 fps，实际低了训练/部署的时间对齐就会出错
  * 干预率：rollout 回合里人工接管帧占比（RECAP 数据健康度）
  * 图像：抽样回合的首/中/尾帧亮度统计，标记全黑/过曝/死帧
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from dobot_plug.config import load_config
from dobot_plug.episode_io import Episode, list_episodes


def episode_cadence(ep: Episode) -> dict:
    t = np.asarray(ep.lowdim["t"], dtype=np.float64)
    if len(t) < 2:
        return {"fps_measured": None, "dt_p99_s": None}
    dt = np.diff(t)
    dur = float(t[-1] - t[0])
    return {
        "fps_measured": round(float((len(t) - 1) / dur), 2) if dur > 0 else None,
        "dt_p99_s": round(float(np.quantile(dt, 0.99)), 4),
    }


def image_stats(ep: Episode, cam: str) -> dict:
    """首/中/尾帧亮度统计（不整集读图，快）。"""
    out = {"mean": [], "std": []}
    for i in (0, len(ep) // 2, len(ep) - 1):
        img = ep.image(cam, int(i)).astype(np.float32) / 255.0
        out["mean"].append(round(float(img.mean()), 3))
        out["std"].append(round(float(img.std()), 3))
    out["suspect"] = bool(min(out["mean"]) < 0.02 or max(out["mean"]) > 0.98 or max(out["std"]) < 0.01)
    return out


def build_report(raw_dir, eps: list[Episode], cfg: dict, sample: int = 4) -> dict:
    nominal_fps = float(cfg["task"]["fps"])
    groups: dict[tuple, list[Episode]] = {}
    for ep in eps:
        groups.setdefault((ep.meta["task_key"], ep.meta["source"], bool(ep.meta["success"])), []).append(ep)

    report: dict = {
        "raw_dir": str(raw_dir),
        "nominal_fps": nominal_fps,
        "total_episodes": len(eps),
        "total_frames": sum(len(e) for e in eps),
        "total_hours": round(sum(float(e.meta.get("duration_s", 0.0)) for e in eps) / 3600, 3),
        "groups": [],
        "episodes": [],
        "warnings": [],
    }
    for (task, source, success), group in sorted(groups.items()):
        report["groups"].append({
            "task": task, "source": source, "success": success,
            "episodes": len(group), "frames": sum(len(e) for e in group),
            "hours": round(sum(float(e.meta.get("duration_s", 0.0)) for e in group) / 3600, 3),
        })

    fps_all, iv_all = [], []
    for ep in eps:
        cad = episode_cadence(ep)
        iv = float(np.mean(ep.lowdim["intervention"])) if len(ep) else 0.0
        row = {
            "episode": str(ep.dir.relative_to(raw_dir)) if str(ep.dir).startswith(str(raw_dir)) else str(ep.dir),
            "task": ep.meta["task_key"], "source": ep.meta["source"],
            "success": bool(ep.meta["success"]), "frames": len(ep),
            "duration_s": ep.meta.get("duration_s"), **cad,
            "intervention_frac": round(iv, 3),
        }
        report["episodes"].append(row)
        if cad["fps_measured"]:
            fps_all.append(cad["fps_measured"])
        if ep.meta["source"] == "rollout":
            iv_all.append(iv)
        if cad["fps_measured"] is not None and cad["fps_measured"] < 0.9 * nominal_fps:
            report["warnings"].append(
                f"{row['episode']}: 实测 {cad['fps_measured']}Hz 明显低于标称 {nominal_fps}Hz")

    # 图像抽查（每任务 sample 个回合）
    if sample > 0:
        by_task: dict[str, list[Episode]] = {}
        for ep in eps:
            by_task.setdefault(ep.meta["task_key"], []).append(ep)
        rng = np.random.default_rng(0)
        report["image_check"] = {}
        for task, group in sorted(by_task.items()):
            picked = [group[i] for i in rng.choice(len(group), size=min(sample, len(group)), replace=False)]
            entries = []
            for ep in picked:
                cams = {}
                for cam in ep.meta["cameras"]:
                    st = image_stats(ep, cam)
                    cams[cam] = st
                    if st["suspect"]:
                        report["warnings"].append(
                            f"{ep.dir.name}/{cam}: 图像可疑（全黑/过曝/无内容）mean={st['mean']}")
                entries.append({"episode": ep.dir.name, "cameras": cams})
            report["image_check"][task] = entries

    if fps_all:
        report["fps_measured"] = {
            "min": round(min(fps_all), 2), "median": round(float(np.median(fps_all)), 2), "max": round(max(fps_all), 2)}
    if iv_all:
        report["rollout_intervention_frac"] = {
            "mean": round(float(np.mean(iv_all)), 3), "min": round(min(iv_all), 3), "max": round(max(iv_all), 3)}
    return report


def print_report(rep: dict) -> None:
    print("=" * 78)
    print(f"数据体检: {rep['raw_dir']}   标称 {rep['nominal_fps']}Hz")
    print(f"回合 {rep['total_episodes']}   帧数 {rep['total_frames']}   总时长 {rep['total_hours']}h")
    print("-" * 78)
    print(f"{'任务':<8}{'来源':<9}{'成败':<6}{'回合':>5}{'帧数':>9}{'小时':>8}")
    for g in rep["groups"]:
        print(f"{g['task']:<8}{g['source']:<9}{'✓' if g['success'] else '✗':<6}{g['episodes']:>5}{g['frames']:>9}{g['hours']:>8.3f}")
    if "fps_measured" in rep:
        f = rep["fps_measured"]
        print(f"\n实测帧率: min={f['min']} median={f['median']} max={f['max']} Hz（应接近 {rep['nominal_fps']}）")
    if "rollout_intervention_frac" in rep:
        iv = rep["rollout_intervention_frac"]
        print(f"rollout 干预率: mean={iv['mean']} min={iv['min']} max={iv['max']}")
    if rep["warnings"]:
        print("\n!! 警告:")
        for w in rep["warnings"]:
            print(f"  - {w}")
    else:
        print("\n无警告")


def save_montages(raw_dir, eps: list[Episode], out_dir: pathlib.Path, per_task: int = 2) -> None:
    from PIL import Image

    by_task: dict[str, list[Episode]] = {}
    for ep in eps:
        by_task.setdefault(ep.meta["task_key"], []).append(ep)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for task, group in sorted(by_task.items()):
        picked = [group[i] for i in rng.choice(len(group), size=min(per_task, len(group)), replace=False)]
        for ep in picked:
            for cam in ep.meta["cameras"]:
                idxs = np.linspace(0, len(ep) - 1, 6).astype(int)
                tiles = [ep.image(cam, int(i)) for i in idxs]
                h, w = tiles[0].shape[:2]
                grid = Image.new("RGB", (w * 3, h * 2))
                for k, tile in enumerate(tiles):
                    grid.paste(Image.fromarray(tile), ((k % 3) * w, (k // 3) * h))
                path = out_dir / f"montage_{task}_{ep.dir.name}_{cam}.jpg"
                grid.save(path, quality=88)
                print(f"拼图: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--sources", nargs="*", default=None, choices=["demo", "rollout"])
    ap.add_argument("--montage", action="store_true", help="另存抽样回合的相机拼图到 data/check/")
    ap.add_argument("--json", default=None, help="把报告另存为 JSON")
    ap.add_argument("--sample", type=int, default=4, help="每任务抽查图像的回合数")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = pathlib.Path(args.raw_dir or cfg["data"]["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = pathlib.Path(cfg["_path"]).resolve().parent.parent / raw_dir
    eps = list_episodes(raw_dir, args.tasks, args.sources)
    if not eps:
        raise SystemExit(f"{raw_dir} 下没有回合")
    rep = build_report(raw_dir, eps, cfg, sample=args.sample)
    print_report(rep)
    if args.montage:
        save_montages(raw_dir, eps, raw_dir.parent / "check")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=2))
        print(f"JSON 报告: {args.json}")


if __name__ == "__main__":
    main()
