"""inspect_data 与 eval 的纯函数测试（无硬件）。"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dobot_plug.apps.eval import load_layouts, summarize  # noqa: E402
from dobot_plug.apps.inspect_data import build_report, episode_cadence, image_stats  # noqa: E402
from dobot_plug.config import load_config  # noqa: E402
from dobot_plug.episode_io import EpisodeWriter, list_episodes  # noqa: E402

CFG = load_config()


def _write_ep(d, task, source, success, n, fps_real, black=False, intervention=0.0):
    w = EpisodeWriter(d, task, f"{task} the plug", 15, ["base"], source=source)
    t0 = time.time()
    iv = int(intervention * n)
    for i in range(n):
        if black:
            img = np.zeros((24, 32, 3), np.uint8)
        else:
            img = (np.full((24, 32, 3), 30 + 5 * i, np.uint8)
                   + np.random.default_rng(i).integers(0, 40, (24, 32, 3))).astype(np.uint8)
        w.add({"base": img}, t=t0 + i / fps_real, q_deg=np.zeros(6), q_target_deg=np.zeros(6),
              tcp_pose=np.zeros(6), ft=np.zeros(6), gripper_pos=0.0,
              action_q_deg=np.zeros(6), action_gripper=0.0, intervention=1 if i < iv else 0, hold=0)
    return w.save(success=success)


def test_inspect_report():
    with tempfile.TemporaryDirectory() as d:
        _write_ep(d, "insert", "demo", True, 30, 15.0)
        _write_ep(d, "insert", "demo", False, 30, 10.0)          # 实测 10Hz -> 警告
        _write_ep(d, "insert", "rollout", True, 30, 15.0, intervention=0.3)
        _write_ep(d, "insert", "demo", True, 30, 15.0, black=True)  # 全黑图 -> 警告
        eps = list_episodes(d)
        rep = build_report(pathlib.Path(d), eps, CFG, sample=4)
        assert rep["total_episodes"] == 4 and rep["total_frames"] == 120
        assert len(rep["groups"]) == 3
        assert any("10.0Hz" in w or "10.0 Hz" in w for w in rep["warnings"]), rep["warnings"]
        assert any("图像可疑" in w for w in rep["warnings"])
        assert rep["rollout_intervention_frac"]["mean"] == 0.3
        ep0 = eps[0]
        cad = episode_cadence(ep0)
        assert abs(cad["fps_measured"] - 15.0) < 0.2
        assert not image_stats(ep0, "base")["suspect"]


def test_eval_summarize_and_layouts():
    rows = [
        {"task": "insert", "success": True, "frames": 30},
        {"task": "insert", "success": True, "frames": 40},
        {"task": "insert", "success": False, "frames": 100},
        {"task": "insert", "success": None, "frames": 10},   # 无效
        {"task": "insert", "success": None, "frames": 5},
    ]
    s = summarize(rows)
    t = s["tasks"]["insert"]
    assert t["success"] == 2 and t["valid"] == 3 and t["invalid"] == 2
    lo, hi = t["wilson95"]
    assert 0.1 < lo < 2 / 3 < hi < 0.98
    assert s["overall"]["valid"] == 3

    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "layouts.yaml"
        p.write_text("insert: ['居中', '左移']\n")
        lay = load_layouts(str(p), ["insert"], 5)
        assert lay["insert"] == ["居中", "左移", "居中", "左移", "居中"]
        lay2 = load_layouts(None, ["insert"], 3)
        assert lay2["insert"][0] == "布局 1/3" and len(lay2["insert"]) == 3


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                import traceback

                traceback.print_exc()
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
