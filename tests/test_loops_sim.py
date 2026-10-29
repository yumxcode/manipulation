"""端到端跑 record.py 与 run_policy.py 的主循环：假控制器 + 假相机 + 假 3D 鼠标 + 脚本化按键 + 假策略服务器。

    python tests/test_loops_sim.py
"""

from __future__ import annotations

import pathlib
import queue
import sys
import tempfile
import threading
import time
import types

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/root/TCP-IP-Python-V4")

from fake_dobot import FakeDobot  # noqa: E402

import dobot_plug.env as env_mod  # noqa: E402
import dobot_plug.teleop as teleop_mod  # noqa: E402


class FakeRig:
    def __init__(self, cfg):
        self.cams = {n: None for n in cfg["cameras"]["devices"]}
        self.wh = (cfg["cameras"]["save"]["width"], cfg["cameras"]["save"]["height"])

    def start(self): ...
    def stop(self): ...

    def get(self):
        w, h = self.wh
        return {n: np.random.randint(0, 255, (h, w, 3), np.uint8) for n in self.cams}


class FakeMouse:
    axes = np.zeros(6)

    def read(self):
        return FakeMouse.axes.copy(), [0, 0]


class ScriptedKeys:
    """[(延迟秒, 按键, 动作回调或 None), ...]"""

    script: list = []

    def __init__(self):
        self.q = queue.Queue()

        def run():
            for delay, key, fn in ScriptedKeys.script:
                time.sleep(delay)
                if fn:
                    fn()
                if key:
                    self.q.put(key)

        threading.Thread(target=run, daemon=True).start()

    def poll(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def install_fakes():
    env_mod.CameraRig = FakeRig
    teleop_mod.SpaceMouse = FakeMouse


def make_cfg(tmp: pathlib.Path) -> pathlib.Path:
    cfg = yaml.safe_load((ROOT / "configs/hardware.yaml").read_text())
    cfg["robot"]["ip"] = "127.0.0.1"
    cfg["robot"]["workspace_mm"] = {"x": [-1e4, 1e4], "y": [-1e4, 1e4], "z": [-1e4, 1e4]}
    cfg["robot"]["home_joints_deg"] = [10.0, -20.0, 30.0, 0.0, -40.0, 5.0]
    cfg["gripper"]["type"] = "none"
    cfg["data"]["raw_dir"] = str(tmp / "raw")
    p = tmp / "hw.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return p


def test_record(cfg_path, raw):
    import record

    record.KeyReader = ScriptedKeys
    set_ax = lambda v: (lambda: setattr(FakeMouse, "axes", np.array(v, float)))  # noqa: E731
    ScriptedKeys.script = [
        (0.5, " ", set_ax([0, 0, 1, 0, 0, 0])),   # 开始录制 + 向上推
        (1.0, None, set_ax([1, 0, 0, 0, 0, 0])),
        (0.6, "s", set_ax([0, 0, 0, 0, 0, 0])),   # 成功
        (0.3, "t", None),                          # 切到 unplug
        (0.2, " ", set_ax([0, 1, 0, 0, 0, 0])),
        (0.8, "f", set_ax([0] * 6)),              # 失败
        (0.3, "q", None),
    ]
    sys.argv = ["record.py", "--config", str(cfg_path), "--task", "insert"]
    record.main()
    from dobot_plug.episode_io import list_episodes

    eps = list_episodes(raw)
    assert len(eps) == 2, eps
    ins = [e for e in eps if e.meta["task_key"] == "insert"][0]
    unp = [e for e in eps if e.meta["task_key"] == "unplug"][0]
    assert ins.success and not unp.success
    assert 20 <= len(ins) <= 30, len(ins)
    ld = ins.lowdim
    # 动作 = 下发目标；下一帧实际关节角应接近本帧动作（假控制器即时跟随，伺服插值有 1 帧延迟）
    err = np.abs(ld["action_q_deg"][:-3] - ld["q_deg"][2:-1]).max()
    assert err < 0.5, err
    tcp = ld["tcp_pose"]
    assert tcp[-1, 2] - tcp[0, 2] > 40 and tcp[-1, 0] - tcp[0, 0] > 20, tcp[[0, -1], :3]
    assert ld["intervention"].all()
    print(f"PASS record.py：2 回合，insert {len(ins)} 帧，TCP 位移 {np.round(tcp[-1, :3] - tcp[0, :3], 1).tolist()}")


def install_fake_openpi_client(fake_ctl):
    calls = []

    class Client:
        def __init__(self, host, port):
            pass

        def get_server_metadata(self):
            return {"fake": True}

        def infer(self, obs):
            time.sleep(0.12)  # 模拟推理延迟
            calls.append(obs)
            state = obs["observation/state"]
            q = state[:6].astype(np.float64)
            # 每步 J1 +0.5°，夹爪闭合
            acts = np.zeros((15, 7))
            for k in range(15):
                acts[k, :6] = q + np.deg2rad([0.5 * (k + 1), 0, 0, 0, 0, 0])
                acts[k, 6] = 1.0
            return {"actions": acts}

    it = types.SimpleNamespace(
        resize_with_pad=lambda img, h, w: img[:h, :w],
        convert_to_uint8=lambda img: img.astype(np.uint8),
    )
    mod = types.ModuleType("openpi_client")
    mod.image_tools = it
    mod.websocket_client_policy = types.SimpleNamespace(WebsocketClientPolicy=Client)
    sys.modules["openpi_client"] = mod
    return calls


def test_run_policy(cfg_path, raw, fake_ctl):
    calls = install_fake_openpi_client(fake_ctl)
    import run_policy

    run_policy.KeyReader = ScriptedKeys
    j1 = {}
    set_ax = lambda v: (lambda: setattr(FakeMouse, "axes", np.array(v, float)))  # noqa: E731
    ScriptedKeys.script = [
        (0.3, " ", lambda: j1.setdefault("start", fake_ctl.q[0])),       # 策略执行
        (2.0, None, lambda: j1.setdefault("after_policy", fake_ctl.q[0])),
        (0.0, None, set_ax([0, 0, 1, 0, 0, 0])),                         # 推鼠标 -> 自动接管
        (1.0, None, set_ax([0] * 6)),
        (0.2, "i", None),                                                 # 交还策略
        (1.0, "s", None),
        (0.3, "q", None),
    ]
    sys.argv = ["run_policy.py", "--config", str(cfg_path), "--task", "insert", "--recap", "--policy-tag", "sim"]
    before = {p.parent for p in pathlib.Path(raw).glob("*/*/meta.json")}
    run_policy.main()
    from dobot_plug.episode_io import Episode

    new = [p.parent for p in pathlib.Path(raw).glob("*/*/meta.json") if p.parent not in before]
    assert len(new) == 1
    ep = Episode(new[0])
    ld = ep.lowdim
    iv = ld["intervention"]
    assert ep.meta["source"] == "rollout" and ep.success
    assert iv.sum() > 5 and (1 - iv).sum() > 20, (iv.sum(), len(iv))
    # 自主段：J1 应持续增加（~0.5°/步）
    moved = j1["after_policy"] - j1["start"]
    assert 8 < moved < 25, moved
    # 接管段：TCP 被推高
    first_h = int(np.argmax(iv > 0))
    assert ld["tcp_pose"][first_h + 10, 2] > ld["tcp_pose"][first_h, 2] + 5
    assert calls and calls[0]["prompt"].endswith("\nAdvantage: positive")
    assert calls[0]["observation/state"].shape == (13,)
    assert set(k for k in calls[0] if k.startswith("observation/")) == {
        "observation/base_0_rgb", "observation/left_wrist_0_rgb", "observation/right_wrist_0_rgb", "observation/state"}
    print(f"PASS run_policy.py：{len(ep)} 帧（人工 {int(iv.sum())} 帧），策略 2 秒内 J1 +{moved:.1f}°，{len(calls)} 次推理")
    return new[0]


if __name__ == "__main__":
    install_fakes()
    fake = FakeDobot()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            cfgp = make_cfg(tmp)
            test_record(cfgp, tmp / "raw")
            test_run_policy(cfgp, tmp / "raw", fake)
    finally:
        fake.close()
