"""无硬件离线测试：python tests/test_offline.py

覆盖：协议解析、state/action 构造、原始回合读写、伺服插值与限幅、遥操作积分、
RECAP 回报/优势/two-hot、openpi 侧 DobotInputs/AdvantagePrompt（用桩模块）。
"""

from __future__ import annotations

import enum
import pathlib
import sys
import tempfile
import time
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dobot_plug.config import load_config  # noqa: E402
from dobot_plug.episode_io import Episode, EpisodeWriter, list_episodes  # noqa: E402
from dobot_plug.features import action_to_command, build_action, build_state, policy_observation  # noqa: E402
from dobot_plug.recap import Bins, n_step_advantage, normalized_returns  # noqa: E402
from dobot_plug.robot import CR5AF, RobotState, ServoThread, parse_response  # noqa: E402

CFG = load_config()


def test_parse_response():
    assert parse_response("0,{1.5,-2,3e-1,4,5,6},InverseKin(...);") == (0, [1.5, -2.0, 0.3, 4.0, 5.0, 6.0])
    assert parse_response("-1,{},InverseKin();")[0] == -1
    assert parse_response("0,{},EnableRobot();") == (0, [])


def test_features():
    q = np.array([10, -20, 90, 0, -90, 45.0])
    s = build_state(q, 0.3, np.arange(6), include_ft=True)
    assert s.shape == (13,) and s.dtype == np.float32
    assert np.allclose(s[:6], np.deg2rad(q)) and np.isclose(s[6], 0.3)
    assert build_state(q, 0.3, np.arange(6), include_ft=False).shape == (7,)
    S = build_state(np.tile(q, (5, 1)), np.zeros(5), np.zeros((5, 6)), True)
    assert S.shape == (5, 13)
    a = build_action(q, 1.0)
    qd, g = action_to_command(a)
    assert np.allclose(qd, q, atol=1e-4) and g == 1.0
    A = build_action(np.tile(q, (5, 1)), np.ones(5))
    assert A.shape == (5, 7)
    obs = policy_observation({"base_0_rgb": np.zeros((4, 4, 3), np.uint8)}, s, "p")
    assert set(obs) == {"observation/base_0_rgb", "observation/state", "prompt"}


def test_episode_io():
    with tempfile.TemporaryDirectory() as d:
        w = EpisodeWriter(d, "insert", "insert the plug", 15, ["base", "wrist"], source="rollout")
        for i in range(12):
            imgs = {c: np.full((24, 32, 3), i * 10, np.uint8) for c in ("base", "wrist")}
            w.add(imgs, t=time.time(), q_deg=np.ones(6) * i, q_target_deg=np.ones(6), tcp_pose=np.zeros(6),
                  ft=np.zeros(6), gripper_pos=0.0, action_q_deg=np.ones(6) * (i + 1), action_gripper=float(i > 5),
                  intervention=int(i > 8), hold=0)
        path = w.save(success=True)
        ep = Episode(path)
        assert len(ep) == 12 and ep.success and ep.meta["source"] == "rollout"
        assert ep.lowdim["q_deg"].shape == (12, 6) and ep.lowdim["intervention"].sum() == 3
        assert abs(int(ep.image("wrist", 7).mean()) - 70) <= 2
        assert len(list_episodes(d, ["insert"], ["rollout"])) == 1
        assert len(list_episodes(d, ["unplug"])) == 0


def test_convert_keep_mask():
    sys.path.insert(0, str(ROOT / "scripts"))
    from convert_to_lerobot import keep_mask

    n = 20
    aq = np.zeros((n, 6))
    aq[5:15, 0] = np.arange(10) * 0.5      # 第 5~14 帧在动
    aq[15:, 0] = aq[14, 0]
    hold = np.zeros(n)
    hold[10] = 1
    class _E:
        lowdim = {"action_q_deg": aq, "action_gripper": np.zeros(n), "hold": hold}

        def __len__(self):
            return n
    k = keep_mask(_E(), drop_hold=True, trim_idle=True)
    kept = np.flatnonzero(k)
    assert kept[0] == 4 and kept[-1] == 16 and 10 not in kept, kept
    assert keep_mask(_E(), drop_hold=False, trim_idle=False).all()


class _FakeMotion:
    def __init__(self):
        self.sent = []

    def ServoJ(self, joints, t, aheadtime, gain):
        self.sent.append(np.array(joints))
        return "0,{1},ServoJ();"


class _FakeArm:
    def __init__(self):
        self.robot = types.SimpleNamespace(motion=_FakeMotion())
        self.euler_seq = "xyz"
        self._pose = np.array([400, 0, 300, 180, 0, 0.0])
        self._q = self._pose * 0.1

    def state(self):
        return RobotState(time.time(), self._q.copy(), self._q.copy(), self._pose.copy(), np.zeros(6), 5, False)

    # 遥操作用：假设关节空间 = 位姿空间的线性映射（只测积分逻辑）
    def inverse_kin(self, pose, q_near):
        return np.asarray(pose[:6]) * 0.1

    def forward_kin(self, q):
        return np.asarray(q) * 10

    pose_to_rot = CR5AF.pose_to_rot
    rot_to_euler = CR5AF.rot_to_euler


def test_servo_interp_and_limit():
    cfg = dict(CFG)
    arm = _FakeArm()
    th = ServoThread(arm, cfg)
    th.start()
    th.set_goal(arm._q + np.array([30.0, 0, 0, 0, 0, 0]))  # 远超单周期限幅
    time.sleep(0.5)
    th.stop()
    sent = np.array(arm.robot.motion.sent)
    steps = np.abs(np.diff(sent[:, 0]))
    max_step = cfg["robot"]["max_joint_vel_deg_s"] / cfg["robot"]["servo_hz"]
    assert len(sent) >= 10, len(sent)
    assert steps.max() <= max_step + 1e-9, steps.max()
    assert sent[-1, 0] - arm._q[0] > 5.0  # 确实在朝目标走


def test_teleop_integration():
    from dobot_plug.teleop import CartesianTeleop

    class _Mouse:
        def read(self):
            return np.array([1.0, 0, 0, 0, 0, 0]), [0, 0]

    arm = _FakeArm()
    tel = CartesianTeleop(CFG, arm, mouse=_Mouse())
    tel.sync_to_robot()
    p0 = tel.target_pose.copy()
    for _ in range(15):  # 1 秒
        tel.step()
    dx = tel.target_pose[0] - p0[0]
    assert abs(dx - CFG["teleop"]["max_lin_vel_mm_s"]) < 1e-6, dx
    assert np.allclose(tel.target_pose[3:], p0[3:])
    # 旋转：绕工具 z 轴 1 秒，欧拉角往返一致
    class _MouseR:
        def read(self):
            return np.array([0, 0, 0, 0, 0, 1.0]), [0, 0]
    tel.mouse = _MouseR()
    r0 = arm.pose_to_rot(tel.target_pose)
    for _ in range(15):
        tel.step()
    rel = (r0.inv() * arm.pose_to_rot(tel.target_pose)).as_rotvec(degrees=True)
    assert np.allclose(rel, [0, 0, CFG["teleop"]["max_rot_vel_deg_s"]], atol=1e-3), rel
    # 夹爪按钮上升沿切换
    class _MouseB:
        def __init__(self):
            self.k = 0
        def read(self):
            self.k += 1
            return np.zeros(6), [1 if self.k in (2, 3) else 0, 0]
    tel.mouse = _MouseB()
    gs = [tel.step()[1] for _ in range(5)]
    assert gs == [0.0, 1.0, 1.0, 1.0, 1.0], gs


def test_recap_math():
    l_max = int(CFG["task"]["max_episode_s"] * CFG["task"]["fps"])
    g_s = normalized_returns(100, True, CFG)
    g_f = normalized_returns(100, False, CFG)
    assert g_s[-1] == 0.0 and np.all(np.diff(g_s) > 0)
    assert np.all(g_f < g_s) and g_f.min() >= -1.0
    assert np.isclose(g_s[0], -99 / (2 * l_max))
    # 价值完美时，优势恒为 0
    assert np.allclose(n_step_advantage(g_s, g_s, 10), 0, atol=1e-6)
    # 价值偏悲观 -> 优势为正
    assert np.all(n_step_advantage(g_s, g_s - 0.1, 10)[-10:] > 0)
    b = Bins(201)
    th = b.two_hot(np.array([-1.0, -0.5, -0.123, 0.0]))
    assert np.allclose(th.sum(1), 1) and np.allclose(b.expectation(th), [-1.0, -0.5, -0.123, 0.0], atol=1e-6)


def test_openpi_transforms_with_stubs():
    """openpi 全量环境装不了时，用桩模块验证 dobot_policy 的逻辑。"""
    policy_file = pathlib.Path.home() / "openpi/src/openpi/policies/dobot_policy.py"
    if not policy_file.exists():
        print("  (跳过：未找到 openpi 仓库)")
        return

    class ModelType(enum.Enum):
        PI0 = "pi0"
        PI0_FAST = "pi0_fast"
        PI05 = "pi05"

    openpi = types.ModuleType("openpi")
    tr = types.ModuleType("openpi.transforms")
    tr.DataTransformFn = object
    models = types.ModuleType("openpi.models")
    mm = types.ModuleType("openpi.models.model")
    mm.ModelType = ModelType
    openpi.transforms, openpi.models, models.model = tr, models, mm
    einops = types.ModuleType("einops")
    einops.rearrange = lambda x, pat: np.transpose(x, (1, 2, 0))
    sys.modules.update({"openpi": openpi, "openpi.transforms": tr, "openpi.models": models,
                        "openpi.models.model": mm, "einops": einops})
    ns: dict = {}
    exec(compile(policy_file.read_text(), str(policy_file), "exec"), ns)

    ex = ns["make_dobot_example"]()
    del ex["observation/right_wrist_0_rgb"]
    out = ns["DobotInputs"](model_type=ModelType.PI05)(ex)
    assert set(out["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert out["image_mask"]["right_wrist_0_rgb"] == np.False_ and out["image_mask"]["base_0_rgb"] == np.True_
    assert out["image"]["right_wrist_0_rgb"].shape == (240, 320, 3)
    chw = dict(ns["make_dobot_example"]())
    chw["observation/base_0_rgb"] = np.random.rand(3, 240, 320).astype(np.float32)
    assert ns["DobotInputs"](model_type=ModelType.PI05)(chw)["image"]["base_0_rgb"].dtype == np.uint8

    ap = ns["AdvantagePrompt"](dropout=0.0)
    r = ap({"prompt": "insert", "advantage": np.array([1.0], np.float32), "state": 0})
    assert r["prompt"] == "insert\nAdvantage: positive" and "advantage" not in r
    assert ap({"prompt": "x", "advantage": np.array([0.0])})["prompt"].endswith("negative")
    assert ap({"prompt": "x"})["prompt"] == "x"  # 推理：无 advantage 字段 -> 不变
    ap_d = ns["AdvantagePrompt"](dropout=1.0)
    assert ap_d({"prompt": "x", "advantage": np.array([1.0])})["prompt"] == "x"
    # 与客户端后缀一致
    from dobot_plug.features import RECAP_SUFFIX_NEG, RECAP_SUFFIX_POS
    assert ns["RECAP_POS"] == RECAP_SUFFIX_POS and ns["RECAP_NEG"] == RECAP_SUFFIX_NEG

    act = np.random.rand(15, 32)
    assert ns["DobotOutputs"]()({"actions": act})["actions"].shape == (15, 7)
    for k in list(sys.modules):
        if k == "einops" or k == "openpi" or k.startswith("openpi."):
            del sys.modules[k]


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
