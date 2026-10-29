"""CR5AF 驱动封装（基于越疆官方 V4 Python SDK: dobot_sdk）。

设计要点
--------
* 统一走关节空间 ServoJ 流式控制：
    - 策略输出 = 目标关节角，直接下发；
    - 遥操作在笛卡尔空间积分目标位姿，再用控制器自带的 InverseKin 解到关节角。
  这样“记录的动作”与“部署时下发的动作”是同一个量（目标关节角），接触时“往里推”的意图也被保留。
* 后台 ServoThread 以 ~30Hz 发送 ServoJ，在相邻两个控制周期(15Hz)的目标之间线性插值，
  并对每个周期的关节增量做速度限幅。
* 状态来自 30004 实时反馈端口（8ms），由 SDK 的反馈线程维护。
* 安全：力/力矩超限、TCP 超出工作空间、控制器报警 -> 进入 HOLD（保持当前位置，忽略新目标），
  需要人工 clear_hold() 才恢复。
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
import time
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

log = logging.getLogger(__name__)

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_response(resp: str) -> tuple[int, list[float]]:
    """解析 'ErrorID,{v1,v2,...},Cmd(...);' -> (error_id, [v...])。"""
    resp = resp.strip()
    err_str, _, rest = resp.partition(",")
    try:
        err = int(err_str)
    except ValueError:
        err = -1
    m = re.search(r"\{([^}]*)\}", rest)
    vals = [float(x) for x in _NUM_RE.findall(m.group(1))] if m else []
    return err, vals


@dataclasses.dataclass
class RobotState:
    t: float                      # 本机 time.time()
    q_deg: np.ndarray             # (6,) 实际关节角
    q_target_deg: np.ndarray      # (6,) 控制器内部目标关节角
    tcp_pose: np.ndarray          # (6,) x,y,z(mm), rx,ry,rz(deg)
    ft: np.ndarray                # (6,) Fx,Fy,Fz(N), Mx,My,Mz(Nm)
    mode: int
    error: bool


class SafetyHold(RuntimeError):
    pass


class CR5AF:
    def __init__(self, cfg: dict):
        from dobot_sdk import DobotRobot  # 延迟导入，方便无硬件时做离线测试

        self.cfg = cfg
        rc = cfg["robot"]
        self.ip = rc["ip"]
        self.euler_seq = rc.get("euler_seq", "xyz")
        self.robot = DobotRobot(self.ip)
        self._servo: ServoThread | None = None
        self.hold_reason: str | None = None

    # ------------------------------------------------------------------ 生命周期
    def connect(self, enable: bool = True):
        rc = self.cfg["robot"]
        self.robot.Connect()
        self.robot.StartFeedbackMonitor()
        ctrl = self.robot.robot_control
        log.info("RequestControl: %s", ctrl.RequestControl())
        ctrl.ClearError()
        if enable:
            cx, cy, cz = rc.get("payload_center_mm", [0, 0, 0])
            resp = ctrl.EnableRobot(load=float(rc["payload_kg"]), centerX=cx, centerY=cy, centerZ=cz, isCheck=0)
            log.info("EnableRobot: %s", resp)
            ctrl.SpeedFactor(int(rc.get("speed_factor", 30)))
        self._wait_feedback()
        if rc.get("zero_ft_on_start", True):
            self.zero_ft()

    def _wait_feedback(self, timeout: float = 3.0):
        t0 = time.time()
        while self.robot.GetStatus() is None:
            if time.time() - t0 > timeout:
                raise TimeoutError("30004 实时反馈端口无数据，检查网线/IP/防火墙")
            time.sleep(0.02)

    def zero_ft(self):
        """力传感器清零：必须在末端无接触、静止时调用。"""
        p = self.robot.plugins
        log.info("EnableFTSensor: %s", p.EnableFTSensor(1))
        time.sleep(0.2)
        log.info("SixForceHome: %s", p.SixForceHome())
        time.sleep(0.5)

    def close(self):
        self.stop_servo()
        try:
            self.robot.robot_control.Stop()
        except Exception:  # noqa: BLE001
            pass
        self.robot.StopFeedbackMonitor()
        self.robot.Disconnect()

    # ------------------------------------------------------------------ 状态
    def state(self) -> RobotState:
        s = self.robot.GetStatus()
        if s is None:
            raise RuntimeError("no feedback")
        tv = s.tool_vector_actual
        ft = s.actual_tcp_force if self.cfg["robot"].get("ft_source") != "six_force_value" else s.six_force_value
        return RobotState(
            t=time.time(),
            q_deg=np.asarray(s.joint_state.q_actual, dtype=np.float64),
            q_target_deg=np.asarray(s.joint_state.q_target, dtype=np.float64),
            tcp_pose=np.array([tv.x, tv.y, tv.z, tv.rx, tv.ry, tv.rz], dtype=np.float64),
            ft=np.asarray(ft, dtype=np.float64),
            mode=int(s.robot_mode),
            error=bool(s.error_status),
        )

    # ------------------------------------------------------------------ 运动学
    def inverse_kin(self, pose: Sequence[float], q_near_deg: Sequence[float]) -> np.ndarray | None:
        resp = self.robot.robot_control.InverseKin(list(map(float, pose)), use_joint_near=1, joint_near=list(map(float, q_near_deg)))
        err, vals = parse_response(resp)
        if err != 0 or len(vals) < 6:
            log.debug("InverseKin 失败: %s", resp)
            return None
        return np.asarray(vals[:6], dtype=np.float64)

    def forward_kin(self, q_deg: Sequence[float]) -> np.ndarray | None:
        err, vals = parse_response(self.robot.robot_control.PositiveKin(list(map(float, q_deg))))
        return np.asarray(vals[:6]) if err == 0 and len(vals) >= 6 else None

    def pose_to_rot(self, pose: np.ndarray) -> Rotation:
        return Rotation.from_euler(self.euler_seq, pose[3:6], degrees=True)

    def rot_to_euler(self, rot: Rotation) -> np.ndarray:
        return rot.as_euler(self.euler_seq, degrees=True)

    # ------------------------------------------------------------------ 点到点（回零位）
    def move_joints_blocking(self, q_deg: Sequence[float], timeout: float = 30.0):
        from dobot_sdk import CoordinateType

        was_running = self._servo is not None
        self.stop_servo()
        self.robot.motion.MovJ(list(map(float, q_deg)), coord_type=CoordinateType.JOINT)
        t0 = time.time()
        time.sleep(0.3)
        while time.time() - t0 < timeout:
            st = self.state()
            if np.max(np.abs(st.q_deg - np.asarray(q_deg))) < 0.2 and st.mode == 5:
                break
            if st.error:
                raise RuntimeError("回零位过程中控制器报警: " + self.robot.GetErrorFormatted())
            time.sleep(0.05)
        if was_running:
            self.start_servo()

    # ------------------------------------------------------------------ 流式伺服
    def start_servo(self):
        if self._servo is None:
            self._servo = ServoThread(self, self.cfg)
            self._servo.start()

    def stop_servo(self):
        if self._servo is not None:
            self._servo.stop()
            self._servo = None

    def set_target(self, q_deg: np.ndarray):
        """设置下一个控制周期的目标关节角（度）。HOLD 状态下被忽略。"""
        if self._servo is None:
            raise RuntimeError("先调用 start_servo()")
        if self.hold_reason is None:
            self._servo.set_goal(np.asarray(q_deg, dtype=np.float64))

    @property
    def commanded_q(self) -> np.ndarray:
        """伺服线程当前的目标（度）。"""
        return self._servo.goal.copy() if self._servo else self.state().q_deg

    # ------------------------------------------------------------------ 安全
    def check_safety(self, st: RobotState) -> str | None:
        rc = self.cfg["robot"]
        if st.error:
            return "控制器报警: " + self.robot.GetErrorFormatted()
        f = np.linalg.norm(st.ft[:3])
        m = np.linalg.norm(st.ft[3:])
        if f > rc["max_force_n"]:
            return f"合力 {f:.1f}N 超过 {rc['max_force_n']}N"
        if m > rc["max_torque_nm"]:
            return f"合力矩 {m:.2f}Nm 超过 {rc['max_torque_nm']}Nm"
        ws = rc["workspace_mm"]
        for i, ax in enumerate("xyz"):
            lo, hi = ws[ax]
            if not lo <= st.tcp_pose[i] <= hi:
                return f"TCP {ax}={st.tcp_pose[i]:.0f}mm 超出工作空间 [{lo},{hi}]"
        return None

    def enter_hold(self, reason: str):
        if self.hold_reason is None:
            log.warning("保护停: %s", reason)
            self.hold_reason = reason
            if self._servo is not None:
                # 目标设为当前实际位置，避免继续往里顶
                self._servo.reset(self.state().q_deg)

    def clear_hold(self):
        st = self.state()
        if st.error:
            self.robot.robot_control.ClearError()
            time.sleep(0.2)
            # 报警后通常需要重新使能
            rc = self.cfg["robot"]
            cx, cy, cz = rc.get("payload_center_mm", [0, 0, 0])
            self.robot.robot_control.EnableRobot(load=float(rc["payload_kg"]), centerX=cx, centerY=cy, centerZ=cz, isCheck=0)
            time.sleep(0.5)
        if self._servo is not None:
            self._servo.reset(self.state().q_deg)
        self.hold_reason = None


class ServoThread(threading.Thread):
    """以 servo_hz 发送 ServoJ，在控制周期之间线性插值，并做关节速度限幅。"""

    def __init__(self, arm: CR5AF, cfg: dict):
        super().__init__(daemon=True, name="ServoJ")
        rc = cfg["robot"]
        self.arm = arm
        self.hz = float(rc.get("servo_hz", 30))
        self.t = float(rc.get("servo_t", 0.1))
        self.aheadtime = float(rc.get("servo_aheadtime", 50))
        self.gain = float(rc.get("servo_gain", 500))
        self.max_step = float(rc.get("max_joint_vel_deg_s", 40)) / self.hz
        self.control_dt = 1.0 / float(cfg["task"]["fps"])
        self._lock = threading.Lock()
        self._running = True
        q0 = arm.state().q_deg
        self.cmd = q0.copy()       # 上一次实际发出的指令
        self.goal = q0.copy()      # 最新目标
        self._seg_start = q0.copy()
        self._seg_t0 = time.time()

    def set_goal(self, q: np.ndarray):
        with self._lock:
            self._seg_start = self.cmd.copy()
            self.goal = q.copy()
            self._seg_t0 = time.time()

    def reset(self, q: np.ndarray):
        with self._lock:
            self.cmd = q.copy()
            self.goal = q.copy()
            self._seg_start = q.copy()
            self._seg_t0 = time.time()

    def stop(self):
        self._running = False
        self.join(timeout=1.0)

    def run(self):
        period = 1.0 / self.hz
        next_t = time.time()
        while self._running:
            with self._lock:
                alpha = min(1.0, (time.time() - self._seg_t0) / self.control_dt)
                desired = self._seg_start + alpha * (self.goal - self._seg_start)
                step = np.clip(desired - self.cmd, -self.max_step, self.max_step)
                self.cmd = self.cmd + step
                cmd = self.cmd.copy()
            try:
                self.arm.robot.motion.ServoJ(cmd.tolist(), t=self.t, aheadtime=self.aheadtime, gain=self.gain)
            except Exception as e:  # noqa: BLE001
                log.error("ServoJ 发送失败: %s", e)
            next_t += period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()
