"""夹爪抽象。统一约定：position / command 取值 0=全开，1=全闭。

* DHModbusGripper : 大寰 AG/PGE 系列，Modbus-RTU，电脑通过 USB-RS485 直连夹爪
* ToolDOGripper   : 末端数字输出控制的开关型夹爪（气动等），无位置反馈，状态=最近一次指令
* NoGripper       : 调试用

如果你的夹爪接在 CR5AF 末端航插的 485 上、由控制器转发，需要改用控制器 Modbus 指令
（ModbusRTUCreate / SetHoldRegs），按同样的接口实现一个类即可。
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


class Gripper:
    def start(self): ...
    def command(self, close_frac: float): ...
    def position(self) -> float: return 0.0
    def stop(self): ...


class NoGripper(Gripper):
    def __init__(self):
        self._cmd = 0.0

    def command(self, close_frac: float):
        self._cmd = float(close_frac)

    def position(self) -> float:
        return self._cmd


class ToolDOGripper(Gripper):
    def __init__(self, arm, index: int = 1, close_level: int = 1):
        self.arm, self.index, self.close_level = arm, index, close_level
        self._cmd = None

    def command(self, close_frac: float):
        close = close_frac >= 0.5
        if close != self._cmd:
            level = self.close_level if close else 1 - self.close_level
            self.arm.robot.io.ToolDOInstant(self.index, level)
            self._cmd = close

    def position(self) -> float:
        return 1.0 if self._cmd else 0.0


class DHModbusGripper(Gripper):
    """大寰夹爪寄存器（AG-95/AG-160/PGE/PGC 通用）:
    0x0100 初始化  0x0101 力(20-100)  0x0103 位置(0-1000, 1000=全开)  0x0104 速度(1-100)
    0x0200 初始化状态  0x0201 夹持状态  0x0202 实际位置
    """

    def __init__(self, port: str, baudrate: int = 115200, slave_id: int = 1, force_pct: int = 60, speed_pct: int = 60):
        import minimalmodbus

        self.inst = minimalmodbus.Instrument(port, slave_id)
        self.inst.serial.baudrate = baudrate
        self.inst.serial.timeout = 0.05
        self.inst.clear_buffers_before_each_transaction = True
        self.force_pct, self.speed_pct = force_pct, speed_pct
        self._io = threading.Lock()
        self._pos = 0.0
        self._target = None
        self._running = False
        self._thread: threading.Thread | None = None

    def _write(self, reg: int, val: int):
        with self._io:
            self.inst.write_register(reg, int(val), functioncode=6)

    def _read(self, reg: int) -> int:
        with self._io:
            return int(self.inst.read_register(reg, functioncode=3))

    def start(self):
        if self._read(0x0200) != 1:
            log.info("夹爪初始化中...")
            self._write(0x0100, 1)
            t0 = time.time()
            while self._read(0x0200) != 1:
                if time.time() - t0 > 10:
                    raise TimeoutError("大寰夹爪初始化超时")
                time.sleep(0.1)
        self._write(0x0101, self.force_pct)
        self._write(0x0104, self.speed_pct)
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True, name="gripper")
        self._thread.start()

    def _poll(self):
        while self._running:
            try:
                raw = self._read(0x0202)
                self._pos = 1.0 - raw / 1000.0
            except Exception as e:  # noqa: BLE001
                log.debug("夹爪读位置失败: %s", e)
            time.sleep(0.05)

    def command(self, close_frac: float):
        target = int(round((1.0 - min(max(close_frac, 0.0), 1.0)) * 1000))
        # 只有变化足够大时才下发，避免串口阻塞
        if self._target is None or abs(target - self._target) >= 20:
            try:
                self._write(0x0103, target)
                self._target = target
            except Exception as e:  # noqa: BLE001
                log.warning("夹爪写指令失败: %s", e)

    def position(self) -> float:
        return self._pos

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)


def make_gripper(cfg: dict, arm=None) -> Gripper:
    gc = cfg["gripper"]
    t = gc["type"]
    if t == "dh_modbus":
        return DHModbusGripper(gc["port"], gc["baudrate"], gc["slave_id"], gc["force_pct"], gc["speed_pct"])
    if t == "tool_do":
        return ToolDOGripper(arm, gc["tool_do_index"], gc["tool_do_close_level"])
    if t == "none":
        return NoGripper()
    raise ValueError(f"未知夹爪类型 {t}")
