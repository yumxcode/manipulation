"""本地假控制器：模拟 CR5AF 的 29999（指令）和 30004（实时反馈）端口，用于无硬件联调。

运动学是假的：pose = 10 * q（仅用于验证通信与控制链路，不代表真实 CR5AF）。
"""

from __future__ import annotations

import re
import socket
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/root/TCP-IP-Python-V4")
from dobot_sdk.protocol.data_types import TEST_VALUE_MAGIC, MyType  # noqa: E402

NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


class FakeDobot:
    def __init__(self, host="127.0.0.1", dash_port=29999, fb_port=30004):
        self.q = np.array([10.0, -20.0, 30.0, 0.0, -40.0, 5.0])
        self.mode = 5
        self.ft = np.zeros(6)
        self.log: list[str] = []
        self.servo_count = 0
        self._running = True
        self.socks = []
        for port, fn in ((dash_port, self._dash_client), (fb_port, self._fb_client)):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(4)
            self.socks.append(s)
            threading.Thread(target=self._accept, args=(s, fn), daemon=True).start()

    def _accept(self, s, fn):
        while self._running:
            try:
                c, _ = s.accept()
            except OSError:
                return
            threading.Thread(target=fn, args=(c,), daemon=True).start()

    def _reply(self, cmd: str) -> str:
        name = cmd.split("(")[0]
        args = [float(x) for x in NUM.findall(cmd[len(name):])]
        self.log.append(name)
        if name == "ServoJ":
            self.q = np.array(args[:6])
            self.servo_count += 1
            return f"0,{{{self.servo_count}}},{cmd};"
        if name == "MovJ":
            self.q = np.array(args[:6])
            return f"0,{{1}},{cmd};"
        if name == "InverseKin":
            q = np.array(args[:6]) * 0.1
            return "0,{" + ",".join(f"{v:.6f}" for v in q) + "}," + cmd + ";"
        if name == "PositiveKin":
            p = np.array(args[:6]) * 10
            return "0,{" + ",".join(f"{v:.6f}" for v in p) + "}," + cmd + ";"
        return f"0,{{}},{cmd};"

    def _dash_client(self, c):
        buf = b""
        while self._running:
            try:
                data = c.recv(4096)
            except OSError:
                return
            if not data:
                return
            buf += data
            while b")" in buf:
                i = buf.index(b")") + 1
                cmd, buf = buf[:i].decode().strip(), buf[i:]
                c.sendall(self._reply(cmd).encode())

    def _fb_client(self, c):
        while self._running:
            pkt = np.zeros(1, dtype=MyType)
            pkt["len"] = 1440
            pkt["TestValue"] = TEST_VALUE_MAGIC
            pkt["RobotMode"] = self.mode
            pkt["QActual"][0] = self.q
            pkt["QTarget"][0] = self.q
            pkt["ToolVectorActual"][0] = self.q * 10
            pkt["ActualTCPForce"][0] = self.ft
            try:
                c.sendall(pkt.tobytes())
            except OSError:
                return
            time.sleep(0.008)

    def close(self):
        self._running = False
        for s in self.socks:
            s.close()
