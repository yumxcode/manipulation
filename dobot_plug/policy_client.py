"""openpi 策略服务器的客户端封装：异步推理 + 动作块执行。

AsyncPolicy: 后台线程做推理——执行当前动作块的同时请求下一个块，机械臂不停顿。
ActionChunkExecutor: 按控制步号对齐消费动作块，带策略目标偏离限幅。

run_policy.py（带人工接管的部署/采数据）与 eval.py（正式评估）共用这里，
保证两条链路的动作执行语义完全一致。
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .features import action_to_command

log = logging.getLogger(__name__)


class AsyncPolicy:
    """后台线程做推理：执行当前动作块的同时请求下一个块，避免机械臂停顿。

    每个请求记录发出时的控制步号 tick；动作块的第 k 个动作对应 tick+k，
    取用时按当前步号对齐（跳过推理延迟期间已经过去的动作）。
    """

    def __init__(self, host: str, port: int, resize: int = 224):
        from openpi_client import image_tools, websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.image_tools = image_tools
        self.resize = resize
        self._lock = threading.Lock()
        self._req = None
        self._busy = False
        self.chunk: np.ndarray | None = None
        self.chunk_tick = -1
        self.generation = 0
        self.latency = 0.0
        self._cv = threading.Condition(self._lock)
        threading.Thread(target=self._loop, daemon=True, name="policy").start()
        log.info("策略服务器 metadata: %s", self.client.get_server_metadata())

    def preprocess(self, images_by_slot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        it = self.image_tools
        return {k: it.convert_to_uint8(it.resize_with_pad(v, self.resize, self.resize)) for k, v in images_by_slot.items()}

    def request(self, obs: dict, tick: int):
        with self._cv:
            if self._busy or self._req is not None:
                return False
            self._req = (obs, tick, self.generation)
            self._cv.notify()
            return True

    def invalidate(self):
        with self._lock:
            self.generation += 1
            self.chunk = None
            self.chunk_tick = -1

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy or self._req is not None

    def _loop(self):
        while True:
            with self._cv:
                while self._req is None:
                    self._cv.wait()
                obs, tick, gen = self._req
                self._req = None
                self._busy = True
            t0 = time.time()
            try:
                actions = np.asarray(self.client.infer(obs)["actions"])
            except Exception as e:  # noqa: BLE001
                log.error("推理失败: %s", e)
                actions = None
            with self._lock:
                self._busy = False
                self.latency = time.time() - t0
                if actions is not None and gen == self.generation:
                    self.chunk, self.chunk_tick = actions, tick

    def action_at(self, tick: int) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self.chunk is None:
                return None, -1
            idx = tick - self.chunk_tick
            if idx >= len(self.chunk):
                return None, idx
            return self.chunk[idx], idx


class ActionChunkExecutor:
    """消费 AsyncPolicy 的动作块：
    - maybe_request(): 动作块耗尽或已执行 replan_steps 步且推理空闲时，用当前观测请求下一块
    - command(): 取当前步的动作 -> (目标关节角 deg, 夹爪 0~1, 块内序号)；目标偏离限幅
    """

    def __init__(self, policy: AsyncPolicy, max_joint_dev_deg: float):
        self.policy = policy
        self.max_dev_deg = float(max_joint_dev_deg)

    def maybe_request(self, make_obs, tick: int, replan_steps: int):
        action, idx = self.policy.action_at(tick)
        if (action is None or idx >= replan_steps) and not self.policy.busy:
            self.policy.request(make_obs(), tick)

    def command(self, tick: int, q_now: np.ndarray) -> tuple[np.ndarray, float, int] | None:
        action, idx = self.policy.action_at(tick)
        if action is None:
            return None
        q_cmd, g_cmd = action_to_command(action)
        dev = q_cmd - q_now
        if np.max(np.abs(dev)) > self.max_dev_deg:
            log.warning("策略目标偏离当前位置 %.1f°，已限幅", np.max(np.abs(dev)))
            q_cmd = q_now + np.clip(dev, -self.max_dev_deg, self.max_dev_deg)
        return q_cmd, g_cmd, idx
