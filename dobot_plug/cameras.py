"""多台 RealSense D435i 采集：每台一个线程，只保留最新一帧（RGB）。

VLA 只用 RGB，深度流默认关闭以节省 USB 带宽。多台相机请尽量分到不同的 USB3 控制器上。
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def list_realsense() -> list[dict]:
    import pyrealsense2 as rs

    out = []
    for dev in rs.context().query_devices():
        out.append(
            {
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
                "usb": dev.get_info(rs.camera_info.usb_type_descriptor)
                if dev.supports(rs.camera_info.usb_type_descriptor)
                else "?",
            }
        )
    return out


class RealSenseCamera(threading.Thread):
    def __init__(self, name: str, serial: str, width: int, height: int, fps: int):
        super().__init__(daemon=True, name=f"cam-{name}")
        import pyrealsense2 as rs

        self.name = name
        self.serial = serial
        self._rs = rs
        self.pipeline = rs.pipeline()
        conf = rs.config()
        conf.enable_device(serial)
        conf.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self._conf = conf
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._running = True

    def start(self):
        self.pipeline.start(self._conf)
        super().start()

    def run(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError as e:
                log.warning("[%s] 取帧超时: %s", self.name, e)
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data()).copy()
            with self._lock:
                self._frame = img
                self._stamp = time.time()

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self._frame, self._stamp

    def stop(self):
        self._running = False
        self.join(timeout=2.0)
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001
            pass


class CameraRig:
    """按配置打开所有相机；get() 返回 {name: RGB uint8 (H,W,3)}，已缩放到存盘分辨率。"""

    def __init__(self, cfg: dict):
        cc = cfg["cameras"]
        cap = cc["capture"]
        self.save_wh = (cc["save"]["width"], cc["save"]["height"])
        self.cams = {
            name: RealSenseCamera(name, str(dev["serial"]), cap["width"], cap["height"], cap["fps"])
            for name, dev in cc["devices"].items()
        }
        self.max_age_s = 0.2

    def start(self):
        for cam in self.cams.values():
            cam.start()
        t0 = time.time()
        while any(c.latest()[0] is None for c in self.cams.values()):
            if time.time() - t0 > 10:
                missing = [n for n, c in self.cams.items() if c.latest()[0] is None]
                raise TimeoutError(f"相机无图像: {missing}")
            time.sleep(0.05)
        # 跳过自动曝光稳定期
        time.sleep(1.0)

    def get(self) -> dict[str, np.ndarray]:
        out = {}
        now = time.time()
        for name, cam in self.cams.items():
            img, stamp = cam.latest()
            if img is None or now - stamp > self.max_age_s:
                raise RuntimeError(f"相机 {name} 图像过期 ({now - stamp:.2f}s)，检查 USB 带宽/线缆")
            out[name] = resize(img, self.save_wh)
        return out

    def stop(self):
        for cam in self.cams.values():
            cam.stop()


def resize(img: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    if (img.shape[1], img.shape[0]) == tuple(wh):
        return img
    return np.asarray(Image.fromarray(img).resize(tuple(wh), Image.BILINEAR, reducing_gap=2.0))
