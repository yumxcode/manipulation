"""原始回合存储格式（所有后续数据集都从这里重新生成，是唯一的“真值”）。

data/raw/<task_key>/<YYYYmmdd_HHMMSS_ffffff>/
    meta.json        任务、来源(demo/rollout)、是否成功、帧数、相机名 ...
    lowdim.npz       每帧低维数据（见 LOWDIM_KEYS）
    images/<cam>/000000.jpg ...
    value.npz        (RECAP) 价值模型预测，label_advantage.py 生成
    advantage.npz    (RECAP) 每帧优势 / 二值标签，label_advantage.py 生成

动作标签 action_q_deg[t] = 第 t 帧之后立即下发给机械臂的目标关节角（遥操作 IK 结果或策略输出），
即“这一帧观测下做了什么”，与部署时策略要预测的量完全一致。
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import queue
import shutil
import threading

import numpy as np
from PIL import Image

LOWDIM_KEYS = (
    "t",               # (T,)   time.time()
    "q_deg",           # (T,6)  实际关节角
    "q_target_deg",    # (T,6)  控制器内部目标
    "tcp_pose",        # (T,6)  mm/deg
    "ft",              # (T,6)  N/Nm
    "gripper_pos",     # (T,)   0开~1闭
    "action_q_deg",    # (T,6)  下发的目标关节角
    "action_gripper",  # (T,)   下发的夹爪指令
    "intervention",    # (T,)   1=该帧由人遥操作（演示或接管），0=策略自主
    "hold",            # (T,)   1=处于保护停
)


class _ImageWriter:
    def __init__(self, quality: int, n_threads: int = 4):
        self.q: queue.Queue = queue.Queue(maxsize=512)
        self.quality = quality
        self.threads = [threading.Thread(target=self._loop, daemon=True) for _ in range(n_threads)]
        for t in self.threads:
            t.start()

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            path, img = item
            Image.fromarray(img).save(path, quality=self.quality)
            self.q.task_done()

    def put(self, path, img):
        self.q.put((path, img))

    def flush(self):
        self.q.join()

    def close(self):
        for _ in self.threads:
            self.q.put(None)
        for t in self.threads:
            t.join()


class EpisodeWriter:
    def __init__(self, raw_dir: str | pathlib.Path, task_key: str, prompt: str, fps: int, cameras: list[str],
                 source: str, jpeg_quality: int = 92, extra_meta: dict | None = None):
        assert source in ("demo", "rollout")
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.dir = pathlib.Path(raw_dir) / task_key / stamp
        for cam in cameras:
            (self.dir / "images" / cam).mkdir(parents=True, exist_ok=True)
        self.meta = {
            "task_key": task_key, "prompt": prompt, "fps": fps, "cameras": cameras, "source": source,
            "created": stamp, "success": None, "num_frames": 0, **(extra_meta or {}),
        }
        self.buf: dict[str, list] = {k: [] for k in LOWDIM_KEYS}
        self.images = _ImageWriter(jpeg_quality)
        self.n = 0

    def add(self, images: dict[str, np.ndarray], **lowdim):
        for cam, img in images.items():
            self.images.put(self.dir / "images" / cam / f"{self.n:06d}.jpg", img)
        for k in LOWDIM_KEYS:
            self.buf[k].append(np.asarray(lowdim[k]))
        self.n += 1

    def save(self, success: bool, notes: str = "") -> pathlib.Path:
        self.images.flush()
        self.images.close()
        np.savez_compressed(self.dir / "lowdim.npz", **{k: np.stack(v) for k, v in self.buf.items()})
        self.meta.update(success=bool(success), num_frames=self.n, notes=notes)
        (self.dir / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2))
        return self.dir

    def discard(self):
        self.images.close()
        shutil.rmtree(self.dir, ignore_errors=True)


# ---------------------------------------------------------------------------- 读取
class Episode:
    def __init__(self, path: str | pathlib.Path):
        self.dir = pathlib.Path(path)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        with np.load(self.dir / "lowdim.npz") as z:
            self.lowdim = {k: z[k] for k in z.files}

    def __len__(self):
        return int(self.meta["num_frames"])

    @property
    def success(self) -> bool:
        return bool(self.meta["success"])

    def image(self, cam: str, i: int) -> np.ndarray:
        return np.asarray(Image.open(self.dir / "images" / cam / f"{i:06d}.jpg").convert("RGB"))

    def load_npz(self, name: str) -> dict | None:
        p = self.dir / name
        if not p.exists():
            return None
        with np.load(p) as z:
            return {k: z[k] for k in z.files}


def list_episodes(raw_dir: str | pathlib.Path, tasks: list[str] | None = None, sources: list[str] | None = None) -> list[Episode]:
    raw_dir = pathlib.Path(raw_dir)
    eps = []
    for meta in sorted(raw_dir.glob("*/*/meta.json")):
        ep = Episode(meta.parent)
        if tasks and ep.meta["task_key"] not in tasks:
            continue
        if sources and ep.meta["source"] not in sources:
            continue
        eps.append(ep)
    return eps
