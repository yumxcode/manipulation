"""读取 configs/hardware.yaml。保持为普通 dict + 少量辅助函数，方便现场改参数。"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import yaml

DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parent.parent / "configs" / "hardware.yaml"


def load_config(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    path = pathlib.Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def camera_slots(cfg: dict) -> dict[str, str]:
    """相机名 -> π0.5 图像槽位。"""
    return {name: dev["slot"] for name, dev in cfg["cameras"]["devices"].items()}


def max_episode_steps(cfg: dict) -> int:
    return int(cfg["task"]["max_episode_s"] * cfg["task"]["fps"])


def merged(cfg: dict, **overrides) -> dict:
    out = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = out
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return out
