from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ServiceConfig:
    camera_index: int | None = None
    camera_auto: bool = True
    camera_probe_max: int = 6
    board_orientation: str = "white"
    change_threshold: float = 20.0
    warp_size: int = 800
    loop_sleep_ms: int = 30
    auto_start: bool = True
    debug: bool = False
    move_post_url: str | None = None


def _to_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | int | None, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return int(value)


def _to_float(value: str | float | None, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, float):
        return value
    return float(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain an object: {path}")
    return data


def load_config(config_path: str | Path | None = None) -> ServiceConfig:
    repo_root = Path(__file__).resolve().parents[2]
    default_path = repo_root / "config" / "settings.yaml"
    config_file = Path(config_path) if config_path else default_path

    yaml_config = _load_yaml(config_file)

    raw_camera_index = os.getenv("CAMERA_INDEX", yaml_config.get("camera_index"))
    camera_auto = _to_bool(os.getenv("CAMERA_AUTO", yaml_config.get("camera_auto", True)), True)

    camera_index: int | None
    if raw_camera_index in (None, "", "AUTO", "auto"):
        camera_index = None
    else:
        camera_index = int(raw_camera_index)

    return ServiceConfig(
        camera_index=camera_index,
        camera_auto=camera_auto,
        camera_probe_max=_to_int(os.getenv("CAMERA_PROBE_MAX"), int(yaml_config.get("camera_probe_max", 6))),
        board_orientation=os.getenv("BOARD_ORIENTATION", str(yaml_config.get("board_orientation", "white"))).lower(),
        change_threshold=_to_float(os.getenv("THRESHOLD"), float(yaml_config.get("change_threshold", 10.0))),
        warp_size=_to_int(os.getenv("WARP_SIZE"), int(yaml_config.get("warp_size", 800))),
        loop_sleep_ms=_to_int(os.getenv("LOOP_SLEEP_MS"), int(yaml_config.get("loop_sleep_ms", 30))),
        auto_start=_to_bool(os.getenv("AUTO_START", yaml_config.get("auto_start", True)), True),
        debug=_to_bool(os.getenv("DEBUG", yaml_config.get("debug", False)), False),
        move_post_url=os.getenv("MOVE_POST_URL", yaml_config.get("move_post_url")) or None,
    )
