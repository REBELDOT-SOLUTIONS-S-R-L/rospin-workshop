from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


@dataclass(frozen=True)
class RuntimeConfig:
    data_root: Path = Path(os.environ.get("ROSPIN_DATA_ROOT", "/workspace/data"))
    tasks_root: Path = Path(os.environ.get("ROSPIN_TASKS_DIR", "tasks"))
    host: str = os.environ.get("ROSPIN_HOST", "0.0.0.0")
    port: int = _int_env("ROSPIN_PORT", 8000)
    control_hz: int = _int_env("ROSPIN_CONTROL_HZ", 60)
    camera_hz: int = _int_env("ROSPIN_CAMERA_HZ", 25)
    image_width: int = _int_env("ROSPIN_IMAGE_WIDTH", 640)
    image_height: int = _int_env("ROSPIN_IMAGE_HEIGHT", 480)
    remote_port: str | None = _optional_env("ROSPIN_REMOTE_PORT")
    remote_hz: int = _int_env("ROSPIN_REMOTE_HZ", 60)

    @property
    def datasets_root(self) -> Path:
        return self.data_root / "datasets"

    @property
    def outputs_root(self) -> Path:
        return self.data_root / "outputs"

    @property
    def remote_calibration_root(self) -> Path:
        return self.data_root / "calibration" / "remote"
