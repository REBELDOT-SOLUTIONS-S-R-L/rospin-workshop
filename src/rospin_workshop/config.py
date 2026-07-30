from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class RuntimeConfig:
    data_root: Path = Path(os.environ.get("ROSPIN_DATA_ROOT", "/workspace/data"))
    host: str = os.environ.get("ROSPIN_HOST", "0.0.0.0")
    port: int = _int_env("ROSPIN_PORT", 8000)
    control_hz: int = _int_env("ROSPIN_CONTROL_HZ", 60)
    camera_hz: int = _int_env("ROSPIN_CAMERA_HZ", 15)
    image_width: int = _int_env("ROSPIN_IMAGE_WIDTH", 320)
    image_height: int = _int_env("ROSPIN_IMAGE_HEIGHT", 240)

    @property
    def datasets_root(self) -> Path:
        return self.data_root / "datasets"

    @property
    def outputs_root(self) -> Path:
        return self.data_root / "outputs"
