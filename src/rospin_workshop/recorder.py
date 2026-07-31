from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rospin_workshop.env import ACTION_NAMES, JOINT_NAMES

DEFAULT_TASK = "unspecified manipulation task"


def _safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip()).strip("._-")
    return value[:64] or "so101_session"


class LeRobotV3Recorder:
    """Local-only, manual episode writer backed by ``LeRobotDataset``."""

    def __init__(
        self,
        *,
        datasets_root: Path,
        fps: int,
        image_width: int,
        image_height: int,
    ) -> None:
        self.datasets_root = Path(datasets_root)
        self.fps = fps
        self.image_width = image_width
        self.image_height = image_height
        self.dataset: Any | None = None
        self.dataset_path: Path | None = None
        self.repo_id: str | None = None
        self.recording = False
        self.finalized = False
        self.task = DEFAULT_TASK
        self.frames_in_episode = 0

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        image_shape = (self.image_height, self.image_width, 3)
        return {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(JOINT_NAMES),),
                "names": list(JOINT_NAMES),
            },
            "observation.velocity": {
                "dtype": "float32",
                "shape": (len(JOINT_NAMES),),
                "names": list(JOINT_NAMES),
            },
            "observation.eef_position": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["x", "y", "z"],
            },
            "observation.eef_orientation": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["w", "x", "y", "z"],
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(ACTION_NAMES),),
                "names": list(ACTION_NAMES),
            },
        }

    @property
    def num_episodes(self) -> int:
        if self.dataset is None:
            return 0
        return int(self.dataset.meta.total_episodes)

    def _create_dataset(self, requested_name: str) -> None:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        session_name = f"{_safe_name(requested_name)}_{timestamp}"
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        root = self.datasets_root / session_name
        suffix = 1
        while root.exists():
            root = self.datasets_root / f"{session_name}_{suffix}"
            suffix += 1

        self.repo_id = f"local/{root.name}"
        self.dataset_path = root
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=root,
            fps=self.fps,
            robot_type="so101_mujoco",
            features=self.features,
            use_videos=True,
            rgb_encoder=RGBEncoderConfig(
                vcodec="h264",
                pix_fmt="yuv420p",
                g=self.fps,
                crf=23,
                preset="veryfast",
            ),
            streaming_encoding=True,
            encoder_queue_maxsize=max(30, self.fps * 5),
            encoder_threads=2,
        )

    def start_episode(self, *, dataset_name: str, task: str) -> None:
        if self.finalized:
            raise RuntimeError(
                "This dataset is finalized; create a new recording session"
            )
        if self.recording:
            raise RuntimeError("An episode is already being recorded")
        if self.dataset is None:
            self._create_dataset(dataset_name)
        if self.dataset.has_pending_frames():
            raise RuntimeError("The previous episode has pending frames")
        self.task = task.strip() or DEFAULT_TASK
        self.frames_in_episode = 0
        self.recording = True

    def add_frame(self, observation: dict[str, np.ndarray], action: np.ndarray) -> None:
        if not self.recording:
            return
        frame = {
            "task": self.task,
            "observation.state": observation["observation.state"].copy(),
            "observation.velocity": observation["observation.velocity"].copy(),
            "observation.eef_position": observation["observation.eef_position"].copy(),
            "observation.eef_orientation": observation[
                "observation.eef_orientation"
            ].copy(),
            "observation.images.wrist": observation["observation.images.wrist"].copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
        }
        self.dataset.add_frame(frame)
        self.frames_in_episode += 1

    def stop_episode(self, *, save: bool = True) -> int:
        if not self.recording:
            raise RuntimeError("No episode is being recorded")
        self.recording = False
        frames = self.frames_in_episode
        if frames == 0:
            save = False
        if save:
            self.dataset.save_episode(parallel_encoding=False)
        elif self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()
        self.frames_in_episode = 0
        return frames

    def finalize(self) -> Path | None:
        if self.finalized:
            return self.dataset_path
        if self.recording:
            self.stop_episode(save=True)
        if self.dataset is not None:
            self.dataset.finalize()
        self.finalized = True
        return self.dataset_path
