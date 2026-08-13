from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rospin_workshop.env import JOINT_NAMES

DEFAULT_TASK = "unspecified manipulation task"
MOTOR_POSITION_NAMES = tuple(f"{name}.pos" for name in JOINT_NAMES)

# The reference recording was made with LeRobot ``use_degrees=true``: the five
# arm values are calibrated degrees, while the gripper is calibrated to 0–100.
# This scene came from a LeIsaac task that applies a -pi/2 simulator-frame
# correction to wrist roll, so invert that offset when exporting motor values.
SIM_GRIPPER_CLOSED_RAD = np.deg2rad(-10.0)
SIM_GRIPPER_OPEN_RAD = np.deg2rad(100.0)
SIM_JOINT_OFFSETS_RADIANS = np.array(
    [0.0, 0.0, 0.0, 0.0, -np.pi / 2.0, 0.0],
    dtype=np.float64,
)


def simulation_to_real_motor_positions(values: np.ndarray) -> np.ndarray:
    """Convert six MuJoCo joint angles to the real LeRobot SO-101 convention."""

    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(JOINT_NAMES),):
        raise ValueError(
            f"Motor positions must contain {len(JOINT_NAMES)} values"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Motor positions must all be finite")
    converted = np.rad2deg(values - SIM_JOINT_OFFSETS_RADIANS)
    converted[-1] = (
        (values[-1] - SIM_GRIPPER_CLOSED_RAD)
        / (SIM_GRIPPER_OPEN_RAD - SIM_GRIPPER_CLOSED_RAD)
        * 100.0
    )
    converted[-1] = np.clip(converted[-1], 0.0, 100.0)
    return converted.astype(np.float32)


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
                "names": list(MOTOR_POSITION_NAMES),
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channels"],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(JOINT_NAMES),),
                "names": list(MOTOR_POSITION_NAMES),
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
            robot_type="so_follower",
            features=self.features,
            use_videos=True,
            rgb_encoder=RGBEncoderConfig(),
            streaming_encoding=True,
            encoder_queue_maxsize=max(30, self.fps * 5),
            encoder_threads=2,
            # Each saved episode is its own closed segment. Besides avoiding
            # increasingly expensive MP4 concatenation, this keeps completed
            # episodes readable if the simulation process is interrupted.
            metadata_buffer_size=1,
            # LeRobot treats zero as "use the default", so use a small
            # positive threshold to force rotation after every episode.
            data_files_size_in_mb=1e-6,
            video_files_size_in_mb=1e-6,
        )

    def resume_dataset(self, dataset_path: Path) -> None:
        """Resume appending to a crash-safe dataset created by this recorder."""

        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if self.dataset is not None:
            raise RuntimeError("A dataset session already exists")
        dataset_path = Path(dataset_path).resolve()
        info_path = dataset_path / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Dataset is missing metadata: {info_path}")

        self.repo_id = f"local/{dataset_path.name}"
        self.dataset_path = dataset_path
        self.dataset = LeRobotDataset.resume(
            repo_id=self.repo_id,
            root=dataset_path,
            rgb_encoder=RGBEncoderConfig(),
            streaming_encoding=True,
            encoder_queue_maxsize=max(30, self.fps * 5),
            encoder_threads=2,
        )
        actual_observations = {
            key
            for key in self.dataset.features
            if key.startswith("observation.")
        }
        if actual_observations != {
            "observation.state",
            "observation.images.wrist",
        } or "action" not in self.dataset.features:
            self.dataset.finalize()
            self.dataset = None
            raise ValueError("Dataset does not use the wrist-only workshop schema")

    def _close_episode_files(self) -> None:
        """Commit Parquet footers at every successful episode boundary."""

        self.dataset.writer.close_writer()
        self.dataset.meta._close_writer()

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
        state = simulation_to_real_motor_positions(observation["observation.state"])
        action = simulation_to_real_motor_positions(action)
        frame = {
            "task": self.task,
            "observation.state": state,
            "observation.images.wrist": observation["observation.images.wrist"].copy(),
            "action": action,
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
            self._close_episode_files()
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
