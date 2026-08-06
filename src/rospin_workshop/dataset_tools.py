from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rospin_workshop.env import JOINT_NAMES
from rospin_workshop.recorder import MOTOR_POSITION_NAMES


def inspect_dataset(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobot v3.0, found {info.get('codebase_version')!r}"
        )
    if info.get("robot_type") != "so_follower":
        raise ValueError(
            f"Expected robot_type 'so_follower', found {info.get('robot_type')!r}"
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # LeRobot v3 info.json intentionally does not serialize repo_id. This
    # project records under a deterministic local/<directory> namespace.
    repo_id = f"local/{root.name}"
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    required = {
        "observation.state",
        "observation.images.top",
        "observation.images.wrist",
        "action",
    }
    missing = required.difference(dataset.features)
    if missing:
        raise ValueError(f"Dataset is missing required features: {sorted(missing)}")
    unexpected_observations = {
        name
        for name in dataset.features
        if name.startswith("observation.") and name not in required
    }
    if unexpected_observations:
        raise ValueError(
            "Dataset contains features absent from the real SO-101 schema: "
            f"{sorted(unexpected_observations)}"
        )
    for feature_name in ("observation.state", "action"):
        feature = dataset.features[feature_name]
        if tuple(feature["shape"]) != (len(JOINT_NAMES),) or list(
            feature.get("names", [])
        ) != list(MOTOR_POSITION_NAMES):
            raise ValueError(
                f"{feature_name} must contain the six real SO-101 *.pos values"
            )
    if dataset.num_episodes < 1 or dataset.num_frames < 1:
        raise ValueError("Dataset has no saved frames or episodes")
    sample = dataset[0]
    video_fps_by_camera: dict[str, float] = {}
    for camera_key in ("observation.images.top", "observation.images.wrist"):
        expected_h, expected_w, _ = dataset.features[camera_key]["shape"]
        if tuple(sample[camera_key].shape) != (3, expected_h, expected_w):
            raise ValueError(
                f"Could not decode {camera_key} at its declared resolution"
            )
        camera_info = dataset.features[camera_key]["info"]
        video_fps = float(camera_info["video.fps"])
        if not math.isclose(video_fps, float(dataset.fps)):
            raise ValueError(
                f"{camera_key} video FPS {video_fps} does not match dataset FPS "
                f"{dataset.fps}"
            )
        if camera_info["video.codec"] != "av1":
            raise ValueError(
                f"{camera_key} must use AV1 to match the real SO-101 dataset"
            )
        video_fps_by_camera[camera_key] = video_fps

    timestamp_step_seconds: float | None = None
    first_episode = int(sample["episode_index"])
    first_timestamp = float(sample["timestamp"])
    for index in range(1, min(dataset.num_frames, 1000)):
        next_sample = dataset[index]
        if int(next_sample["episode_index"]) != first_episode:
            break
        timestamp_step_seconds = float(next_sample["timestamp"]) - first_timestamp
        break
    if timestamp_step_seconds is not None and not math.isclose(
        timestamp_step_seconds,
        1.0 / float(dataset.fps),
        abs_tol=1e-5,
    ):
        raise ValueError(
            f"Dataset timestamp step {timestamp_step_seconds} does not match "
            f"{dataset.fps} FPS"
        )
    return {
        "root": str(root.resolve()),
        "repo_id": dataset.repo_id,
        "codebase_version": info["codebase_version"],
        "robot_type": info["robot_type"],
        "episodes": dataset.num_episodes,
        "frames": dataset.num_frames,
        "fps": dataset.fps,
        "video_fps": video_fps_by_camera["observation.images.wrist"],
        "video_fps_by_camera": video_fps_by_camera,
        "timestamp_step_seconds": timestamp_step_seconds,
        "features": dataset.features,
        "decoded_frame_shapes": {
            key: list(value.shape)
            for key, value in sample.items()
            if key.startswith("observation.images.")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local LeRobot v3 dataset")
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_dataset(args.dataset_root), indent=2, default=str))


if __name__ == "__main__":
    main()
