from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gymnasium
import lerobot
import mujoco
import numpy as np
import torch

from rospin_workshop.dataset_tools import inspect_dataset
from rospin_workshop.env import SO101WorkshopEnv
from rospin_workshop.recorder import LeRobotV3Recorder


def main() -> None:
    env = SO101WorkshopEnv(image_width=96, image_height=72, control_hz=20)
    try:
        observation, _ = env.reset(seed=7)
        start_position = env.eef_position.copy()
        with tempfile.TemporaryDirectory(prefix="rospin-self-check-") as temp_dir:
            recorder = LeRobotV3Recorder(
                datasets_root=Path(temp_dir),
                fps=20,
                image_width=96,
                image_height=72,
            )
            recorder.start_episode(
                dataset_name="self_check", task="verify packaged runtime"
            )
            for _ in range(10):
                action = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
                observation, _, _, _, _ = env.step(action)
                recorder.add_frame(observation, action)
            translated_position = env.eef_position.copy()
            rotation_start = env.eef_orientation.copy()
            for _ in range(10):
                action = np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
                observation, _, _, _, _ = env.step(action)
                recorder.add_frame(observation, action)
            translation_x = float(translated_position[0] - start_position[0])
            rotation_angle = float(
                2
                * np.arccos(
                    np.clip(
                        abs(np.dot(rotation_start, env.eef_orientation)),
                        0.0,
                        1.0,
                    )
                )
            )
            if translation_x <= 0.005:
                raise RuntimeError(
                    f"Positive-X teleop did not move in +X: {translation_x}"
                )
            if rotation_angle <= 0.05:
                raise RuntimeError(
                    f"Positive-yaw teleop did not rotate enough: {rotation_angle}"
                )
            recorder.stop_episode(save=True)
            dataset_path = recorder.finalize()
            dataset = inspect_dataset(dataset_path)

            report = {
                "status": "ok",
                "versions": {
                    "mujoco": mujoco.__version__,
                    "gymnasium": gymnasium.__version__,
                    "lerobot": lerobot.__version__,
                    "torch": torch.__version__,
                },
                "simulation": {
                    "joints": env.model.nq,
                    "actuators": env.model.nu,
                    "cameras": env.model.ncam,
                    "positive_x_displacement_m": translation_x,
                    "positive_yaw_rotation_rad": rotation_angle,
                },
                "dataset": {
                    "format": dataset["codebase_version"],
                    "episodes": dataset["episodes"],
                    "frames": dataset["frames"],
                    "video_codecs": {
                        camera: dataset["features"][camera]["info"]["video.codec"]
                        for camera in (
                            "observation.images.wrist",
                            "observation.images.perspective",
                        )
                    },
                    "decoded_frame_shapes": dataset["decoded_frame_shapes"],
                },
            }
            print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
