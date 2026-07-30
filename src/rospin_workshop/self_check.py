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
            brake_action = np.zeros(7, dtype=np.float32)
            observation, _, _, _, _ = env.step(brake_action)
            recorder.add_frame(observation, brake_action)
            rotation_start = env.eef_orientation.copy()
            joint_start = env.joint_positions.copy()
            target_start = env.data.ctrl.copy()
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
                    f"Positive wrist-roll did not rotate enough: {rotation_angle}"
                )
            if env.joint_positions[4] <= joint_start[4] + 0.01:
                raise RuntimeError("Wrist-roll joint did not move")
            changed_joint_targets = np.flatnonzero(
                np.abs(env.data.ctrl[:5] - target_start[:5]) > 0.01
            )
            if changed_joint_targets.tolist() != [4]:
                raise RuntimeError(
                    "Wrist-roll command targeted joints other than wrist_roll: "
                    f"{changed_joint_targets.tolist()}"
                )
            gripper_range = env.model.jnt_range[env._joint_ids[-1]]
            close_gripper = np.zeros(7, dtype=np.float32)
            close_gripper[6] = -1
            env.step_dynamics(close_gripper)
            for _ in range(40):
                env.step_dynamics(np.zeros(7, dtype=np.float32))
            closed_gripper_position = float(env.joint_positions[-1])
            if closed_gripper_position > gripper_range[0] + 0.05:
                raise RuntimeError(
                    "Gripper did not reach its closed joint limit: "
                    f"{closed_gripper_position}"
                )

            open_gripper = np.zeros(7, dtype=np.float32)
            open_gripper[6] = 1
            env.step_dynamics(open_gripper)
            for _ in range(60):
                env.step_dynamics(np.zeros(7, dtype=np.float32))
            opened_gripper_position = float(env.joint_positions[-1])
            if opened_gripper_position < gripper_range[1] - 0.05:
                raise RuntimeError(
                    "Gripper did not reach its open joint limit: "
                    f"{opened_gripper_position}"
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
                    "positive_wrist_roll_rotation_rad": rotation_angle,
                    "closed_gripper_position_rad": closed_gripper_position,
                    "opened_gripper_position_rad": opened_gripper_position,
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
