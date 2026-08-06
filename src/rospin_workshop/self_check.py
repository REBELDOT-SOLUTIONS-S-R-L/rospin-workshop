from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import gymnasium
import lerobot
import mujoco
import numpy as np
import torch

from rospin_workshop.dataset_tools import inspect_dataset
from rospin_workshop.env import (
    ACTION_NAMES,
    DIRECT_JOINT_ACTIONS,
    JOINT_NAMES,
    SO101WorkshopEnv,
)
from rospin_workshop.recorder import LeRobotV3Recorder
from rospin_workshop.tasks import TaskRegistry


def main() -> None:
    tasks_root = Path(os.environ.get("ROSPIN_TASKS_DIR", "tasks"))
    task = TaskRegistry(tasks_root).get("cube_in_bowl")
    env = SO101WorkshopEnv(
        task=task,
        image_width=96,
        image_height=72,
        control_hz=25,
    )
    try:
        observation, _ = env.reset(seed=7)
        direct_joint_target_checks: dict[str, str] = {}
        for action_index, joint_index in DIRECT_JOINT_ACTIONS:
            before = env.data.ctrl.copy()
            action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            action[action_index] = 1
            env.step_dynamics(action)
            changed = np.flatnonzero(np.abs(env.data.ctrl - before) > 1e-8)
            if changed.tolist() != [joint_index]:
                raise RuntimeError(
                    f"{ACTION_NAMES[action_index]} targeted joints "
                    f"{changed.tolist()}, expected [{joint_index}]"
                )
            direct_joint_target_checks[JOINT_NAMES[joint_index]] = "ok"
            env.reset()

        observation, _ = env.reset(seed=7)
        start_position = env.eef_position.copy()
        with tempfile.TemporaryDirectory(prefix="rospin-self-check-") as temp_dir:
            recorder = LeRobotV3Recorder(
                datasets_root=Path(temp_dir),
                fps=25,
                image_width=96,
                image_height=72,
            )
            recorder.start_episode(
                dataset_name="self_check", task="verify packaged runtime"
            )
            for _ in range(10):
                action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
                action[0] = 1
                observation, _, _, _, _ = env.step(action)
                recorder.add_frame(observation, env.data.ctrl)
            translated_position = env.eef_position.copy()
            brake_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            observation, _, _, _, _ = env.step(brake_action)
            recorder.add_frame(observation, env.data.ctrl)
            rotation_start = env.eef_orientation.copy()
            joint_start = env.joint_positions.copy()
            target_start = env.data.ctrl.copy()
            for _ in range(10):
                action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
                action[ACTION_NAMES.index("wrist_roll_delta")] = 1
                observation, _, _, _, _ = env.step(action)
                recorder.add_frame(observation, env.data.ctrl)
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
            close_gripper = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            close_gripper[-1] = -1
            env.step_dynamics(close_gripper)
            for _ in range(90):
                env.step_dynamics(np.zeros(len(ACTION_NAMES), dtype=np.float32))
            closed_gripper_position = float(env.joint_positions[-1])
            if closed_gripper_position > gripper_range[0] + 0.05:
                raise RuntimeError(
                    "Gripper did not reach its closed joint limit: "
                    f"{closed_gripper_position}"
                )

            open_gripper = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            open_gripper[-1] = 1
            env.step_dynamics(open_gripper)
            for _ in range(150):
                env.step_dynamics(np.zeros(len(ACTION_NAMES), dtype=np.float32))
            opened_gripper_position = float(env.joint_positions[-1])
            if opened_gripper_position < gripper_range[1] - 0.05:
                raise RuntimeError(
                    "Gripper did not reach its open joint limit: "
                    f"{opened_gripper_position}"
                )

            env.reset()
            bowl_region_id = mujoco.mj_name2id(
                env.model,
                mujoco.mjtObj.mjOBJ_SITE,
                "task_bowl__interior",
            )
            if bowl_region_id < 0:
                raise RuntimeError("cube_in_bowl task is missing its success region")
            env.set_task_object_pose(
                "cube",
                env.data.site_xpos[bowl_region_id].copy(),
            )
            env.data.qpos[env._qpos_indices[-1]] = env.joint_ranges[-1, 1]
            env.data.ctrl[-1] = env.joint_ranges[-1, 1]
            mujoco.mj_forward(env.model, env.data)
            for _ in range(60):
                env.step_dynamics(np.zeros(len(ACTION_NAMES), dtype=np.float32))
            if not env.task_status()["success"]:
                raise RuntimeError("cube_in_bowl success predicate did not latch")

            recorder.stop_episode(save=True)
            dataset_path = recorder.finalize()
            dataset = inspect_dataset(dataset_path)
            if "observation.images.top" not in dataset["features"]:
                raise RuntimeError("Dataset is missing the real-schema top camera")
            if dataset["fps"] != 25 or dataset["video_fps"] != 25:
                raise RuntimeError("Dataset videos are not all 25 FPS")
            if tuple(dataset["features"]["action"]["shape"]) != (
                len(JOINT_NAMES),
            ):
                raise RuntimeError("Dataset action is not six joint targets")
            if not np.isclose(dataset["timestamp_step_seconds"], 0.04, atol=1e-5):
                raise RuntimeError("Dataset timestamps are not spaced at 25 Hz")

            report = {
                "status": "ok",
                "versions": {
                    "mujoco": mujoco.__version__,
                    "gymnasium": gymnasium.__version__,
                    "lerobot": lerobot.__version__,
                    "torch": torch.__version__,
                },
                "simulation": {
                    "task": task.id,
                    "task_success": env.task_status()["success"],
                    "joints": env.model.nq,
                    "actuators": env.model.nu,
                    "cameras": env.model.ncam,
                    "positive_x_displacement_m": translation_x,
                    "positive_wrist_roll_rotation_rad": rotation_angle,
                    "direct_joint_targets": direct_joint_target_checks,
                    "closed_gripper_position_rad": closed_gripper_position,
                    "opened_gripper_position_rad": opened_gripper_position,
                },
                "dataset": {
                    "format": dataset["codebase_version"],
                    "episodes": dataset["episodes"],
                    "frames": dataset["frames"],
                    "fps": dataset["fps"],
                    "timestamp_step_seconds": dataset["timestamp_step_seconds"],
                    "video_codecs": {
                        "observation.images.top": dataset["features"][
                            "observation.images.top"
                        ]["info"]["video.codec"],
                        "observation.images.wrist": dataset["features"][
                            "observation.images.wrist"
                        ]["info"]["video.codec"]
                    },
                    "decoded_frame_shapes": dataset["decoded_frame_shapes"],
                },
            }
            print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
