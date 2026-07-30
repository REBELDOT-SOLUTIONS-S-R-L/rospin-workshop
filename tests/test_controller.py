from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.controller import KEY_ACTIONS, WorkshopController
from rospin_workshop.dataset_tools import inspect_dataset


def test_threaded_teleop_and_real_lerobot_recording(tmp_path) -> None:
    controller = WorkshopController(
        RuntimeConfig(
            data_root=tmp_path,
            control_hz=20,
            camera_hz=5,
            image_width=96,
            image_height=72,
        )
    )
    controller.start()
    try:
        start_y = controller.status()["eef_position"][1]
        controller.set_key("w", True)
        time.sleep(0.5)
        controller.set_key("w", False)
        assert controller.status()["eef_position"][1] > start_y + 0.005
        assert controller.status()["sim_time"] >= 0.2
        assert controller.status()["camera_hz"] == 5

        start_x = controller.status()["eef_position"][0]
        controller.set_key("a", True)
        time.sleep(0.5)
        controller.set_key("a", False)
        assert controller.status()["eef_position"][0] > start_x + 0.005

        time.sleep(0.1)
        start_joints = np.asarray(controller.status()["joint_positions"])
        start_targets = np.asarray(controller.status()["joint_targets"])
        controller.set_key("j", True)
        time.sleep(0.25)
        active_targets = np.asarray(controller.status()["joint_targets"])
        controller.set_key("j", False)
        end_joints = np.asarray(controller.status()["joint_positions"])
        assert end_joints[4] > start_joints[4] + 0.01
        target_changes = np.flatnonzero(
            np.abs(active_targets[:5] - start_targets[:5]) > 0.01
        )
        assert target_changes.tolist() == [4]

        gripper_start = controller.status()["joint_positions"][5]
        controller.set_key("[", True)
        controller.set_key("[", False)
        time.sleep(0.6)
        closed = controller.status()
        gripper_range = controller.env.model.jnt_range[controller.env._joint_ids[-1]]
        assert np.isclose(closed["joint_targets"][5], gripper_range[0], atol=1e-4)
        assert closed["joint_positions"][5] < gripper_start - 0.2

        controller.set_key("]", True)
        controller.set_key("]", False)
        time.sleep(0.6)
        opened = controller.status()
        assert np.isclose(opened["joint_targets"][5], gripper_range[1], atol=1e-4)
        assert opened["joint_positions"][5] > closed["joint_positions"][5] + 0.2

        controller.command(
            "start_recording",
            {"dataset_name": "test_session", "task": "move the end effector"},
        )
        time.sleep(0.7)
        controller.command("stop_recording", {})
        controller.command("finish_dataset", {})
        status = controller.status()
        assert status["episodes"] == 1
        assert status["finalized"] is True
        dataset_path = Path(status["dataset_path"])
    finally:
        controller.close()

    details = inspect_dataset(dataset_path)
    assert details["codebase_version"] == "v3.0"
    assert details["episodes"] == 1
    assert details["fps"] == 5
    assert (
        details["features"]["observation.images.wrist"]["info"]["video.codec"] == "h264"
    )
    assert (
        details["features"]["observation.images.perspective"]["info"]["video.codec"]
        == "h264"
    )
    assert details["decoded_frame_shapes"] == {
        "observation.images.wrist": [3, 72, 96],
        "observation.images.perspective": [3, 72, 96],
    }


def test_keyboard_mapping_and_gripper_command_latching(tmp_path) -> None:
    np.testing.assert_array_equal(
        KEY_ACTIONS["w"], np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        KEY_ACTIONS["s"], np.array([0, -1, 0, 0, 0, 0, 0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        KEY_ACTIONS["a"], np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        KEY_ACTIONS["d"], np.array([-1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    )

    controller = WorkshopController(RuntimeConfig(data_root=tmp_path))
    controller.set_key("[", True)
    controller.set_key("[", False)
    assert controller._current_action()[6] == -1
    controller.set_key("]", True)
    controller.set_key("]", False)
    assert controller._current_action()[6] == 1
