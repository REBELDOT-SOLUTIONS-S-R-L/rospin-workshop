from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.controller import (
    KEY_ACTIONS,
    PERSPECTIVE_DEFAULT_DISTANCE,
    PerspectiveCamera,
    WorkshopController,
)
from rospin_workshop.dataset_tools import inspect_dataset
from rospin_workshop.env import ACTION_NAMES, SO101WorkshopEnv


def test_threaded_teleop_and_real_lerobot_recording(tmp_path) -> None:
    controller = WorkshopController(
        RuntimeConfig(
            data_root=tmp_path,
            control_hz=50,
            camera_hz=25,
            image_width=96,
            image_height=72,
        )
    )
    controller.start()
    try:
        controller.select_task("cube_in_bowl")
        start_y = controller.status()["eef_position"][1]
        controller.set_key("w", True)
        time.sleep(0.5)
        controller.set_key("w", False)
        assert controller.status()["eef_position"][1] < start_y - 0.005
        assert controller.status()["sim_time"] >= 0.2
        assert controller.status()["camera_hz"] == 25

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
        recording_started = time.monotonic()
        time.sleep(1.0)
        recorded_frames = controller.status()["frames_in_episode"]
        recording_elapsed = time.monotonic() - recording_started
        assert 24 <= recorded_frames <= 27
        assert 22.5 <= (recorded_frames - 1) / recording_elapsed <= 26.0
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
    assert details["robot_type"] == "so_follower"
    assert details["episodes"] == 1
    assert details["fps"] == 25
    assert details["video_fps"] == 25
    assert np.isclose(details["timestamp_step_seconds"], 0.04, atol=1e-5)
    assert details["features"]["observation.images.wrist"]["info"][
        "video.codec"
    ] == "av1"
    assert (
        details["features"]["observation.images.wrist"]["info"]["video.fps"] == 25
    )
    assert "observation.images.top" not in details["features"]
    assert "observation.images.perspective" not in details["features"]
    assert details["decoded_frame_shapes"] == {
        "observation.images.wrist": [3, 72, 96],
    }
    assert details["features"]["action"]["shape"] == (6,)
    assert details["features"]["action"]["names"][-1] == "gripper.pos"


def test_keyboard_mapping_and_gripper_command_latching(tmp_path) -> None:
    expected_mappings = {
        "w": (1, -1),
        "s": (1, 1),
        "a": (0, 1),
        "d": (0, -1),
        "q": (2, 1),
        "e": (2, -1),
        "u": (3, 1),
        "o": (3, -1),
        "r": (4, 1),
        "f": (4, -1),
        "t": (5, 1),
        "g": (5, -1),
        "i": (6, 1),
        "k": (6, -1),
        "j": (7, 1),
        "l": (7, -1),
        "[": (8, -1),
        "]": (8, 1),
    }
    assert set(KEY_ACTIONS) == set(expected_mappings)
    for key, (action_index, value) in expected_mappings.items():
        expected = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        expected[action_index] = value
        np.testing.assert_array_equal(KEY_ACTIONS[key], expected)

    controller = WorkshopController(RuntimeConfig(data_root=tmp_path))
    assert not controller._render_requested.is_set()
    controller.set_key("[", True)
    controller.set_key("[", False)
    assert not controller._render_requested.is_set()
    assert controller._current_action()[-1] == -1
    controller.set_key("]", True)
    controller.set_key("]", False)
    assert controller._current_action()[-1] == 1


def test_keyboard_step_returns_six_absolute_joint_targets(tmp_path) -> None:
    controller = WorkshopController(RuntimeConfig(data_root=tmp_path))
    env = SO101WorkshopEnv(render_mode=None)
    controller.env = env
    try:
        env.reset()
        targets_before = env.data.ctrl.copy()
        measured_before = env.joint_positions.copy()
        controller.set_key("u", True)

        recorded_action = controller._step_control()

        assert recorded_action.shape == (6,)
        np.testing.assert_allclose(recorded_action, env.data.ctrl)
        assert np.isclose(
            recorded_action[0],
            targets_before[0] + env.joint_step,
        )
        np.testing.assert_allclose(recorded_action[1:], targets_before[1:])
        assert not np.array_equal(recorded_action, measured_before)
        assert controller._active_control_source == "keyboard"
    finally:
        env.close()


def test_perspective_camera_orbit_pan_zoom_and_reset(tmp_path) -> None:
    camera = PerspectiveCamera()
    initial_position, initial_lookat = camera.view()
    initial_status = camera.status()

    camera.apply("orbit", {"dx": 80, "dy": -20})
    orbit_position, orbit_lookat = camera.view()
    assert not np.allclose(orbit_position, initial_position)
    np.testing.assert_allclose(orbit_lookat, initial_lookat)

    camera.apply("pan", {"dx": 25, "dy": -10})
    pan_position, pan_lookat = camera.view()
    assert not np.allclose(pan_lookat, initial_lookat)
    np.testing.assert_allclose(
        pan_position - pan_lookat,
        orbit_position - orbit_lookat,
        atol=1e-12,
    )

    camera.apply("zoom", {"delta": -100})
    assert camera.status()["distance"] < initial_status["distance"]

    camera.apply("reset", {})
    reset_position, reset_lookat = camera.view()
    np.testing.assert_allclose(reset_position, initial_position, atol=1e-8)
    np.testing.assert_allclose(reset_lookat, initial_lookat)
    assert np.isclose(camera.status()["distance"], PERSPECTIVE_DEFAULT_DISTANCE)

    controller = WorkshopController(RuntimeConfig(data_root=tmp_path))
    assert not controller._render_requested.is_set()
    controller.control_perspective_camera("orbit", {"dx": 10, "dy": 0})
    assert not controller._render_requested.is_set()


def test_active_cameras_render_at_configured_size(tmp_path) -> None:
    controller = WorkshopController(
        RuntimeConfig(
            data_root=tmp_path,
            image_width=640,
            image_height=480,
        )
    )

    assert controller._camera_render_size("wrist") == (640, 480)
    assert controller._camera_render_size("perspective") == (640, 480)


def test_only_wrist_and_perspective_renders_are_submitted(tmp_path) -> None:
    controller = WorkshopController(RuntimeConfig(data_root=tmp_path))
    env = SO101WorkshopEnv(render_mode=None)
    controller.env = env
    try:
        env.reset()
        action = env.data.ctrl.astype(np.float32, copy=True)

        assert controller._submit_renders(action) is True
        assert controller._submit_renders(action) is False
        assert controller._render_inflight == {"perspective", "wrist"}
        assert len(controller._pending_renders) == 2
        dataset_pending = next(
            pending
            for pending in controller._pending_renders.values()
            if pending["record_dataset"]
        )
        assert dataset_pending["remaining_cameras"] == {"wrist"}
    finally:
        env.close()
