from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.controller import WorkshopController
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
        start_x = controller.status()["eef_position"][0]
        controller.set_key("w", True)
        time.sleep(0.5)
        controller.set_key("w", False)
        assert controller.status()["eef_position"][0] > start_x + 0.005
        assert controller.status()["sim_time"] >= 0.2
        assert controller.status()["camera_hz"] == 5

        start_quaternion = np.asarray(controller.status()["eef_orientation"])
        controller.set_key("u", True)
        time.sleep(0.25)
        controller.set_key("u", False)
        end_quaternion = np.asarray(controller.status()["eef_orientation"])
        rotation_angle = 2 * np.arccos(
            np.clip(abs(np.dot(start_quaternion, end_quaternion)), 0.0, 1.0)
        )
        assert rotation_angle > 0.01

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
