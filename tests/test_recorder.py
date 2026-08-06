from __future__ import annotations

import numpy as np

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.recorder import (
    MOTOR_POSITION_NAMES,
    LeRobotV3Recorder,
    simulation_to_real_motor_positions,
)


def test_lerobot_v3_feature_schema(tmp_path) -> None:
    assert RuntimeConfig(data_root=tmp_path).camera_hz == 25
    assert RuntimeConfig(data_root=tmp_path).image_width == 640
    assert RuntimeConfig(data_root=tmp_path).image_height == 480
    recorder = LeRobotV3Recorder(
        datasets_root=tmp_path,
        fps=20,
        image_width=640,
        image_height=480,
    )
    features = recorder.features
    assert features["observation.state"]["shape"] == (6,)
    assert features["observation.state"]["names"] == list(MOTOR_POSITION_NAMES)
    assert features["observation.images.top"]["dtype"] == "video"
    assert features["observation.images.top"]["shape"] == (480, 640, 3)
    assert features["observation.images.wrist"]["dtype"] == "video"
    assert features["observation.images.wrist"]["shape"] == (480, 640, 3)
    assert "observation.images.perspective" not in features
    assert "observation.velocity" not in features
    assert "observation.eef_position" not in features
    assert "observation.eef_orientation" not in features
    assert features["action"]["shape"] == (6,)
    assert features["action"]["names"] == list(MOTOR_POSITION_NAMES)


def test_simulation_positions_match_real_so101_units() -> None:
    simulation = np.deg2rad([-90.0, -45.0, 30.0, 75.0, -90.0, -10.0])
    np.testing.assert_allclose(
        simulation_to_real_motor_positions(simulation),
        [-90.0, -45.0, 30.0, 75.0, 0.0, 0.0],
        atol=1e-5,
    )

    simulation[-1] = np.deg2rad(45.0)
    assert np.isclose(simulation_to_real_motor_positions(simulation)[-1], 50.0)

    workshop_home = np.array(
        [0.0, -1.6580628, 1.5707963, 1.2217305, -1.5707963, 0.2617994]
    )
    np.testing.assert_allclose(
        simulation_to_real_motor_positions(workshop_home),
        [0.0, -95.0, 90.0, 70.0, 0.0, 22.7273],
        atol=1e-3,
    )
