from __future__ import annotations

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.recorder import LeRobotV3Recorder


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
    assert features["observation.images.wrist"]["dtype"] == "video"
    assert features["observation.images.wrist"]["shape"] == (480, 640, 3)
    assert "observation.images.perspective" not in features
    assert features["observation.eef_orientation"]["shape"] == (4,)
    assert features["action"]["shape"] == (6,)
    assert features["action"]["names"] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
