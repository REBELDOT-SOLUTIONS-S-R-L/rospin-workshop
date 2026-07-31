from __future__ import annotations

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.recorder import LeRobotV3Recorder


def test_lerobot_v3_feature_schema(tmp_path) -> None:
    assert RuntimeConfig(data_root=tmp_path).camera_hz == 25
    recorder = LeRobotV3Recorder(
        datasets_root=tmp_path,
        fps=20,
        image_width=320,
        image_height=240,
    )
    features = recorder.features
    assert features["observation.state"]["shape"] == (6,)
    assert features["observation.images.wrist"]["dtype"] == "video"
    assert features["observation.images.wrist"]["shape"] == (240, 320, 3)
    assert "observation.images.perspective" not in features
    assert features["observation.eef_orientation"]["shape"] == (4,)
    assert features["action"]["shape"] == (9,)
    assert features["action"]["names"][4:6] == [
        "shoulder_lift_delta",
        "elbow_flex_delta",
    ]
    assert features["action"]["names"][-1] == "gripper_command"
