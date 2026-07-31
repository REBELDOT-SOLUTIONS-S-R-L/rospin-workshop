from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rospin_workshop.env import JOINT_NAMES
from rospin_workshop.remote import (
    SO101RemoteReader,
    remote_positions_to_joint_targets,
)


@dataclass
class FakeCalibration:
    id: int
    drive_mode: int = 0
    homing_offset: int = 0
    range_min: int = 500
    range_max: int = 3500


class FakeBus:
    def __init__(self) -> None:
        self.connected = False
        self.calibration = {}
        self.disable_torque_calls = 0
        self.disconnect_calls = 0

    def connect(self) -> None:
        self.connected = True

    def read_calibration(self) -> dict[str, FakeCalibration]:
        return {
            name: FakeCalibration(id=motor_id)
            for motor_id, name in enumerate(JOINT_NAMES, start=1)
        }

    def disable_torque(self) -> None:
        self.disable_torque_calls += 1

    def disconnect(self, *, disable_torque: bool) -> None:
        assert disable_torque is False
        self.disconnect_calls += 1
        self.connected = False


class FakeLeader:
    def __init__(self) -> None:
        self.bus = FakeBus()
        self.calibration = {}
        self.reads = 0

    def get_action(self) -> dict[str, float]:
        assert self.bus.connected
        assert self.bus.disable_torque_calls >= 1
        self.reads += 1
        values = (10.0, -20.0, 30.0, -40.0, 50.0, 75.0)
        return {
            f"{name}.pos": value
            for name, value in zip(JOINT_NAMES, values, strict=True)
        }


def test_remote_mapping_converts_degrees_and_gripper_percentage() -> None:
    ranges = np.array(
        [
            [-1.0, 1.0],
            [-2.0, 2.0],
            [-3.0, 3.0],
            [-4.0, 4.0],
            [-5.0, 5.0],
            [-0.2, 1.8],
        ]
    )
    positions = {
        "shoulder_pan.pos": 30.0,
        "shoulder_lift.pos": -45.0,
        "elbow_flex.pos": 90.0,
        "wrist_flex.pos": -120.0,
        "wrist_roll.pos": 180.0,
        "gripper.pos": 25.0,
    }
    targets = remote_positions_to_joint_targets(positions, ranges)
    np.testing.assert_allclose(
        targets[:-1],
        np.deg2rad([30.0, -45.0, 90.0, -120.0, 180.0]),
    )
    assert np.isclose(targets[-1], 0.3)


def test_remote_reader_uses_motor_calibration_and_disables_torque(tmp_path) -> None:
    leader = FakeLeader()
    reader = SO101RemoteReader(
        port="/dev/fake-so101",
        calibration_dir=tmp_path,
        poll_hz=100,
        reconnect_after=0.01,
        leader_factory=lambda: leader,
    )
    reader.start()
    deadline = time.monotonic() + 2
    while reader.latest_positions() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    positions = reader.latest_positions()
    status = reader.status()
    reader.stop()

    assert positions is not None
    np.testing.assert_allclose(positions, [10, -20, 30, -40, 50, 75])
    assert status["configured"] is True
    assert status["connected"] is True
    assert status["calibrated"] is True
    assert status["available"] is True
    assert status["error"] is None
    assert leader.calibration == leader.bus.calibration
    assert set(leader.calibration) == set(JOINT_NAMES)
    assert leader.bus.disable_torque_calls >= 2
    assert leader.bus.disconnect_calls == 1
