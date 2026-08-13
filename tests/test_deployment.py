from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rospin_workshop.deployment import (
    PolicyDeploymentManager,
    resolve_act_checkpoint,
)
from rospin_workshop.recorder import real_to_simulation_motor_positions


def make_checkpoint(
    outputs_root: Path,
    name: str = "act/checkpoints/000100/pretrained_model",
) -> Path:
    checkpoint = outputs_root / name
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {
                    "observation.state": {"shape": [6]},
                    "observation.images.wrist": {"shape": [3, 72, 96]},
                },
                "output_features": {"action": {"shape": [6]}},
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (checkpoint / name).write_bytes(b"test")
    return checkpoint


def test_resolve_act_checkpoint_is_local_complete_and_compatible(tmp_path) -> None:
    outputs_root = tmp_path / "outputs"
    checkpoint = make_checkpoint(outputs_root)

    assert resolve_act_checkpoint(
        "act/checkpoints/000100/pretrained_model", outputs_root
    ) == checkpoint.resolve()

    with pytest.raises(ValueError, match="under the workshop outputs root"):
        resolve_act_checkpoint(tmp_path, outputs_root)
    (checkpoint / "policy_postprocessor.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        resolve_act_checkpoint(checkpoint, outputs_root)


class FakeController:
    def __init__(self, *, succeed_after: int | None = 2) -> None:
        self.config = SimpleNamespace(camera_hz=25, image_height=72, image_width=96)
        self.succeed_after = succeed_after
        self.policy_active = False
        self.sequence = 0
        self.actions_this_episode = 0
        self.targets: list[np.ndarray] = []
        self.resets: list[int] = []

    def status(self) -> dict:
        return {
            "task_ready": True,
            "recording": False,
            "error": None,
            "task_success": (
                self.succeed_after is not None
                and self.actions_this_episode >= self.succeed_after
            ),
            "episode_timeout_seconds": 0.04,
        }

    def begin_policy_control(self) -> None:
        assert not self.policy_active
        self.policy_active = True

    def end_policy_control(self) -> None:
        self.policy_active = False

    def observation_sequence(self) -> int:
        return self.sequence

    def command(self, command: str, payload: dict) -> None:
        assert command == "reset"
        self.resets.append(payload["seed"])
        self.actions_this_episode = 0
        self.sequence += 1

    def wait_for_policy_observation(
        self, *, after_sequence: int, timeout: float
    ) -> tuple[int, dict[str, np.ndarray]]:
        assert timeout > 0
        time.sleep(0.002)
        self.sequence = max(self.sequence, after_sequence) + 1
        return self.sequence, {
            "observation.state": np.zeros(6, dtype=np.float32),
            "observation.images.wrist": np.zeros((72, 96, 3), dtype=np.uint8),
        }

    def set_policy_joint_targets(self, targets: np.ndarray) -> None:
        assert self.policy_active
        self.targets.append(targets.copy())
        self.actions_this_episode += 1


class FakeRuntime:
    def __init__(self) -> None:
        self.resets = 0
        self.observations: list[dict[str, np.ndarray]] = []

    def reset(self) -> None:
        self.resets += 1

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        self.observations.append(observation)
        return np.arange(6, dtype=np.float64)


def wait_until_finished(manager: PolicyDeploymentManager) -> dict:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = manager.status()
        if not status["running"]:
            return status
        time.sleep(0.005)
    raise AssertionError("Deployment did not finish")


def test_policy_deployment_runs_seeded_success_episodes(tmp_path) -> None:
    outputs_root = tmp_path / "outputs"
    make_checkpoint(outputs_root)
    controller = FakeController(succeed_after=2)
    runtime = FakeRuntime()
    manager = PolicyDeploymentManager(
        controller,
        outputs_root,
        runtime_factory=lambda _path, _device: runtime,
    )

    started = manager.start(
        checkpoint="act/checkpoints/000100/pretrained_model",
        episodes=2,
        seed=40,
        device="cpu",
    )
    assert started["running"] is True
    status = wait_until_finished(manager)

    assert status["phase"] == "complete"
    assert status["error"] is None
    assert status["completed_episodes"] == 2
    assert status["successful_episodes"] == 2
    assert status["timed_out_episodes"] == 0
    assert status["actions_sent"] == 4
    assert [result["seed"] for result in status["results"]] == [40, 41]
    assert [result["outcome"] for result in status["results"]] == [
        "success",
        "success",
    ]
    assert controller.resets == [40, 41]
    assert controller.policy_active is False
    assert runtime.resets == 2
    assert len(runtime.observations) == 4
    np.testing.assert_allclose(
        runtime.observations[0]["observation.state"],
        [0.0, 0.0, 0.0, 0.0, 90.0, 9.090909],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        controller.targets[0],
        real_to_simulation_motor_positions(np.arange(6, dtype=np.float64)),
    )


def test_policy_deployment_reports_timeout(tmp_path) -> None:
    outputs_root = tmp_path / "outputs"
    make_checkpoint(outputs_root)
    controller = FakeController(succeed_after=None)
    manager = PolicyDeploymentManager(
        controller,
        outputs_root,
        runtime_factory=lambda _path, _device: FakeRuntime(),
    )

    manager.start(
        checkpoint="act/checkpoints/000100/pretrained_model",
        episodes=1,
        seed=9,
        device="cpu",
    )
    status = wait_until_finished(manager)

    assert status["phase"] == "complete"
    assert status["successful_episodes"] == 0
    assert status["timed_out_episodes"] == 1
    assert status["results"][0]["outcome"] == "timeout"
    assert controller.policy_active is False


def test_policy_deployment_can_be_stopped_cleanly(tmp_path) -> None:
    outputs_root = tmp_path / "outputs"
    make_checkpoint(outputs_root)
    controller = FakeController(succeed_after=None)
    manager = PolicyDeploymentManager(
        controller,
        outputs_root,
        runtime_factory=lambda _path, _device: FakeRuntime(),
    )

    manager.start(
        checkpoint="act/checkpoints/000100/pretrained_model",
        episodes=10,
        seed=9,
        device="cpu",
    )
    manager.stop()
    status = wait_until_finished(manager)

    assert status["phase"] == "cancelled"
    assert status["completed_episodes"] == 0
    assert controller.policy_active is False
