from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from rospin_workshop.controller import WorkshopController
from rospin_workshop.env import JOINT_NAMES
from rospin_workshop.recorder import (
    real_to_simulation_motor_positions,
    simulation_to_real_motor_positions,
)


REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
)


class PolicyRuntime(Protocol):
    def reset(self) -> None: ...

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray: ...


def resolve_act_checkpoint(checkpoint: str | Path, outputs_root: Path) -> Path:
    """Resolve and validate a local ACT checkpoint under the workshop outputs root."""

    outputs_root = Path(outputs_root).resolve()
    requested = Path(checkpoint)
    resolved = (
        requested.resolve()
        if requested.is_absolute()
        else (outputs_root / requested).resolve()
    )
    if not resolved.is_relative_to(outputs_root):
        raise ValueError("Policy checkpoint must be under the workshop outputs root")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {resolved}")
    missing = [
        name
        for name in REQUIRED_CHECKPOINT_FILES
        if not (resolved / name).is_file()
    ]
    if missing:
        raise ValueError(
            f"Policy checkpoint is incomplete; missing: {', '.join(missing)}"
        )

    config = json.loads((resolved / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "act":
        raise ValueError("Only ACT policy checkpoints can be deployed")
    inputs = config.get("input_features", {})
    outputs = config.get("output_features", {})
    if set(inputs) != {
        "observation.state",
        "observation.images.wrist",
    }:
        raise ValueError(
            "ACT checkpoint inputs must contain only state and the wrist camera"
        )
    if inputs.get("observation.state", {}).get("shape") != [len(JOINT_NAMES)]:
        raise ValueError("ACT checkpoint must consume six joint positions")
    if inputs.get("observation.images.wrist", {}).get("shape") is None:
        raise ValueError("ACT checkpoint must consume observation.images.wrist")
    if outputs.get("action", {}).get("shape") != [len(JOINT_NAMES)]:
        raise ValueError("ACT checkpoint must predict six absolute joint targets")
    if set(outputs) != {"action"}:
        raise ValueError("ACT checkpoint must have only the six-joint action output")
    return resolved


class ACTPolicyRuntime:
    """LeRobot ACT checkpoint plus its persisted normalization processors."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        import torch
        from lerobot.common.control_utils import predict_action
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA deployment was requested but PyTorch cannot access an NVIDIA GPU"
            )
        self.device = torch.device(device)
        config = PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=checkpoint,
            local_files_only=True,
        )
        if config.type != "act":
            raise ValueError(f"Expected an ACT checkpoint, found {config.type!r}")
        config.device = device
        config.pretrained_path = str(checkpoint)
        # The checkpoint already contains the trained backbone. Disabling the
        # constructor's ImageNet initialization avoids an unnecessary network
        # download before those checkpoint weights are restored.
        config.pretrained_backbone_weights = None
        policy_class = get_policy_class(config.type)
        self.policy = policy_class.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
        )
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
            },
        )
        self.use_amp = bool(config.use_amp)
        self._predict_action = predict_action
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

    def reset(self) -> None:
        self.policy.reset()

    def predict(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        action = self._predict_action(
            observation=observation,
            policy=self.policy,
            device=self.device,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=self.use_amp,
            robot_type="so_follower",
        )
        return action.squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)


class PolicyDeploymentManager:
    """Run a trained ACT policy against seeded episodes in the live simulator."""

    def __init__(
        self,
        controller: WorkshopController,
        outputs_root: Path,
        *,
        runtime_factory: Callable[[Path, str], PolicyRuntime] = ACTPolicyRuntime,
    ) -> None:
        self.controller = controller
        self.outputs_root = Path(outputs_root)
        self.runtime_factory = runtime_factory
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._episode_started_at: float | None = None
        self._status: dict[str, Any] = self._initial_status()

    @staticmethod
    def _initial_status() -> dict[str, Any]:
        return {
            "running": False,
            "checkpoint": None,
            "device": None,
            "requested_episodes": 0,
            "completed_episodes": 0,
            "successful_episodes": 0,
            "timed_out_episodes": 0,
            "current_episode": None,
            "current_seed": None,
            "phase": "idle",
            "episode_elapsed_seconds": None,
            "last_outcome": None,
            "last_inference_ms": None,
            "average_inference_ms": None,
            "actions_sent": 0,
            "results": [],
            "error": None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self._status)
            status["results"] = [dict(item) for item in self._status["results"]]
            if self._episode_started_at is not None:
                status["episode_elapsed_seconds"] = round(
                    time.monotonic() - self._episode_started_at, 2
                )
            return status

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._status = {**self._status, **values}

    def start(
        self,
        *,
        checkpoint: str,
        episodes: int,
        seed: int,
        device: str,
    ) -> dict[str, Any]:
        if not 1 <= episodes <= 1_000:
            raise ValueError("episodes must be between 1 and 1000")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        resolved = resolve_act_checkpoint(checkpoint, self.outputs_root)
        checkpoint_config = json.loads(
            (resolved / "config.json").read_text(encoding="utf-8")
        )
        wrist_shape = checkpoint_config["input_features"][
            "observation.images.wrist"
        ]["shape"]
        expected_wrist_shape = [
            3,
            self.controller.config.image_height,
            self.controller.config.image_width,
        ]
        if wrist_shape != expected_wrist_shape:
            raise ValueError(
                "Checkpoint wrist image shape "
                f"{wrist_shape} does not match the simulator shape "
                f"{expected_wrist_shape}"
            )
        controller_status = self.controller.status()
        if not controller_status.get("task_ready"):
            raise RuntimeError("Select a task before deploying a policy")
        if controller_status.get("recording"):
            raise RuntimeError("Stop the active recording before deploying a policy")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A policy deployment is already running")

        self.controller.begin_policy_control()
        self._stop_event = threading.Event()
        self._episode_started_at = None
        self._status = {
            **self._initial_status(),
            "running": True,
            "checkpoint": str(resolved),
            "device": device,
            "requested_episodes": episodes,
            "current_seed": seed,
            "phase": "loading",
        }
        self._thread = threading.Thread(
            target=self._run,
            kwargs={
                "checkpoint": resolved,
                "episodes": episodes,
                "seed": seed,
                "device": device,
            },
            name="act-policy-deployment",
            daemon=True,
        )
        self._thread.start()
        return self.status()

    def _run(
        self,
        *,
        checkpoint: Path,
        episodes: int,
        seed: int,
        device: str,
    ) -> None:
        completed = 0
        successful = 0
        timed_out = 0
        action_count = 0
        inference_total = 0.0
        results: list[dict[str, Any]] = []
        try:
            runtime = self.runtime_factory(checkpoint, device)
            self._update(phase="ready")
            for episode_index in range(episodes):
                if self._stop_event.is_set():
                    break
                episode_seed = seed + episode_index
                previous_sequence = self.controller.observation_sequence()
                self.controller.command("reset", {"seed": episode_seed})
                runtime.reset()
                self._episode_started_at = time.monotonic()
                self._update(
                    current_episode=episode_index + 1,
                    current_seed=episode_seed,
                    phase="running",
                    episode_elapsed_seconds=0.0,
                    last_outcome=None,
                )

                outcome = "timeout"
                while not self._stop_event.is_set():
                    status = self.controller.status()
                    if status.get("error"):
                        raise RuntimeError(str(status["error"]))
                    if status.get("task_success"):
                        outcome = "success"
                        break
                    timeout_seconds = float(
                        status.get("episode_timeout_seconds") or 0.0
                    )
                    if timeout_seconds <= 0:
                        raise RuntimeError(
                            "Selected task has no positive episode timeout"
                        )
                    elapsed = time.monotonic() - self._episode_started_at
                    if elapsed >= timeout_seconds:
                        break

                    try:
                        sequence, observation = (
                            self.controller.wait_for_policy_observation(
                                after_sequence=previous_sequence,
                                timeout=min(
                                    5.0,
                                    max(0.1, timeout_seconds - elapsed),
                                ),
                            )
                        )
                    except TimeoutError:
                        if self._stop_event.is_set():
                            break
                        raise
                    previous_sequence = sequence
                    policy_observation = {
                        **observation,
                        "observation.state": simulation_to_real_motor_positions(
                            observation["observation.state"]
                        ),
                    }
                    inference_started = time.perf_counter()
                    calibrated_targets = np.asarray(
                        runtime.predict(policy_observation), dtype=np.float64
                    )
                    inference_seconds = time.perf_counter() - inference_started
                    if calibrated_targets.shape != (len(JOINT_NAMES),):
                        raise ValueError(
                            "Policy predicted shape "
                            f"{calibrated_targets.shape}; expected "
                            f"({len(JOINT_NAMES)},)"
                        )
                    if not np.all(np.isfinite(calibrated_targets)):
                        raise ValueError("Policy predicted non-finite joint targets")
                    targets = real_to_simulation_motor_positions(calibrated_targets)
                    self.controller.set_policy_joint_targets(targets)
                    action_count += 1
                    inference_total += inference_seconds
                    self._update(
                        last_inference_ms=round(inference_seconds * 1_000, 2),
                        average_inference_ms=round(
                            inference_total * 1_000 / action_count, 2
                        ),
                        actions_sent=action_count,
                    )

                if self._stop_event.is_set():
                    break
                duration = time.monotonic() - self._episode_started_at
                completed += 1
                if outcome == "success":
                    successful += 1
                else:
                    timed_out += 1
                result = {
                    "episode": episode_index + 1,
                    "seed": episode_seed,
                    "outcome": outcome,
                    "duration_seconds": round(duration, 2),
                }
                results.append(result)
                self._episode_started_at = None
                self._update(
                    completed_episodes=completed,
                    successful_episodes=successful,
                    timed_out_episodes=timed_out,
                    last_outcome=outcome,
                    results=results,
                    phase="episode_complete",
                )

            self._update(phase="cancelled" if self._stop_event.is_set() else "complete")
        except Exception as exc:  # noqa: BLE001 - expose inference/load failures
            self._update(error=str(exc), phase="failed")
        finally:
            self._episode_started_at = None
            self.controller.end_policy_control()
            self._update(
                running=False,
                completed_episodes=completed,
                successful_episodes=successful,
                timed_out_episodes=timed_out,
                actions_sent=action_count,
                results=results,
                episode_elapsed_seconds=None,
            )

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        return self.status()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=30)
        self.controller.end_policy_control()
