from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from rospin_workshop.controller import WorkshopController
from rospin_workshop.env import HOME_JOINT_POSITIONS, JOINT_NAMES
from rospin_workshop.trajectory.planner import MotionPlan, TrajectoryPlanner
from rospin_workshop.trajectory.program import (
    TrajectoryProgram,
    load_trajectory_program,
)


class TrajectoryExecutionError(RuntimeError):
    pass


class TrajectoryCancelled(TrajectoryExecutionError):
    pass


class EpisodeContext:
    """Blocking, participant-facing API for one deterministic episode."""

    def __init__(
        self,
        controller: WorkshopController,
        planner: TrajectoryPlanner,
        *,
        seed: int,
        stop_event: threading.Event,
        phase_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.controller = controller
        self.planner = planner
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._stop_event = stop_event
        self._phase_callback = phase_callback or (lambda _: None)
        self._period = 1.0 / controller.config.control_hz

    def _check_cancelled(self) -> None:
        if self._stop_event.is_set():
            raise TrajectoryCancelled("Trajectory generation was stopped")

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._check_cancelled()
            self._stop_event.wait(min(0.05, deadline - time.monotonic()))

    def _status(self) -> dict[str, Any]:
        self._check_cancelled()
        status = self.controller.status()
        if status.get("error"):
            raise TrajectoryExecutionError(str(status["error"]))
        return status

    def _current_plan_seed(self) -> np.ndarray:
        status = self._status()
        positions = np.asarray(status["joint_positions"], dtype=np.float64)
        targets = np.asarray(status["joint_targets"], dtype=np.float64)
        if positions.shape != (len(JOINT_NAMES),) or targets.shape != (
            len(JOINT_NAMES),
        ):
            raise TrajectoryExecutionError("Robot joint state is unavailable")
        # Arm IK starts from measured joints. Preserve the commanded gripper
        # target so a force-limited grasp stays latched during arm motion.
        positions[-1] = targets[-1]
        return positions

    @property
    def current_position(self) -> np.ndarray:
        position = np.asarray(self._status()["eef_position"], dtype=np.float64)
        if position.shape != (3,):
            raise TrajectoryExecutionError("End-effector position is unavailable")
        return position

    @property
    def task_success(self) -> bool:
        return bool(self._status()["task_success"])

    def object_position(self, name: str) -> np.ndarray:
        objects = self._status().get("task_objects", {})
        if name not in objects:
            raise KeyError(f"Task has no object named {name!r}")
        return np.asarray(objects[name]["position"], dtype=np.float64)

    def _execute_plan(
        self,
        plan: MotionPlan,
        *,
        name: str,
        allow_contact: bool = False,
    ) -> None:
        self._phase_callback(name)
        if not plan.joint_targets:
            return
        for target in plan.joint_targets:
            self._check_cancelled()
            self.controller.set_trajectory_joint_targets(target)
            self._sleep(self._period)

        final_target = plan.joint_targets[-1]
        deadline = time.monotonic() + max(2.0, len(plan.joint_targets) * self._period)
        while time.monotonic() < deadline:
            status = self._status()
            positions = np.asarray(status["joint_positions"], dtype=np.float64)
            velocities = np.asarray(status["joint_velocities"], dtype=np.float64)
            arm_at_target = (
                np.max(np.abs(positions[:-1] - final_target[:-1])) < 0.025
            )
            arm_still = np.max(np.abs(velocities[:-1])) < 0.15
            if arm_still and (arm_at_target or allow_contact):
                break
            self._sleep(self._period)
        else:
            raise TrajectoryExecutionError(f"Phase {name!r} did not settle")

        if plan.target_position is not None:
            error = float(np.linalg.norm(self.current_position - plan.target_position))
            allowed_error = 0.025 if allow_contact else 0.012
            if error > allowed_error:
                raise TrajectoryExecutionError(
                    f"Phase {name!r} missed its Cartesian target by {error:.4f} m"
                )

    def move_to(
        self,
        position: Sequence[float] | np.ndarray,
        *,
        speed: float = 0.06,
        safe_height: float = 0.84,
        name: str = "move_to",
    ) -> None:
        """Move via a vertical-horizontal-vertical tabletop-safe path."""

        plan = self.planner.plan_safe_move(
            self._current_plan_seed(),
            np.asarray(position, dtype=np.float64),
            speed=speed,
            safe_height=safe_height,
        )
        self._execute_plan(plan, name=name)

    def move_linear(
        self,
        position: Sequence[float] | np.ndarray,
        *,
        speed: float = 0.035,
        allow_contact: bool = False,
        name: str = "move_linear",
    ) -> None:
        """Move the EEF along a straight Cartesian segment."""

        plan = self.planner.plan_linear(
            self._current_plan_seed(),
            np.asarray(position, dtype=np.float64),
            speed=speed,
        )
        self._execute_plan(plan, name=name, allow_contact=allow_contact)

    def move_relative(
        self,
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        speed: float = 0.04,
        name: str = "move_relative",
    ) -> None:
        offset = np.asarray([x, y, z], dtype=np.float64)
        if not np.all(np.isfinite(offset)):
            raise ValueError("Relative movement must contain finite values")
        self.move_linear(self.current_position + offset, speed=speed, name=name)

    def move_joints(
        self,
        positions: Sequence[float] | np.ndarray,
        *,
        speed: float = 0.7,
        name: str = "move_joints",
    ) -> None:
        plan = self.planner.plan_joints(
            self._current_plan_seed(),
            np.asarray(positions, dtype=np.float64),
            speed=speed,
        )
        self._execute_plan(plan, name=name)

    def move_home(
        self,
        *,
        speed: float = 0.7,
        preserve_gripper: bool = True,
        name: str = "return_home",
    ) -> None:
        target = HOME_JOINT_POSITIONS.copy()
        if preserve_gripper:
            target[-1] = self._current_plan_seed()[-1]
        self.move_joints(target, speed=speed, name=name)

    def _command_gripper(
        self,
        target_position: float,
        *,
        until_contact: bool,
        timeout: float,
        name: str,
    ) -> None:
        self._phase_callback(name)
        target = np.asarray(self._status()["joint_targets"], dtype=np.float64)
        target[-1] = target_position
        self.controller.set_trajectory_joint_targets(target)
        samples: deque[tuple[float, float]] = deque()
        started = time.monotonic()
        deadline = started + timeout
        while time.monotonic() < deadline:
            status = self._status()
            now = time.monotonic()
            position = float(status["joint_positions"][-1])
            if abs(position - target_position) < 0.06:
                return
            samples.append((now, position))
            while samples and now - samples[0][0] > 0.30:
                samples.popleft()
            if (
                until_contact
                and now - started >= 0.5
                and len(samples) >= 4
                and np.ptp([sample[1] for sample in samples]) < 0.002
            ):
                return
            self._sleep(0.05)
        qualifier = " or contact" if until_contact else ""
        raise TrajectoryExecutionError(
            f"Phase {name!r} did not reach its target{qualifier}"
        )

    def open_gripper(self, *, timeout: float = 6.0) -> None:
        self._command_gripper(
            float(self.planner.gripper_range[1]),
            until_contact=False,
            timeout=timeout,
            name="open_gripper",
        )

    def close_gripper(
        self,
        *,
        until_contact: bool = True,
        timeout: float = 4.0,
    ) -> None:
        self._command_gripper(
            float(self.planner.gripper_range[0]),
            until_contact=until_contact,
            timeout=timeout,
            name="close_gripper",
        )

    def wait(self, seconds: float, *, name: str = "wait") -> None:
        if not 0 <= seconds <= 30:
            raise ValueError("Wait duration must be between 0 and 30 seconds")
        self._phase_callback(name)
        self._sleep(seconds)

    def wait_until_settled(
        self,
        object_name: str,
        *,
        timeout: float = 3.0,
        linear_speed: float = 0.03,
        angular_speed: float = 0.3,
    ) -> None:
        self._phase_callback(f"wait_for_{object_name}_to_settle")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._status().get("task_objects", {}).get(object_name)
            if state is None:
                raise KeyError(f"Task has no object named {object_name!r}")
            if (
                np.linalg.norm(state["linear_velocity"]) <= linear_speed
                and np.linalg.norm(state["angular_velocity"]) <= angular_speed
            ):
                return
            self._sleep(0.05)
        raise TrajectoryExecutionError(f"Object {object_name!r} did not settle")

    def assert_condition(self, condition: bool, message: str) -> None:
        if not condition:
            raise TrajectoryExecutionError(message)


class TrajectoryManager:
    """Load participant programs and execute preview or recorded batches."""

    def __init__(self, controller: WorkshopController, trajectories_root: Path) -> None:
        self.controller = controller
        self.trajectories_root = Path(trajectories_root)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status: dict[str, Any] = {
            "running": False,
            "program": None,
            "task_id": None,
            "preview": False,
            "requested_episodes": 0,
            "completed_episodes": 0,
            "saved_episodes": 0,
            "discarded_episodes": 0,
            "current_seed": None,
            "phase": None,
            "dataset_path": None,
            "task_success": False,
            "error": None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._status = {**self._status, **values}

    def start(
        self,
        *,
        program_path: str,
        episodes: int,
        seed: int,
        preview: bool,
        dataset_name: str,
        preflight: bool,
    ) -> dict[str, Any]:
        if not 1 <= episodes <= 10_000:
            raise ValueError("episodes must be between 1 and 10000")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A trajectory program is already running")
        program = load_trajectory_program(program_path, self.trajectories_root)
        self.controller.select_task(program.task_id)
        self._stop_event = threading.Event()
        self._status = {
            "running": True,
            "program": program_path,
            "task_id": program.task_id,
            "preview": preview,
            "requested_episodes": 1 if preview else episodes,
            "completed_episodes": 0,
            "saved_episodes": 0,
            "discarded_episodes": 0,
            "current_seed": seed,
            "phase": "starting",
            "dataset_path": None,
            "task_success": False,
            "error": None,
        }
        self._thread = threading.Thread(
            target=self._run,
            kwargs={
                "program": program,
                "episodes": episodes,
                "seed": seed,
                "preview": preview,
                "dataset_name": dataset_name,
                "preflight": preflight,
            },
            name="trajectory-runner",
            daemon=True,
        )
        self._thread.start()
        return self.status()

    def _execute_once(self, program: TrajectoryProgram, seed: int) -> bool:
        planner = TrajectoryPlanner(control_hz=self.controller.config.control_hz)
        self.controller.begin_trajectory_control()
        try:
            context = EpisodeContext(
                self.controller,
                planner,
                seed=seed,
                stop_event=self._stop_event,
                phase_callback=lambda phase: self._update(phase=phase),
            )
            program.function(context)
        finally:
            self.controller.end_trajectory_control()
            planner.close()

        hold_seconds = float(
            self.controller.status().get("task_success_hold_seconds") or 0.0
        )
        deadline = time.monotonic() + hold_seconds + 2.0
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                raise TrajectoryCancelled("Trajectory generation was stopped")
            if self.controller.status().get("task_success"):
                return True
            self._stop_event.wait(0.05)
        return bool(self.controller.status().get("task_success"))

    def _run(
        self,
        *,
        program: TrajectoryProgram,
        episodes: int,
        seed: int,
        preview: bool,
        dataset_name: str,
        preflight: bool,
    ) -> None:
        completed = 0
        saved = 0
        discarded = 0
        try:
            if preview:
                self.controller.command("reset", {"seed": seed})
                success = self._execute_once(program, seed)
                completed = 1
                self._update(
                    completed_episodes=completed,
                    task_success=success,
                    phase="preview_complete",
                )
                return

            self.controller.command("new_dataset", {})
            for index in range(episodes):
                self._check_stopped()
                episode_seed = seed + index
                self._update(
                    current_seed=episode_seed,
                    phase="preflight" if preflight else "recording_setup",
                )
                if preflight:
                    self.controller.command("reset", {"seed": episode_seed})
                    if not self._execute_once(program, episode_seed):
                        completed += 1
                        discarded += 1
                        self._update(
                            completed_episodes=completed,
                            discarded_episodes=discarded,
                            task_success=False,
                        )
                        continue

                self.controller.command(
                    "start_recording",
                    {
                        "dataset_name": dataset_name,
                        "seed": episode_seed,
                        "scripted": True,
                    },
                )
                try:
                    self._update(phase="recording")
                    success = self._execute_once(program, episode_seed)
                except Exception:
                    if self.controller.status().get("recording"):
                        self.controller.command(
                            "discard_recording", {"outcome": "trajectory_error"}
                        )
                    raise
                if success:
                    self.controller.command(
                        "stop_recording", {"outcome": "trajectory_success"}
                    )
                    saved += 1
                else:
                    self.controller.command(
                        "discard_recording", {"outcome": "trajectory_failure"}
                    )
                    discarded += 1
                completed += 1
                self._update(
                    completed_episodes=completed,
                    saved_episodes=saved,
                    discarded_episodes=discarded,
                    task_success=success,
                )

            if saved:
                self.controller.command("finish_dataset", {})
            self._update(
                dataset_path=self.controller.status().get("dataset_path"),
                phase="complete",
            )
        except TrajectoryCancelled as exc:
            if self.controller.status().get("recording"):
                self.controller.command(
                    "discard_recording", {"outcome": "trajectory_cancelled"}
                )
            self._update(error=str(exc), phase="cancelled")
        except Exception as exc:  # noqa: BLE001 - surface participant errors
            self._update(error=str(exc), phase="failed")
        finally:
            self.controller.end_trajectory_control()
            self._update(
                running=False,
                completed_episodes=completed,
                saved_episodes=saved,
                discarded_episodes=discarded,
                dataset_path=self.controller.status().get("dataset_path"),
            )

    def _check_stopped(self) -> None:
        if self._stop_event.is_set():
            raise TrajectoryCancelled("Trajectory generation was stopped")

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        return self.status()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=30)
