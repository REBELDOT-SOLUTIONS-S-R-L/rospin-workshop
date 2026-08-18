from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.env import (
    ACTION_NAMES,
    JOINT_NAMES,
    SO101WorkshopEnv,
)
from rospin_workshop.recorder import LeRobotV3Recorder
from rospin_workshop.tasks import TaskDefinition, TaskRegistry

LOGGER = logging.getLogger(__name__)

PERSPECTIVE_MAX_RENDER_WIDTH = 640
PERSPECTIVE_RECORDING_HZ = 5.0
DATASET_CAMERA_NAMES = ("wrist",)
RENDER_CAMERA_NAMES = DATASET_CAMERA_NAMES + ("perspective",)


class TaskSessionConflictError(RuntimeError):
    pass


PERSPECTIVE_DEFAULT_POSITION = np.array([0.2, 1.0, 1.38], dtype=np.float64)
PERSPECTIVE_DEFAULT_FORWARD = np.array(
    [-0.25900871575, -0.835511986289, -0.484596952047],
    dtype=np.float64,
)
PERSPECTIVE_DEFAULT_DISTANCE = 1.197
PERSPECTIVE_DEFAULT_LOOKAT = (
    PERSPECTIVE_DEFAULT_POSITION
    + PERSPECTIVE_DEFAULT_FORWARD * PERSPECTIVE_DEFAULT_DISTANCE
)

KEY_ACTIONS: dict[str, np.ndarray] = {
    "w": np.array([0, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "s": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "a": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "d": np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "q": np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "e": np.array([0, 0, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "u": np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    "o": np.array([0, 0, 0, -1, 0, 0, 0, 0, 0], dtype=np.float32),
    "r": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    "f": np.array([0, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32),
    "t": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
    "g": np.array([0, 0, 0, 0, 0, -1, 0, 0, 0], dtype=np.float32),
    "i": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
    "k": np.array([0, 0, 0, 0, 0, 0, -1, 0, 0], dtype=np.float32),
    "j": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
    "l": np.array([0, 0, 0, 0, 0, 0, 0, -1, 0], dtype=np.float32),
    "[": np.array([0, 0, 0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
    "]": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
}


class PerspectiveCamera:
    """Thread-safe-by-owner orbit-camera state used by the web viewer."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        offset = PERSPECTIVE_DEFAULT_POSITION - PERSPECTIVE_DEFAULT_LOOKAT
        self.lookat = PERSPECTIVE_DEFAULT_LOOKAT.copy()
        self.distance = float(np.linalg.norm(offset))
        self.azimuth = float(np.arctan2(offset[0], offset[1]))
        self.elevation = float(np.arcsin(offset[2] / self.distance))

    @staticmethod
    def _delta(payload: dict[str, Any], name: str) -> float:
        value = float(payload.get(name, 0.0))
        if not np.isfinite(value):
            raise ValueError(f"Camera {name} must be finite")
        return float(np.clip(value, -500.0, 500.0))

    def position(self) -> np.ndarray:
        horizontal = self.distance * np.cos(self.elevation)
        return self.lookat + np.array(
            [
                horizontal * np.sin(self.azimuth),
                horizontal * np.cos(self.azimuth),
                self.distance * np.sin(self.elevation),
            ],
            dtype=np.float64,
        )

    def apply(self, action: str, payload: dict[str, Any]) -> None:
        if action == "orbit":
            dx = self._delta(payload, "dx")
            dy = self._delta(payload, "dy")
            self.azimuth = (
                (self.azimuth - dx * 0.006 + np.pi) % (2 * np.pi)
            ) - np.pi
            self.elevation = float(
                np.clip(
                    self.elevation - dy * 0.006,
                    np.deg2rad(5.0),
                    np.deg2rad(85.0),
                )
            )
        elif action == "pan":
            dx = self._delta(payload, "dx")
            dy = self._delta(payload, "dy")
            position = self.position()
            forward = self.lookat - position
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            scale = self.distance * 0.0015
            self.lookat += (-right * dx + up * dy) * scale
            self.lookat = np.clip(
                self.lookat,
                np.array([-2.0, -2.0, 0.0]),
                np.array([2.0, 2.0, 2.5]),
            )
        elif action == "zoom":
            delta = self._delta(payload, "delta")
            self.distance = float(
                np.clip(
                    self.distance * np.exp(delta * 0.0015),
                    0.3,
                    3.0,
                )
            )
        elif action == "reset":
            self.reset()
        else:
            raise ValueError(f"Unknown perspective camera action: {action}")

    def view(self) -> tuple[np.ndarray, np.ndarray]:
        return self.position(), self.lookat.copy()

    def status(self) -> dict[str, Any]:
        return {
            "azimuth_degrees": round(float(np.rad2deg(self.azimuth)), 1),
            "elevation_degrees": round(float(np.rad2deg(self.elevation)), 1),
            "distance": round(self.distance, 3),
            "lookat": self.lookat.round(3).tolist(),
        }


class WorkshopController:
    def __init__(
        self,
        config: RuntimeConfig,
        *,
        task_registry: TaskRegistry | None = None,
    ) -> None:
        if config.camera_hz <= 0:
            raise ValueError("camera_hz must be positive")
        if config.camera_hz > config.control_hz:
            raise ValueError("camera_hz cannot exceed control_hz")
        self.config = config
        self.task_registry = task_registry or TaskRegistry(config.tasks_root)
        self._selected_task: TaskDefinition | None = None
        # Each MuJoCo instance has one owner: the control thread advances
        # physics, while one worker per camera owns an independent OSMesa
        # context. Camera work is scheduled independently so the viewer cannot
        # delay the wrist frames handed to the recorder.
        self.env: SO101WorkshopEnv | None = None
        self._observation: dict[str, np.ndarray] | None = None
        self._jpeg_frames: dict[str, bytes] = {}
        self._keys: set[str] = set()
        self._gripper_command = 0.0
        self._control_mode_changed = False
        self._active_control_source = "keyboard"
        self._joint_control_source: str | None = None
        self._joint_control_targets: np.ndarray | None = None
        self._perspective_camera = PerspectiveCamera()
        self._recorder: LeRobotV3Recorder | None = None
        self._last_dataset_path: Path | None = None
        self._episode_started_at: float | None = None
        self._scripted_recording = False
        self._last_episode_outcome: str | None = None
        self._message = "Waiting for task selection"
        self._error: str | None = None
        self._lock = threading.RLock()
        self._observation_condition = threading.Condition(self._lock)
        self._observation_sequence = 0
        self._status_cache: dict[str, Any] = {
            "message": "Waiting for task selection",
            "error": None,
            "recording": False,
            "finalized": False,
            "frames_in_episode": 0,
            "episodes": 0,
            "dataset_path": None,
            "task": None,
            "task_ready": False,
            "task_id": None,
            "task_title": None,
            "task_instruction": None,
            "task_success": False,
            "task_success_progress": 0.0,
            "task_success_hold_seconds": None,
            "episode_elapsed_seconds": None,
            "episode_timeout_seconds": None,
            "last_episode_outcome": None,
            "keys": [],
            "active_control_source": "keyboard",
            "trajectory_active": False,
            "policy_active": False,
            "scripted_recording": False,
            "eef_position": None,
            "eef_orientation": None,
            "joint_positions": None,
            "joint_velocities": None,
            "joint_targets": None,
            "task_objects": {},
            "gripper_force_nm": None,
            "sim_time": None,
            "control_hz": self.config.control_hz,
            "camera_hz": self.config.camera_hz,
            "image_size": [self.config.image_width, self.config.image_height],
            "perspective_camera": self._perspective_camera.status(),
        }
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._render_requests: dict[
            str,
            queue.Queue[
                tuple[
                    int,
                    dict[str, np.ndarray | float],
                    np.ndarray,
                    int | None,
                    int,
                    tuple[np.ndarray, np.ndarray] | None,
                ]
            ],
        ] = {camera: queue.Queue(maxsize=1) for camera in RENDER_CAMERA_NAMES}
        self._render_results: queue.Queue[
            tuple[
                int,
                str,
                dict[str, np.ndarray] | None,
                np.ndarray | None,
                int | None,
                int,
                BaseException | None,
            ]
        ] = queue.Queue()
        # OSMesa is not safe under concurrent rendering in this workload. Keep
        # independent camera contexts, but never execute native Mesa calls at
        # the same time (the previous concurrent path eventually segfaulted in
        # libOSMesa during long synthetic batches).
        self._render_lock = threading.Lock()
        self._render_inflight: set[str] = set()
        self._render_sequence = 0
        self._pending_renders: dict[int, dict[str, Any]] = {}
        self._render_requested = threading.Event()
        self._render_epoch = 0
        self._recording_generation = 0
        self._next_recording_perspective_at = 0.0
        self._commands: queue.Queue[
            tuple[str, dict[str, Any], threading.Event, dict[str, Any]]
        ] = queue.Queue()
        self._thread = threading.Thread(
            target=self._control_loop, name="so101-control", daemon=True
        )
        self._render_threads = [
            threading.Thread(
                target=self._render_loop,
                args=(camera,),
                name=f"so101-render-{camera}",
                daemon=True,
            )
            for camera in RENDER_CAMERA_NAMES
        ]

    def start(self) -> None:
        self._thread.start()
        if not self._ready_event.wait(timeout=30):
            raise RuntimeError("Simulation startup timed out")
        if self._error is not None:
            raise RuntimeError(f"Simulation startup failed: {self._error}")

    def set_key(self, key: str, pressed: bool) -> None:
        key = key.lower()
        if key not in KEY_ACTIONS:
            return
        with self._lock:
            if self._joint_control_source is not None:
                return
            if pressed:
                self._keys.add(key)
                if key == "[":
                    self._gripper_command = -1.0
                elif key == "]":
                    self._gripper_command = 1.0
            else:
                self._keys.discard(key)
            self._status_cache = {
                **self._status_cache,
                "keys": sorted(self._keys),
            }

    def clear_keys(self) -> None:
        with self._lock:
            self._keys.clear()
            self._status_cache = {**self._status_cache, "keys": []}

    def _begin_joint_control(self, source: str) -> None:
        if source not in {"trajectory", "policy"}:
            raise ValueError(f"Unsupported joint control source: {source}")
        with self._lock:
            if self._joint_control_source is not None:
                raise RuntimeError(
                    f"{self._joint_control_source.title()} control is already active"
                )
            self._joint_control_source = source
            self._joint_control_targets = None
            self._control_mode_changed = True
            self._keys.clear()
            self._gripper_command = 0.0

    def _set_joint_control_targets(self, source: str, targets: np.ndarray) -> None:
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (len(JOINT_NAMES),):
            raise ValueError(f"Joint target must contain {len(JOINT_NAMES)} joints")
        if not np.all(np.isfinite(targets)):
            raise ValueError("Joint targets must all be finite")
        with self._lock:
            if self._joint_control_source != source:
                raise RuntimeError(f"{source.title()} control is not active")
            self._joint_control_targets = targets.copy()

    def _end_joint_control(self, source: str) -> None:
        with self._lock:
            if self._joint_control_source != source:
                return
            self._joint_control_source = None
            self._joint_control_targets = None
            self._control_mode_changed = True

    def begin_trajectory_control(self) -> None:
        """Give a trajectory worker exclusive access to the simulated robot."""

        self._begin_joint_control("trajectory")

    def set_trajectory_joint_targets(self, targets: np.ndarray) -> None:
        self._set_joint_control_targets("trajectory", targets)

    def end_trajectory_control(self) -> None:
        """Release trajectory ownership and hold the current simulated pose."""

        self._end_joint_control("trajectory")

    def begin_policy_control(self) -> None:
        """Give an inference worker exclusive absolute-joint control."""

        self._begin_joint_control("policy")

    def set_policy_joint_targets(self, targets: np.ndarray) -> None:
        self._set_joint_control_targets("policy", targets)

    def end_policy_control(self) -> None:
        """Release policy ownership and hold the current simulated pose."""

        self._end_joint_control("policy")

    def control_perspective_camera(
        self,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        """Update the viewer-only orbit camera without affecting recording."""

        with self._lock:
            self._perspective_camera.apply(action, payload)
            self._status_cache = {
                **self._status_cache,
                "perspective_camera": self._perspective_camera.status(),
            }

    def _current_action(self) -> np.ndarray:
        action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        with self._lock:
            keys = tuple(self._keys)
            gripper_command = self._gripper_command
        for key in keys:
            action += KEY_ACTIONS[key]
        action[-1] = gripper_command
        return np.clip(action, -1.0, 1.0)

    def _step_control(self) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Simulation environment is not ready")
        with self._lock:
            joint_control_source = self._joint_control_source
            joint_control_targets = (
                self._joint_control_targets.copy()
                if self._joint_control_targets is not None
                else None
            )
            control_mode_changed = self._control_mode_changed
            self._control_mode_changed = False

        if control_mode_changed:
            self.env.hold_current_pose()

        if joint_control_source is not None:
            if joint_control_targets is None:
                self._observation, _, _, _, _ = self.env.step_dynamics(
                    np.zeros(len(ACTION_NAMES), dtype=np.float32)
                )
            else:
                self._observation, _, _, _, _ = self.env.step_joint_targets(
                    joint_control_targets
                )
            source = joint_control_source
        else:
            action = self._current_action()
            self._observation, _, _, _, _ = self.env.step_dynamics(action)
            source = "keyboard"

        with self._lock:
            self._active_control_source = source
        # The dataset action is the six absolute motor-angle targets sent to
        # the simulated follower. Keyboard and trajectory control both
        # converge to this representation before recording; the measured
        # angles remain in observation.state.
        return self.env.data.ctrl.astype(np.float32, copy=True)

    @staticmethod
    def _encode_image(camera: str, rgb: np.ndarray) -> bytes:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 86])
        if not ok:
            raise RuntimeError(f"Could not encode {camera} camera")
        return encoded.tobytes()

    def _control_loop(self) -> None:
        try:
            self._ready_event.set()

            period = 1.0 / self.config.control_hz
            camera_period = 1.0 / self.config.camera_hz
            next_tick = time.monotonic()
            next_camera_tick = next_tick + camera_period
            while not self._stop_event.is_set():
                self._process_commands()
                if self.env is None:
                    with self._lock:
                        self._refresh_status_locked()
                    self._stop_event.wait(0.01)
                    next_tick = time.monotonic()
                    next_camera_tick = next_tick + camera_period
                    continue
                action = self._step_control()
                self._collect_render_results()
                self._update_episode_lifecycle()

                now = time.monotonic()
                render_requested = self._render_requested.is_set()
                if render_requested:
                    if self._submit_renders(action):
                        self._render_requested.clear()
                        next_camera_tick = now + camera_period
                elif now >= next_camera_tick:
                    self._submit_renders(action)
                    next_camera_tick += camera_period
                    if next_camera_tick <= now:
                        next_camera_tick = now + camera_period

                with self._lock:
                    self._refresh_status_locked()
                next_tick += period
                wait_time = next_tick - time.monotonic()
                if wait_time > 0:
                    self._stop_event.wait(wait_time)
                else:
                    next_tick = time.monotonic()
        except Exception as exc:  # keep the server observable if the loop fails
            LOGGER.exception("Control loop failed")
            with self._lock:
                self._error = str(exc)
                self._message = "Control loop error"
                self._refresh_status_locked()
            self._stop_event.set()
            self._ready_event.set()
        finally:
            self._stop_event.set()
            for render_thread in self._render_threads:
                if render_thread.is_alive():
                    render_thread.join(timeout=30)
            if self._recorder is not None and not self._recorder.finalized:
                try:
                    self._last_dataset_path = self._recorder.finalize()
                except Exception:
                    LOGGER.exception("Could not finalize dataset during shutdown")
            if self.env is not None:
                self.env.close()

    def _render_loop(self, camera: str) -> None:
        render_env: SO101WorkshopEnv | None = None
        try:
            if self._selected_task is None:
                raise RuntimeError("Render worker started without a selected task")
            image_width, image_height = self._camera_render_size(camera)
            with self._render_lock:
                render_env = SO101WorkshopEnv(
                    task=self._selected_task,
                    render_mode="rgb_array",
                    image_width=image_width,
                    image_height=image_height,
                    control_hz=self.config.control_hz,
                )
            while not self._stop_event.is_set():
                try:
                    (
                        sequence,
                        snapshot,
                        action,
                        recording_generation,
                        epoch,
                        camera_view,
                    ) = self._render_requests[camera].get(timeout=0.1)
                except queue.Empty:
                    continue
                with self._render_lock:
                    render_env.restore_simulation_snapshot(snapshot)
                    if camera_view is not None:
                        position, lookat = camera_view
                        render_env.set_camera_lookat(camera, position, lookat)
                    observation = render_env.capture_camera_observation(camera)
                self._render_results.put(
                    (
                        sequence,
                        camera,
                        observation,
                        action,
                        recording_generation,
                        epoch,
                        None,
                    )
                )
        except BaseException as exc:
            LOGGER.exception("%s camera render loop failed", camera)
            self._render_results.put((-1, camera, None, None, None, -1, exc))
        finally:
            if render_env is not None:
                with self._render_lock:
                    render_env.close()

    def _camera_render_size(self, camera: str) -> tuple[int, int]:
        """Keep the recorded wrist feed native while bounding viewer render cost."""

        if (
            camera != "perspective"
            or self.config.image_width <= PERSPECTIVE_MAX_RENDER_WIDTH
        ):
            return self.config.image_width, self.config.image_height
        scale = PERSPECTIVE_MAX_RENDER_WIDTH / self.config.image_width
        return PERSPECTIVE_MAX_RENDER_WIDTH, max(
            1, round(self.config.image_height * scale)
        )

    def _submit_renders(self, action: np.ndarray) -> bool:
        """Submit a wrist dataset frame and an independent viewer frame."""

        dataset_submitted = self._submit_camera_group(DATASET_CAMERA_NAMES, action)
        # Outside recording, both viewer feeds run at the configured camera
        # rate. During recording, perspective work is scheduled only after a
        # wrist frame completes so serialized OSMesa rendering cannot take
        # priority over the dataset camera.
        if self._recorder is None or not self._recorder.recording:
            self._submit_camera_group(("perspective",), action)
        return dataset_submitted

    def _submit_recording_perspective_if_due(self, action: np.ndarray) -> None:
        """Refresh the viewer during recording without adding a dataset frame."""

        if self._recorder is None or not self._recorder.recording:
            return
        now = time.monotonic()
        if now < self._next_recording_perspective_at:
            return
        if self._submit_camera_group(("perspective",), action):
            self._next_recording_perspective_at = (
                now + 1.0 / PERSPECTIVE_RECORDING_HZ
            )

    def _submit_camera_group(
        self,
        cameras: tuple[str, ...],
        action: np.ndarray,
    ) -> bool:
        """Render a camera group from one state/action snapshot."""

        if self.env is None or any(
            camera in self._render_inflight for camera in cameras
        ):
            return False
        recording_generation = (
            self._recording_generation
            if self._recorder is not None and self._recorder.recording
            else None
        )
        self._render_sequence += 1
        sequence = self._render_sequence
        snapshot = self.env.simulation_snapshot()
        action_copy = np.asarray(action, dtype=np.float32).copy()
        self._pending_renders[sequence] = {
            "observation": {},
            "action": action_copy,
            "recording_generation": recording_generation,
            "epoch": self._render_epoch,
            "remaining_cameras": set(cameras),
            "record_dataset": set(cameras) == set(DATASET_CAMERA_NAMES),
        }
        with self._lock:
            perspective_view = self._perspective_camera.view()
        for camera in cameras:
            camera_view = perspective_view if camera == "perspective" else None
            self._render_requests[camera].put_nowait(
                (
                    sequence,
                    snapshot,
                    action_copy,
                    recording_generation,
                    self._render_epoch,
                    camera_view,
                )
            )
            self._render_inflight.add(camera)
        return True

    def _collect_render_results(self) -> None:
        while True:
            try:
                result = self._render_results.get_nowait()
            except queue.Empty:
                return
            self._consume_render_result(result)

    def _consume_render_result(
        self,
        result: tuple[
            int,
            str,
            dict[str, np.ndarray] | None,
            np.ndarray | None,
            int | None,
            int,
            BaseException | None,
        ],
    ) -> None:
        sequence, camera, observation, action, recording_generation, epoch, error = (
            result
        )
        if error is not None:
            raise RuntimeError("Camera rendering failed") from error
        self._render_inflight.discard(camera)
        pending = self._pending_renders.get(sequence)
        if pending is None or observation is None or action is None:
            return
        if epoch == self._render_epoch:
            for name, value in observation.items():
                if not name.startswith("observation.images."):
                    pending["observation"].setdefault(name, value)
            image_key = f"observation.images.{camera}"
            pending["observation"][image_key] = observation[image_key]
            jpeg = self._encode_image(camera, observation[image_key])
            with self._lock:
                self._jpeg_frames[camera] = jpeg

        pending["remaining_cameras"].discard(camera)
        if pending["remaining_cameras"]:
            return
        self._pending_renders.pop(sequence)
        camera_observation = pending["observation"]
        if epoch != self._render_epoch:
            return
        if (
            pending["record_dataset"]
            and self._recorder is not None
            and self._recorder.recording
            and recording_generation == self._recording_generation
        ):
            self._recorder.add_frame(camera_observation, pending["action"])
        with self._lock:
            self._observation = camera_observation
            self._observation_sequence += 1
            self._observation_condition.notify_all()
            self._refresh_status_locked()
        if camera == "wrist":
            self._submit_recording_perspective_if_due(pending["action"])

    def _process_commands(self) -> None:
        while True:
            try:
                command, payload, event, response = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                response["result"] = self._execute_command(command, payload)
            except Exception as exc:  # noqa: BLE001 - return command errors to caller
                response["error"] = exc
            finally:
                with self._lock:
                    self._refresh_status_locked()
                event.set()

    def camera_jpeg(self, camera: str) -> bytes:
        if camera not in ("wrist", "perspective"):
            raise KeyError(camera)
        with self._lock:
            if camera not in self._jpeg_frames:
                raise RuntimeError("Camera is not ready")
            return self._jpeg_frames[camera]

    def observation_sequence(self) -> int:
        with self._lock:
            return self._observation_sequence

    def wait_for_policy_observation(
        self,
        *,
        after_sequence: int,
        timeout: float = 5.0,
    ) -> tuple[int, dict[str, np.ndarray]]:
        """Wait for a fresh synchronized wrist image and measured joint state."""

        deadline = time.monotonic() + timeout
        with self._observation_condition:
            while True:
                observation = self._observation
                if (
                    self._observation_sequence > after_sequence
                    and observation is not None
                    and "observation.state" in observation
                    and "observation.images.wrist" in observation
                ):
                    return self._observation_sequence, {
                        "observation.state": np.asarray(
                            observation["observation.state"], dtype=np.float32
                        ).copy(),
                        "observation.images.wrist": np.asarray(
                            observation["observation.images.wrist"], dtype=np.uint8
                        ).copy(),
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a fresh wrist observation")
                self._observation_condition.wait(remaining)

    def command(self, command: str, payload: dict[str, Any]) -> Any:
        if not self._thread.is_alive():
            raise RuntimeError("Simulation control loop is not running")
        event = threading.Event()
        response: dict[str, Any] = {}
        self._commands.put((command, payload, event, response))
        if not event.wait(timeout=120):
            raise TimeoutError(f"Command timed out: {command}")
        if "error" in response:
            raise response["error"]
        return response.get("result")

    def tasks(self) -> list[dict[str, Any]]:
        return self.task_registry.list()

    def select_task(self, task_id: str) -> None:
        task = self.task_registry.get(task_id)
        self.command("select_task", {"task": task})

    def _initialize_task(self, task: TaskDefinition) -> None:
        if self.env is not None:
            raise TaskSessionConflictError(
                f"Task session is already locked to {self._selected_task.id!r}"
            )
        self._selected_task = task
        try:
            self.env = SO101WorkshopEnv(
                task=task,
                render_mode=None,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
                control_hz=self.config.control_hz,
            )
            self._observation, _ = self.env.reset()
            for render_thread in self._render_threads:
                render_thread.start()
            startup_action = self.env.data.ctrl.astype(np.float32, copy=True)
            self._submit_renders(startup_action)
            startup_deadline = time.monotonic() + 30
            while self._render_inflight:
                timeout = startup_deadline - time.monotonic()
                if timeout <= 0:
                    raise TimeoutError("Initial camera rendering timed out")
                self._consume_render_result(self._render_results.get(timeout=timeout))
        except Exception:
            if self.env is not None:
                self.env.close()
            self.env = None
            self._selected_task = None
            raise
        self._message = f"Task ready: {task.title}"

    def _reset_task_environment(self, *, seed: int | None = None) -> None:
        if self.env is None:
            raise RuntimeError("Select a task before resetting the simulation")
        with self._lock:
            self._keys.clear()
            self._gripper_command = 0.0
            if self._joint_control_source is not None:
                self._joint_control_targets = None
        self._observation, _ = self.env.reset(seed=seed)
        self._render_epoch += 1
        self._render_requested.set()

    def _complete_episode(self, *, save: bool, outcome: str) -> int:
        recorder = self._require_recorder(recording=True)
        frames = recorder.stop_episode(save=save)
        self._recording_generation += 1
        self._episode_started_at = None
        self._scripted_recording = False
        self._last_episode_outcome = outcome
        self._reset_task_environment()
        verb = "Saved" if save else "Discarded"
        self._message = f"{verb} episode with {frames} frames ({outcome})"
        return frames

    def _update_episode_lifecycle(self) -> None:
        if (
            self.env is None
            or self._selected_task is None
            or self._recorder is None
            or not self._recorder.recording
            or self._episode_started_at is None
            or self._scripted_recording
        ):
            return
        task_status = self.env.task_status() or {}
        if bool(task_status.get("success")):
            self._complete_episode(save=True, outcome="success")
            return
        if (
            time.monotonic() - self._episode_started_at
            >= self._selected_task.timeout_seconds
        ):
            self._complete_episode(save=False, outcome="timeout")

    def _execute_command(self, command: str, payload: dict[str, Any]) -> Any:
        try:
            self._error = None
            if command == "select_task":
                task = payload.get("task")
                if not isinstance(task, TaskDefinition):
                    raise ValueError("select_task requires a validated task")
                if self._selected_task is not None:
                    if self._selected_task.id != task.id:
                        raise TaskSessionConflictError(
                            "This server is already locked to task "
                            f"{self._selected_task.id!r}; restart it to select another"
                        )
                    self._message = f"Task ready: {task.title}"
                else:
                    self._initialize_task(task)
            elif command == "reset":
                if self._recorder is not None and self._recorder.recording:
                    raise RuntimeError("Stop or discard the recording before reset")
                seed_value = payload.get("seed")
                if seed_value is not None and not isinstance(seed_value, int):
                    raise ValueError("reset seed must be an integer")
                self._reset_task_environment(seed=seed_value)
                self._last_episode_outcome = "reset"
                self._message = "Simulation reset"
            elif command == "start_recording":
                if self._selected_task is None:
                    raise RuntimeError("Select a task before recording")
                if self._recorder is None or self._recorder.finalized:
                    self._recorder = LeRobotV3Recorder(
                        datasets_root=self.config.datasets_root,
                        fps=self.config.camera_hz,
                        image_width=self.config.image_width,
                        image_height=self.config.image_height,
                    )
                seed_value = payload.get("seed")
                if seed_value is not None and not isinstance(seed_value, int):
                    raise ValueError("recording seed must be an integer")
                self._reset_task_environment(seed=seed_value)
                self._recording_generation += 1
                self._next_recording_perspective_at = 0.0
                self._recorder.start_episode(
                    dataset_name=str(payload.get("dataset_name", "so101_session")),
                    task=self._selected_task.dataset_description,
                )
                self._episode_started_at = time.monotonic()
                self._scripted_recording = bool(payload.get("scripted", False))
                self._last_episode_outcome = None
                self._render_requested.set()
                self._message = "Recording episode"
            elif command == "stop_recording":
                self._complete_episode(
                    save=True,
                    outcome=str(payload.get("outcome", "manual_save")),
                )
            elif command == "discard_recording":
                self._complete_episode(
                    save=False,
                    outcome=str(payload.get("outcome", "manual_discard")),
                )
            elif command == "finish_dataset":
                recorder = self._require_recorder(recording=False)
                if recorder.num_episodes < 1 and not recorder.recording:
                    raise RuntimeError("Save at least one episode before finishing")
                self._last_dataset_path = recorder.finalize()
                self._message = f"Dataset finalized: {self._last_dataset_path}"
            elif command == "new_dataset":
                if self._recorder is not None and self._recorder.recording:
                    raise RuntimeError("Stop or discard the active recording first")
                if (
                    self._recorder is not None
                    and not self._recorder.finalized
                    and self._recorder.num_episodes > 0
                ):
                    self._last_dataset_path = self._recorder.finalize()
                self._recorder = None
                self._last_dataset_path = None
                self._message = "Ready for a new dataset"
            elif command == "resume_dataset":
                if self._recorder is not None and self._recorder.recording:
                    raise RuntimeError("Stop or discard the active recording first")
                requested_path = Path(str(payload.get("dataset_path", "")))
                if not requested_path.is_absolute():
                    requested_path = self.config.datasets_root / requested_path
                requested_path = requested_path.resolve()
                datasets_root = self.config.datasets_root.resolve()
                if requested_path.parent != datasets_root:
                    raise ValueError("Resume dataset must be under the datasets root")
                self._recorder = LeRobotV3Recorder(
                    datasets_root=self.config.datasets_root,
                    fps=self.config.camera_hz,
                    image_width=self.config.image_width,
                    image_height=self.config.image_height,
                )
                self._recorder.resume_dataset(requested_path)
                self._last_dataset_path = requested_path
                self._message = f"Resumed dataset: {requested_path}"
            else:
                raise ValueError(f"Unknown command: {command}")
        except Exception as exc:
            self._error = str(exc)
            self._message = "Command failed"
            raise

    def _require_recorder(self, *, recording: bool) -> LeRobotV3Recorder:
        if self._recorder is None:
            raise RuntimeError("No dataset session exists")
        if recording and not self._recorder.recording:
            raise RuntimeError("No episode is being recorded")
        return self._recorder

    def _refresh_status_locked(self) -> None:
        recorder = self._recorder
        dataset_path = (
            recorder.dataset_path
            if recorder is not None and recorder.dataset_path is not None
            else self._last_dataset_path
        )
        selected_task = self._selected_task
        task_status = self.env.task_status() if self.env is not None else None
        episode_elapsed = (
            max(0.0, time.monotonic() - self._episode_started_at)
            if self._episode_started_at is not None
            else None
        )
        self._status_cache = {
            "message": self._message,
            "error": self._error,
            "recording": bool(recorder and recorder.recording),
            "finalized": bool(recorder and recorder.finalized),
            "frames_in_episode": recorder.frames_in_episode if recorder else 0,
            "episodes": recorder.num_episodes if recorder else 0,
            "dataset_path": str(dataset_path) if dataset_path else None,
            "task": recorder.task if recorder else None,
            "task_ready": self.env is not None and selected_task is not None,
            "task_id": selected_task.id if selected_task else None,
            "task_title": selected_task.title if selected_task else None,
            "task_instruction": selected_task.instruction if selected_task else None,
            "task_success": bool(task_status and task_status["success"]),
            "task_success_progress": (
                round(float(task_status["success_progress"]), 3)
                if task_status
                else 0.0
            ),
            "task_success_hold_seconds": (
                selected_task.success_hold_seconds if selected_task else None
            ),
            "episode_elapsed_seconds": (
                round(episode_elapsed, 2) if episode_elapsed is not None else None
            ),
            "episode_timeout_seconds": (
                selected_task.timeout_seconds if selected_task else None
            ),
            "last_episode_outcome": self._last_episode_outcome,
            "keys": sorted(self._keys),
            "active_control_source": self._active_control_source,
            "trajectory_active": self._joint_control_source == "trajectory",
            "policy_active": self._joint_control_source == "policy",
            "scripted_recording": self._scripted_recording,
            "eef_position": (
                self.env.eef_position.round(4).tolist()
                if self.env is not None
                else None
            ),
            "eef_orientation": (
                self.env.eef_orientation.round(4).tolist()
                if self.env is not None
                else None
            ),
            "joint_positions": (
                self.env.joint_positions.round(4).tolist()
                if self.env is not None
                else None
            ),
            "joint_velocities": (
                self.env.joint_velocities.round(4).tolist()
                if self.env is not None
                else None
            ),
            "joint_targets": (
                self.env.data.ctrl.round(4).tolist()
                if self.env is not None
                else None
            ),
            "task_objects": (
                self.env.task_object_states() if self.env is not None else {}
            ),
            "gripper_force_nm": (
                round(float(self.env.data.actuator_force[-1]), 4)
                if self.env is not None
                else None
            ),
            "sim_time": (
                round(float(self.env.data.time), 4) if self.env is not None else None
            ),
            "control_hz": self.config.control_hz,
            "camera_hz": self.config.camera_hz,
            "image_size": [self.config.image_width, self.config.image_height],
            "perspective_camera": self._perspective_camera.status(),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status_cache)

    def close(self) -> None:
        self._stop_event.set()
        with self._observation_condition:
            self._observation_condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=30)
