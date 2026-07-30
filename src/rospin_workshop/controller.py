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
from rospin_workshop.env import ACTION_NAMES, CAMERA_NAMES, SO101WorkshopEnv
from rospin_workshop.recorder import LeRobotV3Recorder

LOGGER = logging.getLogger(__name__)

KEY_ACTIONS: dict[str, np.ndarray] = {
    "w": np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "s": np.array([-1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    "a": np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    "d": np.array([0, -1, 0, 0, 0, 0, 0], dtype=np.float32),
    "q": np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
    "e": np.array([0, 0, -1, 0, 0, 0, 0], dtype=np.float32),
    "i": np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
    "k": np.array([0, 0, 0, 0, -1, 0, 0], dtype=np.float32),
    "j": np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
    "l": np.array([0, 0, 0, 0, 0, -1, 0], dtype=np.float32),
    "u": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
    "o": np.array([0, 0, 0, -1, 0, 0, 0], dtype=np.float32),
    "[": np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
    "]": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
}


class WorkshopController:
    def __init__(self, config: RuntimeConfig) -> None:
        if config.camera_hz <= 0:
            raise ValueError("camera_hz must be positive")
        if config.camera_hz > config.control_hz:
            raise ValueError("camera_hz cannot exceed control_hz")
        self.config = config
        # Each MuJoCo instance has one owner: the control thread advances
        # physics, while one worker per camera owns an independent OSMesa
        # context. Both camera workers receive the same simulation snapshot.
        self.env: SO101WorkshopEnv | None = None
        self._observation: dict[str, np.ndarray] | None = None
        self._jpeg_frames: dict[str, bytes] = {}
        self._keys: set[str] = set()
        self._recorder: LeRobotV3Recorder | None = None
        self._last_dataset_path: Path | None = None
        self._message = "Ready"
        self._error: str | None = None
        self._lock = threading.RLock()
        self._status_cache: dict[str, Any] = {
            "message": "Starting simulation",
            "error": None,
            "recording": False,
            "finalized": False,
            "frames_in_episode": 0,
            "episodes": 0,
            "dataset_path": None,
            "task": None,
            "keys": [],
            "eef_position": None,
            "eef_orientation": None,
            "joint_positions": None,
            "joint_targets": None,
            "sim_time": None,
            "control_hz": self.config.control_hz,
            "camera_hz": self.config.camera_hz,
            "image_size": [self.config.image_width, self.config.image_height],
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
                ]
            ],
        ] = {camera: queue.Queue(maxsize=1) for camera in CAMERA_NAMES}
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
        self._render_inflight = False
        self._render_sequence = 0
        self._pending_renders: dict[int, dict[str, Any]] = {}
        self._render_requested = threading.Event()
        self._render_epoch = 0
        self._recording_generation = 0
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
            for camera in CAMERA_NAMES
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
            if pressed:
                self._keys.add(key)
            else:
                self._keys.discard(key)
            self._status_cache = {
                **self._status_cache,
                "keys": sorted(self._keys),
            }
        self._render_requested.set()

    def clear_keys(self) -> None:
        with self._lock:
            self._keys.clear()
            self._status_cache = {**self._status_cache, "keys": []}
        self._render_requested.set()

    def _current_action(self) -> np.ndarray:
        action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        with self._lock:
            keys = tuple(self._keys)
        for key in keys:
            action += KEY_ACTIONS[key]
        return np.clip(action, -1.0, 1.0)

    @staticmethod
    def _encode_image(camera: str, rgb: np.ndarray) -> bytes:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 86])
        if not ok:
            raise RuntimeError(f"Could not encode {camera} camera")
        return encoded.tobytes()

    def _control_loop(self) -> None:
        try:
            self.env = SO101WorkshopEnv(
                render_mode=None,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
                control_hz=self.config.control_hz,
            )
            self._observation, _ = self.env.reset()
            for render_thread in self._render_threads:
                render_thread.start()
            self._submit_render(np.zeros(len(ACTION_NAMES), dtype=np.float32))
            startup_deadline = time.monotonic() + 30
            while self._render_inflight:
                timeout = startup_deadline - time.monotonic()
                if timeout <= 0:
                    raise TimeoutError("Initial camera rendering timed out")
                self._consume_render_result(self._render_results.get(timeout=timeout))
            self._ready_event.set()

            period = 1.0 / self.config.control_hz
            camera_period = 1.0 / self.config.camera_hz
            next_tick = time.monotonic()
            next_camera_tick = next_tick + camera_period
            while not self._stop_event.is_set():
                self._process_commands()
                action = self._current_action()
                self._observation, _, _, _, _ = self.env.step_dynamics(action)
                self._collect_render_results()

                now = time.monotonic()
                render_requested = self._render_requested.is_set()
                if not self._render_inflight and (
                    render_requested or now >= next_camera_tick
                ):
                    self._submit_render(action)
                    self._render_requested.clear()
                    if render_requested:
                        next_camera_tick = now + camera_period
                    else:
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
            render_env = SO101WorkshopEnv(
                render_mode="rgb_array",
                image_width=self.config.image_width,
                image_height=self.config.image_height,
                control_hz=self.config.control_hz,
            )
            while not self._stop_event.is_set():
                try:
                    sequence, snapshot, action, recording_generation, epoch = (
                        self._render_requests[camera].get(timeout=0.1)
                    )
                except queue.Empty:
                    continue
                render_env.restore_simulation_snapshot(snapshot)
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
                render_env.close()

    def _submit_render(self, action: np.ndarray) -> None:
        if self.env is None or self._render_inflight:
            return
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
            "cameras": set(),
            "action": action_copy,
            "recording_generation": recording_generation,
            "epoch": self._render_epoch,
        }
        for camera in CAMERA_NAMES:
            self._render_requests[camera].put_nowait(
                (
                    sequence,
                    snapshot,
                    action_copy,
                    recording_generation,
                    self._render_epoch,
                )
            )
        self._render_inflight = True

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
        pending = self._pending_renders.get(sequence)
        if pending is None or observation is None or action is None:
            return
        pending["cameras"].add(camera)
        if epoch == self._render_epoch:
            for name, value in observation.items():
                if not name.startswith("observation.images."):
                    pending["observation"].setdefault(name, value)
            image_key = f"observation.images.{camera}"
            pending["observation"][image_key] = observation[image_key]
            jpeg = self._encode_image(camera, observation[image_key])
            with self._lock:
                self._jpeg_frames[camera] = jpeg

        if pending["cameras"] != set(CAMERA_NAMES):
            return

        self._pending_renders.pop(sequence)
        self._render_inflight = False
        paired_observation = pending["observation"]
        if epoch != self._render_epoch:
            return
        if (
            self._recorder is not None
            and self._recorder.recording
            and recording_generation == self._recording_generation
        ):
            self._recorder.add_frame(paired_observation, action)
        with self._lock:
            self._observation = paired_observation
            self._refresh_status_locked()

    def _process_commands(self) -> None:
        while True:
            try:
                command, payload, event, response = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._execute_command(command, payload)
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

    def command(self, command: str, payload: dict[str, Any]) -> None:
        if not self._thread.is_alive():
            raise RuntimeError("Simulation control loop is not running")
        event = threading.Event()
        response: dict[str, Any] = {}
        self._commands.put((command, payload, event, response))
        if not event.wait(timeout=120):
            raise TimeoutError(f"Command timed out: {command}")
        if "error" in response:
            raise response["error"]

    def _execute_command(self, command: str, payload: dict[str, Any]) -> None:
        try:
            self._error = None
            if command == "reset":
                if self._recorder is not None and self._recorder.recording:
                    raise RuntimeError("Stop or discard the recording before reset")
                with self._lock:
                    self._keys.clear()
                self._observation, _ = self.env.reset()
                self._render_epoch += 1
                self._render_requested.set()
                self._message = "Simulation reset"
            elif command == "start_recording":
                if self._recorder is None or self._recorder.finalized:
                    self._recorder = LeRobotV3Recorder(
                        datasets_root=self.config.datasets_root,
                        fps=self.config.camera_hz,
                        image_width=self.config.image_width,
                        image_height=self.config.image_height,
                    )
                self._recording_generation += 1
                self._recorder.start_episode(
                    dataset_name=str(payload.get("dataset_name", "so101_session")),
                    task=str(payload.get("task", "")),
                )
                self._render_requested.set()
                self._message = "Recording episode"
            elif command == "stop_recording":
                recorder = self._require_recorder(recording=True)
                frames = recorder.stop_episode(save=True)
                self._recording_generation += 1
                self._message = f"Saved episode with {frames} frames"
            elif command == "discard_recording":
                recorder = self._require_recorder(recording=True)
                frames = recorder.stop_episode(save=False)
                self._recording_generation += 1
                self._message = f"Discarded {frames} frames"
            elif command == "finish_dataset":
                recorder = self._require_recorder(recording=False)
                if recorder.num_episodes < 1 and not recorder.recording:
                    raise RuntimeError("Save at least one episode before finishing")
                self._last_dataset_path = recorder.finalize()
                self._message = f"Dataset finalized: {self._last_dataset_path}"
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
        self._status_cache = {
            "message": self._message,
            "error": self._error,
            "recording": bool(recorder and recorder.recording),
            "finalized": bool(recorder and recorder.finalized),
            "frames_in_episode": recorder.frames_in_episode if recorder else 0,
            "episodes": recorder.num_episodes if recorder else 0,
            "dataset_path": str(dataset_path) if dataset_path else None,
            "task": recorder.task if recorder else None,
            "keys": sorted(self._keys),
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
            "joint_targets": (
                self.env.data.ctrl.round(4).tolist()
                if self.env is not None
                else None
            ),
            "sim_time": (
                round(float(self.env.data.time), 4) if self.env is not None else None
            ),
            "control_hz": self.config.control_hz,
            "camera_hz": self.config.camera_hz,
            "image_size": [self.config.image_width, self.config.image_height],
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status_cache)

    def close(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=30)
