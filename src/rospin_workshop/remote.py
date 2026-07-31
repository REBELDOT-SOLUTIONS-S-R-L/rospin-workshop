from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from rospin_workshop.env import JOINT_NAMES

LOGGER = logging.getLogger(__name__)

REMOTE_POSITION_KEYS = tuple(f"{name}.pos" for name in JOINT_NAMES)


def remote_positions_to_joint_targets(
    positions: Mapping[str, float] | np.ndarray,
    joint_ranges: np.ndarray,
) -> np.ndarray:
    """Map calibrated SO-101 leader positions to MuJoCo joint targets.

    LeRobot reports the five arm motors in degrees and the gripper on a
    calibrated 0–100 range. MuJoCo uses radians for every joint.
    """

    if isinstance(positions, Mapping):
        values = np.asarray(
            [positions[key] for key in REMOTE_POSITION_KEYS],
            dtype=np.float64,
        )
    else:
        values = np.asarray(positions, dtype=np.float64)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if values.shape != (len(JOINT_NAMES),):
        raise ValueError(
            f"Expected {len(JOINT_NAMES)} remote positions, got {values.shape}"
        )
    if ranges.shape != (len(JOINT_NAMES), 2):
        raise ValueError(
            f"Expected joint ranges shape {(len(JOINT_NAMES), 2)}, got {ranges.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Remote positions must all be finite")

    targets = np.empty(len(JOINT_NAMES), dtype=np.float64)
    targets[:-1] = np.deg2rad(values[:-1])
    gripper_fraction = np.clip(values[-1], 0.0, 100.0) / 100.0
    targets[-1] = ranges[-1, 0] + gripper_fraction * np.ptp(ranges[-1])
    return np.clip(targets, ranges[:, 0], ranges[:, 1])


class SO101RemoteReader:
    """Read an SO-101 leader in the background without blocking simulation.

    The leader's existing calibration is read directly from its motor
    registers. No interactive calibration or motor configuration is run, and
    torque is disabled immediately after the serial connection is established.
    """

    def __init__(
        self,
        *,
        port: str | None,
        calibration_dir: Path,
        remote_id: str = "workshop_remote",
        poll_hz: int = 60,
        stale_after: float = 0.5,
        reconnect_after: float = 1.0,
        leader_factory: Callable[[], Any] | None = None,
    ) -> None:
        if poll_hz <= 0:
            raise ValueError("Remote poll rate must be positive")
        if stale_after <= 0:
            raise ValueError("Remote stale timeout must be positive")
        self.port = port
        self.calibration_dir = Path(calibration_dir)
        self.remote_id = remote_id
        self.poll_hz = poll_hz
        self.stale_after = stale_after
        self.reconnect_after = reconnect_after
        self._leader_factory = leader_factory

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._leader: Any | None = None
        self._connected = False
        self._calibrated = False
        self._error: str | None = None
        self._positions: np.ndarray | None = None
        self._last_read_at: float | None = None
        self._read_hz: float | None = None

    @property
    def configured(self) -> bool:
        return bool(self.port)

    def start(self) -> None:
        if not self.configured:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="so101-remote",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        if thread is None or not thread.is_alive():
            self._disconnect()

    def latest_positions(self) -> np.ndarray | None:
        with self._lock:
            if (
                not self._connected
                or self._positions is None
                or self._last_read_at is None
                or time.monotonic() - self._last_read_at > self.stale_after
            ):
                return None
            return self._positions.copy()

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            age_ms = (
                round((now - self._last_read_at) * 1000)
                if self._last_read_at is not None
                else None
            )
            fresh = bool(
                self._connected
                and age_ms is not None
                and age_ms <= self.stale_after * 1000
            )
            positions = (
                {
                    name: round(float(value), 3)
                    for name, value in zip(JOINT_NAMES, self._positions, strict=True)
                }
                if self._positions is not None
                else None
            )
            return {
                "configured": self.configured,
                "port": self.port,
                "connected": self._connected,
                "calibrated": self._calibrated,
                "available": fresh,
                "error": self._error,
                "age_ms": age_ms,
                "read_hz": (
                    round(self._read_hz, 1) if self._read_hz is not None else None
                ),
                "positions": positions,
            }

    def _make_leader(self) -> Any:
        if self._leader_factory is not None:
            return self._leader_factory()

        # Import lazily so the portable simulation remains usable when no
        # serial remote is configured.
        from lerobot.teleoperators.so_leader import (  # noqa: PLC0415
            SO101Leader,
            SO101LeaderConfig,
        )

        config = SO101LeaderConfig(
            port=str(self.port),
            use_degrees=True,
            id=self.remote_id,
            calibration_dir=self.calibration_dir,
        )
        return SO101Leader(config)

    @staticmethod
    def _validate_calibration(calibration: Mapping[str, Any]) -> None:
        missing = set(JOINT_NAMES) - set(calibration)
        extra = set(calibration) - set(JOINT_NAMES)
        if missing or extra:
            raise RuntimeError(
                "Remote calibration motor mismatch "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )
        for expected_id, name in enumerate(JOINT_NAMES, start=1):
            entry = calibration[name]
            if int(entry.id) != expected_id:
                raise RuntimeError(
                    f"Remote motor {name} has ID {entry.id}, expected {expected_id}"
                )
            if int(entry.range_max) <= int(entry.range_min):
                raise RuntimeError(
                    f"Remote motor {name} has an invalid calibration range"
                )

    def _connect(self) -> Any:
        leader = self._make_leader()
        leader.bus.connect()
        try:
            calibration = leader.bus.read_calibration()
            self._validate_calibration(calibration)
            # get_action() normalizes through the bus calibration, so replace
            # both copies with the values just read from the motor registers.
            leader.calibration = calibration
            leader.bus.calibration = calibration
            leader.bus.disable_torque()
        except BaseException:
            try:
                leader.bus.disable_torque()
            except BaseException:
                LOGGER.exception("Could not disable SO-101 remote torque")
            try:
                leader.bus.disconnect(disable_torque=False)
            except BaseException:
                LOGGER.exception("Could not close SO-101 remote after connect failure")
            raise

        with self._lock:
            self._leader = leader
            self._connected = True
            self._calibrated = True
            self._error = None
            self._positions = None
            self._last_read_at = None
            self._read_hz = None
        LOGGER.info("SO-101 remote connected on %s with torque disabled", self.port)
        return leader

    def _disconnect(self) -> None:
        with self._lock:
            leader = self._leader
            self._leader = None
            self._connected = False
            self._calibrated = False
        if leader is None:
            return
        try:
            leader.bus.disable_torque()
        except BaseException:
            LOGGER.exception("Could not disable SO-101 remote torque on disconnect")
        try:
            leader.bus.disconnect(disable_torque=False)
        except BaseException:
            LOGGER.exception("Could not close SO-101 remote serial connection")

    def _record_action(self, action: Mapping[str, float]) -> None:
        try:
            positions = np.asarray(
                [action[key] for key in REMOTE_POSITION_KEYS],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise RuntimeError(f"Remote action is missing {exc.args[0]}") from exc
        if not np.all(np.isfinite(positions)):
            raise RuntimeError("Remote returned a non-finite motor position")

        read_at = time.monotonic()
        with self._lock:
            if self._last_read_at is not None:
                interval = read_at - self._last_read_at
                if interval > 0:
                    instantaneous_hz = 1.0 / interval
                    self._read_hz = (
                        instantaneous_hz
                        if self._read_hz is None
                        else 0.15 * instantaneous_hz + 0.85 * self._read_hz
                    )
            self._positions = positions
            self._last_read_at = read_at
            self._error = None

    def _run(self) -> None:
        period = 1.0 / self.poll_hz
        leader: Any | None = None
        while not self._stop_event.is_set():
            try:
                if leader is None:
                    leader = self._connect()
                started = time.monotonic()
                self._record_action(leader.get_action())
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    self._stop_event.wait(remaining)
            except BaseException as exc:  # keep retrying after USB interruptions
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                LOGGER.warning("SO-101 remote unavailable on %s: %s", self.port, exc)
                with self._lock:
                    self._error = str(exc)
                    self._connected = False
                    self._calibrated = False
                self._disconnect()
                leader = None
                self._stop_event.wait(self.reconnect_after)
        self._disconnect()
