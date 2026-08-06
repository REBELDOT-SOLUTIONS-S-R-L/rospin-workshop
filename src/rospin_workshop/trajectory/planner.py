from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from rospin_workshop.env import ARM_JOINT_NAMES, JOINT_NAMES, SO101WorkshopEnv


class MotionPlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class MotionPlan:
    joint_targets: tuple[np.ndarray, ...]
    target_position: np.ndarray | None = None


class TrajectoryPlanner:
    """Small deterministic IK planner for the five-axis SO-101 arm.

    Cartesian goals constrain XYZ only. The two remaining arm degrees of
    freedom retain a continuous posture through damped least-squares IK; an
    arbitrary six-dimensional EEF pose cannot be imposed on a five-axis arm.
    """

    def __init__(self, *, control_hz: int, ik_damping: float = 0.025) -> None:
        self.control_hz = control_hz
        self.ik_damping = ik_damping
        self._env = SO101WorkshopEnv(render_mode=None, control_hz=control_hz)
        self._env.reset()
        self.joint_ranges = self._env.joint_ranges
        self.gripper_range = self.joint_ranges[-1].copy()

    def _forward_position(self, joints: np.ndarray) -> np.ndarray:
        self._env.data.qpos[self._env._qpos_indices] = joints
        mujoco.mj_forward(self._env.model, self._env.data)
        return self._env.data.site_xpos[self._env._eef_site_id].copy()

    def solve_position(
        self,
        position: np.ndarray,
        seed: np.ndarray,
        *,
        tolerance: float = 0.0015,
        max_iterations: int = 160,
    ) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64)
        joints = np.asarray(seed, dtype=np.float64).copy()
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("Cartesian position must contain three finite values")
        if joints.shape != (len(JOINT_NAMES),):
            raise ValueError(f"IK seed must contain {len(JOINT_NAMES)} joints")
        joints = np.clip(joints, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

        for _ in range(max_iterations):
            current = self._forward_position(joints)
            error = position - current
            if np.linalg.norm(error) <= tolerance:
                return joints
            jac_pos = np.zeros((3, self._env.model.nv), dtype=np.float64)
            jac_rot = np.zeros((3, self._env.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(
                self._env.model,
                self._env.data,
                jac_pos,
                jac_rot,
                self._env._eef_site_id,
            )
            jacobian = jac_pos[:, self._env._arm_dof_indices]
            regularizer = (self.ik_damping**2) * np.eye(3)
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + regularizer,
                error,
            )
            delta = np.clip(delta, -0.06, 0.06)
            joints[: len(ARM_JOINT_NAMES)] = np.clip(
                joints[: len(ARM_JOINT_NAMES)] + delta,
                self.joint_ranges[:-1, 0],
                self.joint_ranges[:-1, 1],
            )

        final_error = float(np.linalg.norm(position - self._forward_position(joints)))
        raise MotionPlanningError(
            f"No IK solution within {tolerance:.4f} m; final error {final_error:.4f} m"
        )

    @staticmethod
    def _segment_points(
        start: np.ndarray,
        end: np.ndarray,
        *,
        speed: float,
        control_hz: int,
    ) -> tuple[np.ndarray, ...]:
        distance = float(np.linalg.norm(end - start))
        if distance < 1e-9:
            return ()
        count = max(1, int(np.ceil(distance / (speed / control_hz))))
        return tuple(
            start + (end - start) * (index / count)
            for index in range(1, count + 1)
        )

    def plan_linear(
        self,
        start_joints: np.ndarray,
        target_position: np.ndarray,
        *,
        speed: float,
    ) -> MotionPlan:
        if not 0 < speed <= 0.15:
            raise ValueError("Cartesian speed must be in (0, 0.15] m/s")
        seed = np.asarray(start_joints, dtype=np.float64).copy()
        start_position = self._forward_position(seed)
        targets: list[np.ndarray] = []
        for waypoint in self._segment_points(
            start_position,
            np.asarray(target_position, dtype=np.float64),
            speed=speed,
            control_hz=self.control_hz,
        ):
            seed = self.solve_position(waypoint, seed)
            targets.append(seed.copy())
        return MotionPlan(tuple(targets), np.asarray(target_position, dtype=np.float64))

    def plan_safe_move(
        self,
        start_joints: np.ndarray,
        target_position: np.ndarray,
        *,
        speed: float,
        safe_height: float,
    ) -> MotionPlan:
        target_position = np.asarray(target_position, dtype=np.float64)
        seed = np.asarray(start_joints, dtype=np.float64).copy()
        start_position = self._forward_position(seed)
        travel_height = max(float(safe_height), start_position[2], target_position[2])
        waypoints = (
            np.array([start_position[0], start_position[1], travel_height]),
            np.array([target_position[0], target_position[1], travel_height]),
            target_position,
        )
        targets: list[np.ndarray] = []
        current_position = start_position
        for waypoint in waypoints:
            for point in self._segment_points(
                current_position,
                waypoint,
                speed=speed,
                control_hz=self.control_hz,
            ):
                seed = self.solve_position(point, seed)
                targets.append(seed.copy())
            current_position = waypoint
        return MotionPlan(tuple(targets), target_position.copy())

    def plan_joints(
        self,
        start_joints: np.ndarray,
        target_joints: np.ndarray,
        *,
        speed: float,
    ) -> MotionPlan:
        if not 0 < speed <= 1.5:
            raise ValueError("Joint speed must be in (0, 1.5] rad/s")
        start = np.asarray(start_joints, dtype=np.float64)
        target = np.asarray(target_joints, dtype=np.float64)
        if start.shape != (len(JOINT_NAMES),) or target.shape != (len(JOINT_NAMES),):
            raise ValueError(f"Joint plans require {len(JOINT_NAMES)} values")
        target = np.clip(target, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        count = max(
            1,
            int(np.ceil(np.max(np.abs(target - start)) / (speed / self.control_hz))),
        )
        targets = tuple(
            start + (target - start) * (index / count)
            for index in range(1, count + 1)
        )
        return MotionPlan(targets)

    def close(self) -> None:
        self._env.close()
