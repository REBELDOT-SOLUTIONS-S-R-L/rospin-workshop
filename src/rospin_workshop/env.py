from __future__ import annotations

import os
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, ClassVar
from xml.sax.saxutils import escape

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from rospin_workshop.collision import compose_robot_collisions
from rospin_workshop.tasks import OBJECT_CATALOG, TaskDefinition, compose_task_model

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINT_NAMES = JOINT_NAMES[:-1]
CAMERA_NAMES = ("wrist", "top", "perspective")
ACTION_NAMES = (
    "eef_dx",
    "eef_dy",
    "eef_dz",
    "shoulder_pan_delta",
    "shoulder_lift_delta",
    "elbow_flex_delta",
    "wrist_flex_delta",
    "wrist_roll_delta",
    "gripper_command",
)

DIRECT_JOINT_ACTIONS = (
    (3, 0),  # shoulder pan
    (4, 1),  # shoulder lift
    (5, 2),  # elbow flex
    (6, 3),  # wrist flex
    (7, 4),  # wrist roll
)

HOME_JOINT_POSITIONS = np.array(
    [0.0, -1.6580628, 1.5707963, 1.2217305, -1.5707963, 0.2617994],
    dtype=np.float64,
)


def _so101_asset_dir() -> Path:
    configured = os.environ.get("ROSPIN_SO101_ASSET_DIR")
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).resolve().parents[2] / "assets/robots/so101",
        Path("/assets/robots/so101"),
        Path("/workspace/assets/robots/so101"),
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "so101_new_calib.urdf").is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        f"Could not locate the vendored SO-101 URDF/STL directory; searched: {searched}"
    )


def _workshop_object_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "assets/objects",
        Path("/assets/objects"),
        Path("/workspace/assets/objects"),
    ]
    for candidate in candidates:
        if (candidate / "cube_green.obj").is_file() and (
            candidate / "oala_cuburi.obj"
        ).is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not locate the vendored workshop object meshes; searched: {searched}"
    )


class SO101WorkshopEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """MuJoCo/Gymnasium environment with Cartesian translation and joint rotation.

    Actions are normalized deltas
    ``[dx, dy, dz, shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    wrist_roll, gripper]`` in ``[-1, 1]``. World-frame EEF translation uses
    damped least-squares differential IK. Rotation controls address all five
    arm joints directly. The final element is a latched absolute gripper
    command: negative closes fully, positive opens fully, and zero retains the
    existing target.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        task: TaskDefinition | None = None,
        render_mode: str | None = "rgb_array",
        image_width: int = 640,
        image_height: int = 480,
        control_hz: int = 60,
        translation_speed: float = 0.12,
        joint_speed: float = 0.8,
        gripper_speed: float = 2.5,
        ik_damping: float = 0.04,
        max_joint_step: float = 0.10,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")

        self.render_mode = render_mode
        self.image_width = image_width
        self.image_height = image_height
        self.control_hz = control_hz
        self.translation_step = translation_speed / control_hz
        self.joint_step = joint_speed / control_hz
        self.gripper_step = gripper_speed / control_hz
        self.ik_damping = ik_damping
        self.max_joint_step = max_joint_step
        self.task = task
        self._task_condition_started_at: float | None = None
        self._task_success_latched = False

        if model_path is None:
            model_resource = files("rospin_workshop").joinpath(
                "models/so101_workshop.xml"
            )
            with as_file(model_resource) as resolved_model:
                model_xml = resolved_model.read_text(encoding="utf-8")
            mesh_dir = escape(
                str(_so101_asset_dir() / "assets"),
                {'"': "&quot;"},
            )
            model_xml = model_xml.replace("SO101_MESH_DIR", mesh_dir)
            model_xml = compose_robot_collisions(model_xml)
            if self.task is not None:
                model_xml = compose_task_model(model_xml, self.task)
                object_dir = escape(
                    str(_workshop_object_dir()),
                    {'"': "&quot;"},
                )
                model_xml = model_xml.replace("WORKSHOP_OBJECT_DIR", object_dir)
            self.model = mujoco.MjModel.from_xml_string(model_xml)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(Path(model_path)))
        self.data = mujoco.MjData(self.model)

        self._joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in JOINT_NAMES
            ]
        )
        self._qpos_indices = self.model.jnt_qposadr[self._joint_ids]
        self._dof_indices = self.model.jnt_dofadr[self._joint_ids]
        self._arm_dof_indices = self._dof_indices[:-1]
        self._eef_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_site"
        )
        self._camera_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in CAMERA_NAMES
        }
        self._task_body_ids: dict[str, int] = {}
        self._task_region_ids: dict[str, int] = {}
        if self.task is not None:
            for instance in self.task.objects:
                body_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"task_{instance.name}",
                )
                if body_id < 0:
                    raise ValueError(
                        f"Task object was not composed into the model: {instance.name}"
                    )
                self._task_body_ids[instance.name] = body_id
            for condition in self.task.success_conditions:
                if condition.type != "object_fully_inside_region":
                    continue
                region = str(condition.values["region"])
                owner, region_name = region.split(".", 1)
                site_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_SITE,
                    f"task_{owner}__{region_name}",
                )
                if site_id < 0:
                    raise ValueError(f"Task success region does not exist: {region}")
                self._task_region_ids[region] = site_id
        if (
            np.any(self._joint_ids < 0)
            or self._eef_site_id < 0
            or any(camera_id < 0 for camera_id in self._camera_ids.values())
        ):
            raise ValueError(
                "MuJoCo model is missing required SO-101 joints, site, or cameras"
            )

        self._physics_steps = max(
            1, round((1.0 / self.control_hz) / float(self.model.opt.timestep))
        )
        # Keep simulation time synchronized with wall time even when the
        # configured control period is not an integer multiple of the XML step.
        self.model.opt.timestep = 1.0 / (self.control_hz * self._physics_steps)
        if self.render_mode == "rgb_array":
            # MuJoCo's software renderer performs an expensive additional scene
            # pass when any material has reflectance. The workshop cameras need
            # deterministic 25 Hz capture more than subtle glossy highlights.
            self.model.mat_reflectance[:] = 0.0
        self._renderer = (
            mujoco.Renderer(
                self.model, height=self.image_height, width=self.image_width
            )
            if self.render_mode == "rgb_array"
            else None
        )
        self._last_images: dict[str, np.ndarray] | None = None
        self._previous_action = np.zeros(len(ACTION_NAMES), dtype=np.float32)

        def image_space() -> spaces.Box:
            return spaces.Box(
                low=0,
                high=255,
                shape=(self.image_height, self.image_width, 3),
                dtype=np.uint8,
            )

        joint_ranges = self.model.jnt_range[self._joint_ids].astype(np.float32)
        observation_spaces: dict[str, spaces.Space[np.ndarray]] = {
            "observation.state": spaces.Box(
                low=joint_ranges[:, 0],
                high=joint_ranges[:, 1],
                dtype=np.float32,
            ),
            "observation.velocity": spaces.Box(
                low=-20.0,
                high=20.0,
                shape=(len(JOINT_NAMES),),
                dtype=np.float32,
            ),
            "observation.eef_position": spaces.Box(
                low=-2.0, high=2.0, shape=(3,), dtype=np.float32
            ),
            "observation.eef_orientation": spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            ),
        }
        if self._renderer is not None:
            observation_spaces.update(
                {
                    "observation.images.wrist": image_space(),
                    "observation.images.top": image_space(),
                    "observation.images.perspective": image_space(),
                }
            )
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(ACTION_NAMES),), dtype=np.float32
        )

    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[self._qpos_indices].astype(np.float32, copy=True)

    @property
    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self._dof_indices].astype(np.float32, copy=True)

    @property
    def joint_ranges(self) -> np.ndarray:
        return self.model.jnt_range[self._joint_ids].astype(np.float64, copy=True)

    @property
    def eef_position(self) -> np.ndarray:
        return self.data.site_xpos[self._eef_site_id].astype(np.float32, copy=True)

    @property
    def eef_orientation(self) -> np.ndarray:
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            quaternion,
            self.data.site_xmat[self._eef_site_id],
        )
        return quaternion.astype(np.float32)

    def _render_cameras(self) -> dict[str, np.ndarray]:
        if self._renderer is None:
            raise RuntimeError("Camera rendering requires render_mode='rgb_array'")
        images: dict[str, np.ndarray] = {}
        for camera_name in CAMERA_NAMES:
            self._renderer.update_scene(self.data, camera=camera_name)
            images[camera_name] = self._renderer.render().copy()
        self._last_images = images
        return images

    def capture_camera_observation(self, camera: str) -> dict[str, np.ndarray]:
        """Capture state and one camera for a dedicated render worker."""

        if self._renderer is None:
            raise RuntimeError("Camera rendering requires render_mode='rgb_array'")
        if camera not in CAMERA_NAMES:
            raise KeyError(camera)
        observation = self._get_state_obs()
        self._renderer.update_scene(self.data, camera=camera)
        observation[f"observation.images.{camera}"] = self._renderer.render().copy()
        return observation

    def set_camera_lookat(
        self,
        camera: str,
        position: np.ndarray,
        lookat: np.ndarray,
    ) -> None:
        """Place a fixed camera in world coordinates and aim it at a point."""

        if camera not in CAMERA_NAMES:
            raise KeyError(camera)
        position = np.asarray(position, dtype=np.float64)
        lookat = np.asarray(lookat, dtype=np.float64)
        if position.shape != (3,) or lookat.shape != (3,):
            raise ValueError("Camera position and lookat must each have shape (3,)")

        forward = lookat - position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            raise ValueError("Camera position and lookat cannot be the same")
        forward /= forward_norm
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            raise ValueError("Camera cannot look exactly along the world Z axis")
        right /= right_norm
        up = np.cross(right, forward)
        rotation = np.column_stack((right, up, -forward))
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.ravel())

        camera_id = self._camera_ids[camera]
        self.model.cam_pos[camera_id] = position
        self.model.cam_quat[camera_id] = quaternion
        mujoco.mj_forward(self.model, self.data)

    def _get_state_obs(self) -> dict[str, np.ndarray]:
        return {
            "observation.state": self.joint_positions,
            "observation.velocity": self.joint_velocities,
            "observation.eef_position": self.eef_position,
            "observation.eef_orientation": self.eef_orientation,
        }

    def _get_obs(self) -> dict[str, np.ndarray]:
        observation = self._get_state_obs()
        images = self._render_cameras()
        observation.update(
            {
                "observation.images.wrist": images["wrist"],
                "observation.images.top": images["top"],
                "observation.images.perspective": images["perspective"],
            }
        )
        return observation

    def _get_info(self) -> dict[str, Any]:
        return {
            "eef_position": self.eef_position,
            "eef_orientation": self.eef_orientation,
            "joint_targets": self.data.ctrl.copy().astype(np.float32),
            "sim_time": float(self.data.time),
            "task": self.task_status(),
        }

    def _object_fully_inside_region(self, object_name: str, region: str) -> bool:
        if self.task is None:
            return False
        instance = next(item for item in self.task.objects if item.name == object_name)
        half_extents = OBJECT_CATALOG[instance.catalog_id].half_extents
        if half_extents is None:
            raise ValueError(
                f"Object {instance.catalog_id!r} does not define containment extents"
            )
        body_id = self._task_body_ids[object_name]
        site_id = self._task_region_ids[region]
        body_rotation = self.data.xmat[body_id].reshape(3, 3)
        body_position = self.data.xpos[body_id]
        corners = np.array(
            [
                [x, y, z]
                for x in (-half_extents[0], half_extents[0])
                for y in (-half_extents[1], half_extents[1])
                for z in (-half_extents[2], half_extents[2])
            ],
            dtype=np.float64,
        )
        world_corners = corners @ body_rotation.T + body_position
        site_rotation = self.data.site_xmat[site_id].reshape(3, 3)
        local_corners = (world_corners - self.data.site_xpos[site_id]) @ site_rotation
        radius, half_height = self.model.site_size[site_id, :2]
        radial = np.linalg.norm(local_corners[:, :2], axis=1)
        return bool(
            np.all(radial <= radius + 1e-8)
            and np.all(np.abs(local_corners[:, 2]) <= half_height + 1e-8)
        )

    def _task_conditions_met(self) -> bool:
        if self.task is None:
            return False
        for condition in self.task.success_conditions:
            values = condition.values
            if condition.type == "object_fully_inside_region":
                if not self._object_fully_inside_region(
                    str(values["object"]), str(values["region"])
                ):
                    return False
            elif condition.type == "gripper_open":
                gripper_range = self.joint_ranges[-1]
                fraction = (self.joint_positions[-1] - gripper_range[0]) / np.ptp(
                    gripper_range
                )
                if fraction < float(values["minimum_fraction"]):
                    return False
            elif condition.type == "body_speed_below":
                body_id = self._task_body_ids[str(values["object"])]
                angular_speed = float(np.linalg.norm(self.data.cvel[body_id, :3]))
                linear_speed = float(np.linalg.norm(self.data.cvel[body_id, 3:]))
                if (
                    linear_speed > float(values["linear_mps"])
                    or angular_speed > float(values["angular_rps"])
                ):
                    return False
            else:  # validated task definitions cannot reach this branch
                raise ValueError(f"Unsupported task condition: {condition.type}")
        return True

    def _update_task_success(self) -> None:
        if self.task is None or self._task_success_latched:
            return
        if not self._task_conditions_met():
            self._task_condition_started_at = None
            return
        if self._task_condition_started_at is None:
            self._task_condition_started_at = float(self.data.time)
            return
        if (
            float(self.data.time) - self._task_condition_started_at
            >= self.task.success_hold_seconds
        ):
            self._task_success_latched = True

    def task_status(self) -> dict[str, Any] | None:
        if self.task is None:
            return None
        elapsed = (
            max(0.0, float(self.data.time) - self._task_condition_started_at)
            if self._task_condition_started_at is not None
            else 0.0
        )
        return {
            "conditions_met": self._task_condition_started_at is not None,
            "success": self._task_success_latched,
            "success_progress": min(1.0, elapsed / self.task.success_hold_seconds),
        }

    def task_object_states(self) -> dict[str, dict[str, list[float]]]:
        """Return world-frame poses and velocities for configured task objects."""

        states: dict[str, dict[str, list[float]]] = {}
        for name, body_id in self._task_body_ids.items():
            quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quaternion, self.data.xmat[body_id])
            states[name] = {
                "position": self.data.xpos[body_id].astype(float).tolist(),
                "quaternion": quaternion.tolist(),
                "angular_velocity": self.data.cvel[body_id, :3]
                .astype(float)
                .tolist(),
                "linear_velocity": self.data.cvel[body_id, 3:]
                .astype(float)
                .tolist(),
            }
        return states

    def _apply_action(self, action: np.ndarray) -> None:
        if not np.any(action):
            # A released key must stop the robot at its current pose instead of
            # letting a queued position target continue moving it. Latch only
            # on the active-to-idle transition; repeatedly following qpos would
            # let gravity walk the idle robot downward.
            if np.any(self._previous_action[:-1]):
                self.data.ctrl[:-1] = self.joint_positions[:-1]
            self._previous_action = action.copy()
            return

        position_delta = action[:3] * self.translation_step
        jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self._eef_site_id)
        jacobian = jac_pos[:, self._arm_dof_indices]
        regularizer = (self.ik_damping**2) * np.eye(3)
        joint_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + regularizer,
            position_delta,
        )
        joint_delta = np.clip(joint_delta, -self.max_joint_step, self.max_joint_step)

        arm_targets = self.data.ctrl[:-1].copy()
        if np.any(self._previous_action[:3]) and not np.any(action[:3]):
            # Cancel the multi-joint IK target before beginning a direct-joint
            # command, even when the user switches keys between control ticks.
            arm_targets[:] = self.joint_positions[:-1]
        arm_targets += joint_delta
        for action_index, joint_index in DIRECT_JOINT_ACTIONS:
            if self._previous_action[action_index] and not action[action_index]:
                arm_targets[joint_index] = self.joint_positions[joint_index]
            arm_targets[joint_index] += action[action_index] * self.joint_step
        arm_ranges = self.model.jnt_range[self._joint_ids[:-1]]
        self.data.ctrl[:-1] = np.clip(arm_targets, arm_ranges[:, 0], arm_ranges[:, 1])

        gripper_range = self.model.jnt_range[self._joint_ids[-1]]
        if action[-1] < 0:
            self.data.ctrl[-1] = gripper_range[0]
        elif action[-1] > 0:
            self.data.ctrl[-1] = gripper_range[1]
        self._previous_action = action.copy()

    def step_dynamics(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance control and physics without waiting for camera rasterization."""

        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}"
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)
        for _ in range(self._physics_steps):
            mujoco.mj_step(self.model, self.data)
        self._update_task_success()
        return self._get_state_obs(), 0.0, False, False, self._get_info()

    def step_joint_targets(
        self,
        targets: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Rate-limit absolute joint targets and advance one control tick."""

        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (len(JOINT_NAMES),):
            raise ValueError(
                f"Expected {len(JOINT_NAMES)} joint targets, got {targets.shape}"
            )
        if not np.all(np.isfinite(targets)):
            raise ValueError("Joint targets must all be finite")

        ranges = self.joint_ranges
        desired = np.clip(targets, ranges[:, 0], ranges[:, 1])
        max_steps = np.full(len(JOINT_NAMES), self.joint_step, dtype=np.float64)
        max_steps[-1] = self.gripper_step
        delta = np.clip(desired - self.data.ctrl, -max_steps, max_steps)
        self.data.ctrl[:] = np.clip(
            self.data.ctrl + delta,
            ranges[:, 0],
            ranges[:, 1],
        )
        # Absolute trajectory targets and normalized keyboard deltas have
        # separate transition semantics. A later keyboard command should
        # start cleanly.
        self._previous_action.fill(0)
        for _ in range(self._physics_steps):
            mujoco.mj_step(self.model, self.data)
        self._update_task_success()
        return self._get_state_obs(), 0.0, False, False, self._get_info()

    def hold_current_pose(self) -> None:
        """Cancel queued target motion at the current simulated joint pose."""

        self.data.ctrl[:] = np.clip(
            self.joint_positions,
            self.joint_ranges[:, 0],
            self.joint_ranges[:, 1],
        )
        self._previous_action.fill(0)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.step_dynamics(action)
        if self._renderer is not None:
            images = self._render_cameras()
            observation.update(
                {
                    "observation.images.wrist": images["wrist"],
                    "observation.images.top": images["top"],
                    "observation.images.perspective": images["perspective"],
                }
            )
        return observation, reward, terminated, truncated, info

    def simulation_snapshot(self) -> dict[str, np.ndarray | float]:
        """Copy the state needed to render this simulation in another thread."""

        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
        }

    def restore_simulation_snapshot(
        self, snapshot: dict[str, np.ndarray | float]
    ) -> None:
        """Restore a copied state before rendering it on this environment."""

        self.data.qpos[:] = np.asarray(snapshot["qpos"])
        self.data.qvel[:] = np.asarray(snapshot["qvel"])
        self.data.ctrl[:] = np.asarray(snapshot["ctrl"])
        self.data.time = float(snapshot["time"])
        mujoco.mj_forward(self.model, self.data)

    def capture_observation(self) -> dict[str, np.ndarray]:
        """Return state and both cameras without advancing physics."""

        return self._get_obs()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if self.task is not None:
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self._qpos_indices] = HOME_JOINT_POSITIONS
            self.data.ctrl[:] = HOME_JOINT_POSITIONS
            for instance in self.task.objects:
                if instance.spawn is None:
                    continue
                body_id = self._task_body_ids[instance.name]
                joint_id = int(self.model.body_jntadr[body_id])
                qpos_address = int(self.model.jnt_qposadr[joint_id])
                dof_address = int(self.model.jnt_dofadr[joint_id])
                for _ in range(128):
                    position = np.asarray(
                        instance.pose.position, dtype=np.float64
                    )
                    position[0] = self.np_random.uniform(*instance.spawn.x)
                    position[1] = self.np_random.uniform(*instance.spawn.y)
                    self.data.qpos[qpos_address : qpos_address + 3] = position
                    self.data.qvel[dof_address : dof_address + 6] = 0
                    mujoco.mj_forward(self.model, self.data)
                    if not any(
                        body_id
                        in (
                            self.model.geom_bodyid[contact.geom1],
                            self.model.geom_bodyid[contact.geom2],
                        )
                        for contact in self.data.contact
                    ):
                        break
                else:
                    raise RuntimeError(
                        f"Could not place {instance.name!r} without contact "
                        "after 128 workspace samples"
                    )
        else:
            home_key = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
            )
            if home_key >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
            else:
                mujoco.mj_resetData(self.model, self.data)
        self._previous_action.fill(0)
        self._task_condition_started_at = None
        self._task_success_latched = False
        mujoco.mj_forward(self.model, self.data)
        observation = (
            self._get_obs() if self._renderer is not None else self._get_state_obs()
        )
        return observation, self._get_info()

    def set_task_object_pose(
        self,
        name: str,
        position: np.ndarray,
        quaternion: np.ndarray | None = None,
    ) -> None:
        """Place a free task object, primarily for deterministic setup checks."""

        if name not in self._task_body_ids:
            raise KeyError(name)
        body_id = self._task_body_ids[name]
        joint_id = int(self.model.body_jntadr[body_id])
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Task object is not freely movable: {name}")
        qpos_address = int(self.model.jnt_qposadr[joint_id])
        dof_address = int(self.model.jnt_dofadr[joint_id])
        position_array = np.asarray(position, dtype=np.float64)
        if position_array.shape != (3,) or not np.all(np.isfinite(position_array)):
            raise ValueError("Object position must contain three finite values")
        quaternion_array = (
            self.data.qpos[qpos_address + 3 : qpos_address + 7].copy()
            if quaternion is None
            else np.asarray(quaternion, dtype=np.float64)
        )
        if quaternion_array.shape != (4,) or not np.all(np.isfinite(quaternion_array)):
            raise ValueError("Object quaternion must contain four finite values")
        norm = np.linalg.norm(quaternion_array)
        if norm < 1e-8:
            raise ValueError("Object quaternion cannot be zero")
        self.data.qpos[qpos_address : qpos_address + 3] = position_array
        self.data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion_array / norm
        self.data.qvel[dof_address : dof_address + 6] = 0
        self._task_condition_started_at = None
        self._task_success_latched = False
        mujoco.mj_forward(self.model, self.data)

    def render(self) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("Rendering requires render_mode='rgb_array'")
        if self._last_images is None:
            self._render_cameras()
        return self._last_images["perspective"].copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
