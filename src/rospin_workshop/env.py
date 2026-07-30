from __future__ import annotations

import os
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINT_NAMES = JOINT_NAMES[:-1]
CAMERA_NAMES = ("wrist", "perspective")
ACTION_NAMES = (
    "eef_dx",
    "eef_dy",
    "eef_dz",
    "eef_droll",
    "eef_dpitch",
    "eef_dyaw",
    "gripper_delta",
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


class SO101WorkshopEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """MuJoCo/Gymnasium environment controlled by Cartesian EEF increments.

    Actions are normalized deltas
    ``[dx, dy, dz, droll, dpitch, dyaw, gripper]`` in ``[-1, 1]``.
    The world-frame translation and rotation components are mapped to the five
    arm joint targets by weighted damped least-squares differential IK.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        render_mode: str | None = "rgb_array",
        image_width: int = 320,
        image_height: int = 240,
        control_hz: int = 20,
        translation_speed: float = 0.08,
        rotation_speed: float = 0.5,
        gripper_speed: float = 0.35,
        ik_damping: float = 0.04,
        ik_rotation_weight: float = 0.35,
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
        self.rotation_step = rotation_speed / control_hz
        self.gripper_step = gripper_speed / control_hz
        self.ik_damping = ik_damping
        self.ik_rotation_weight = ik_rotation_weight
        self.max_joint_step = max_joint_step

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
        self._renderer = (
            mujoco.Renderer(
                self.model, height=self.image_height, width=self.image_width
            )
            if self.render_mode == "rgb_array"
            else None
        )
        self._last_images: dict[str, np.ndarray] | None = None

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
        }

    def _apply_cartesian_action(self, action: np.ndarray) -> None:
        position_delta = action[:3] * self.translation_step
        rotation_delta = action[3:6] * self.rotation_step
        jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self._eef_site_id)
        jacobian = np.vstack(
            (
                jac_pos[:, self._arm_dof_indices],
                self.ik_rotation_weight * jac_rot[:, self._arm_dof_indices],
            )
        )
        task_delta = np.concatenate(
            (position_delta, self.ik_rotation_weight * rotation_delta)
        )
        regularizer = (self.ik_damping**2) * np.eye(len(ARM_JOINT_NAMES))
        joint_delta = np.linalg.solve(
            jacobian.T @ jacobian + regularizer,
            jacobian.T @ task_delta,
        )
        joint_delta = np.clip(joint_delta, -self.max_joint_step, self.max_joint_step)

        arm_targets = self.data.ctrl[:-1] + joint_delta
        arm_ranges = self.model.jnt_range[self._joint_ids[:-1]]
        self.data.ctrl[:-1] = np.clip(arm_targets, arm_ranges[:, 0], arm_ranges[:, 1])

        gripper_range = self.model.jnt_range[self._joint_ids[-1]]
        self.data.ctrl[-1] = np.clip(
            self.data.ctrl[-1] + action[6] * self.gripper_step,
            gripper_range[0],
            gripper_range[1],
        )

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
        self._apply_cartesian_action(action)
        for _ in range(self._physics_steps):
            mujoco.mj_step(self.model, self.data)
        return self._get_state_obs(), 0.0, False, False, self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.step_dynamics(action)
        if self._renderer is not None:
            images = self._render_cameras()
            observation.update(
                {
                    "observation.images.wrist": images["wrist"],
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
        home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        observation = (
            self._get_obs() if self._renderer is not None else self._get_state_obs()
        )
        return observation, self._get_info()

    def render(self) -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("Rendering requires render_mode='rgb_array'")
        if self._last_images is None:
            self._render_cameras()
        return self._last_images["perspective"].copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
