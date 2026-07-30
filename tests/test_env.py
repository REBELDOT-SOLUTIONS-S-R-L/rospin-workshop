from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils.env_checker import check_env

from rospin_workshop import ENV_ID
from rospin_workshop.env import (
    ACTION_NAMES,
    CAMERA_NAMES,
    JOINT_NAMES,
    SO101WorkshopEnv,
)


def test_environment_contract_and_cameras() -> None:
    env = SO101WorkshopEnv(image_width=96, image_height=72)
    try:
        observation, info = env.reset(seed=7)
        assert env.observation_space.contains(observation)
        assert env.action_space.shape == (len(ACTION_NAMES),)
        assert len(JOINT_NAMES) == 6
        assert set(CAMERA_NAMES) == {"wrist", "perspective"}
        assert observation["observation.images.wrist"].shape == (72, 96, 3)
        assert observation["observation.images.perspective"].dtype == np.uint8
        assert info["eef_position"].shape == (3,)
        assert info["eef_orientation"].shape == (4,)
        np.testing.assert_allclose(
            np.linalg.norm(observation["observation.eef_orientation"]),
            1.0,
            atol=1e-5,
        )
        # Reset must start over the table in a useful manipulation pose.
        assert abs(info["eef_position"][0]) < 0.30
        assert abs(info["eef_position"][1]) < 0.30
        assert 0.80 < info["eef_position"][2] < 1.00
        assert env.model.nmesh == 13
    finally:
        env.close()


def test_cartesian_action_moves_eef_and_respects_joint_limits() -> None:
    env = SO101WorkshopEnv(image_width=64, image_height=48)
    try:
        env.reset()
        start = env.eef_position.copy()
        for _ in range(4):
            env.step(np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        assert np.linalg.norm(env.eef_position - start) > 0.005
        ranges = env.model.jnt_range[env._joint_ids]
        assert np.all(env.data.ctrl >= ranges[:, 0] - 1e-8)
        assert np.all(env.data.ctrl <= ranges[:, 1] + 1e-8)
    finally:
        env.close()


def test_cartesian_rotation_action_changes_eef_orientation() -> None:
    env = SO101WorkshopEnv(image_width=64, image_height=48)
    try:
        env.reset()
        start = env.eef_orientation.copy()
        for _ in range(4):
            env.step(np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.float32))
        rotation_angle = 2 * np.arccos(
            np.clip(abs(np.dot(start, env.eef_orientation)), 0.0, 1.0)
        )
        assert rotation_angle > 0.02
    finally:
        env.close()


def test_gymnasium_checker() -> None:
    env = SO101WorkshopEnv(image_width=64, image_height=48)
    try:
        # GPU/driver rasterization can differ by one value in a handful of pixels
        # even when MuJoCo state is bit-for-bit deterministic.
        env.spec = EnvSpec(
            id="ROSpin/SO101Workshop-v0",
            entry_point=SO101WorkshopEnv,
            nondeterministic=True,
        )
        check_env(env, skip_render_check=True, skip_close_check=True)
    finally:
        env.close()


def test_registered_environment_can_be_created_with_gym_make() -> None:
    env = gym.make(ENV_ID, image_width=64, image_height=48)
    try:
        observation, _ = env.reset()
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_state_only_mode_advances_without_a_renderer() -> None:
    env = SO101WorkshopEnv(render_mode=None, image_width=64, image_height=48)
    try:
        observation, _ = env.reset()
        assert env._renderer is None
        assert not any(key.startswith("observation.images.") for key in observation)
        observation, _, _, _, _ = env.step_dynamics(
            np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        )
        assert env.observation_space.contains(observation)
        assert env.data.time > 0
    finally:
        env.close()
