from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils.env_checker import check_env

from rospin_workshop import ENV_ID
from rospin_workshop.env import (
    ACTION_NAMES,
    CAMERA_NAMES,
    DIRECT_JOINT_ACTIONS,
    JOINT_NAMES,
    SO101WorkshopEnv,
)


def _action(index: int, value: float = 1.0) -> np.ndarray:
    action = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    action[index] = value
    return action


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
        for _ in range(12):
            env.step(_action(0))
        assert np.linalg.norm(env.eef_position - start) > 0.01
        ranges = env.model.jnt_range[env._joint_ids]
        assert np.all(env.data.ctrl >= ranges[:, 0] - 1e-8)
        assert np.all(env.data.ctrl <= ranges[:, 1] + 1e-8)
    finally:
        env.close()


def test_rotation_actions_address_only_the_named_joint_target() -> None:
    env = SO101WorkshopEnv(
        render_mode=None,
        image_width=64,
        image_height=48,
        control_hz=60,
    )
    try:
        env.reset()
        for action_index, joint_index in DIRECT_JOINT_ACTIONS:
            before = env.data.ctrl.copy()
            env.step_dynamics(_action(action_index))
            changed = np.flatnonzero(np.abs(env.data.ctrl - before) > 1e-8)
            assert changed.tolist() == [joint_index]
            env.reset()
    finally:
        env.close()


def test_releasing_controls_cancels_queued_servo_motion() -> None:
    env = SO101WorkshopEnv(render_mode=None, control_hz=60)
    try:
        env.reset()
        action = _action(0)
        for _ in range(12):
            env.step_dynamics(action)
        env.step_dynamics(np.zeros(len(ACTION_NAMES), dtype=np.float32))
        np.testing.assert_allclose(env.data.ctrl, env.joint_positions, atol=0.01)
    finally:
        env.close()


def test_idle_controller_holds_the_latched_pose() -> None:
    env = SO101WorkshopEnv(render_mode=None, control_hz=60)
    try:
        env.reset()
        start = env.eef_position.copy()
        idle = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        for _ in range(300):
            env.step_dynamics(idle)
        assert np.linalg.norm(env.eef_position - start) < 0.01
    finally:
        env.close()


def test_gripper_commands_latch_full_joint_travel() -> None:
    env = SO101WorkshopEnv(render_mode=None, control_hz=60)
    try:
        env.reset()
        gripper_range = env.model.jnt_range[env._joint_ids[-1]]
        close = _action(-1, -1)
        env.step_dynamics(close)
        assert env.data.ctrl[-1] == gripper_range[0]
        idle = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        for _ in range(90):
            env.step_dynamics(idle)
        assert env.joint_positions[-1] < gripper_range[0] + 0.05

        open_gripper = _action(-1)
        env.step_dynamics(open_gripper)
        assert env.data.ctrl[-1] == gripper_range[1]
        for _ in range(120):
            env.step_dynamics(idle)
        assert env.joint_positions[-1] > gripper_range[1] - 0.05
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
        observation, _, _, _, _ = env.step_dynamics(_action(0))
        assert env.observation_space.contains(observation)
        assert env.data.time > 0
    finally:
        env.close()
