from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.controller import WorkshopController
from rospin_workshop.env import HOME_JOINT_POSITIONS, SO101WorkshopEnv
from rospin_workshop.trajectory.planner import TrajectoryPlanner
from rospin_workshop.trajectory.program import load_trajectory_program


def test_participant_program_loader_is_hot_loaded_and_confined(tmp_path) -> None:
    program_path = tmp_path / "example.py"
    program_path.write_text(
        "from rospin_workshop.trajectory import trajectory\n"
        "@trajectory(task='cube_in_bowl')\n"
        "def run(ctx):\n"
        "    ctx.move_home()\n",
        encoding="utf-8",
    )

    program = load_trajectory_program("example.py", tmp_path)
    assert program.task_id == "cube_in_bowl"
    assert program.name == "run"

    outside = tmp_path.parent / "outside.py"
    outside.write_text(program_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the trajectory directory"):
        load_trajectory_program(outside, tmp_path)


def test_loader_requires_exactly_one_decorated_program(tmp_path) -> None:
    empty = tmp_path / "empty.py"
    empty.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_trajectory_program(empty.name, tmp_path)


def test_cartesian_planner_reaches_goal_and_respects_joint_limits() -> None:
    planner = TrajectoryPlanner(control_hz=60)
    try:
        start = HOME_JOINT_POSITIONS.copy()
        target = planner._forward_position(start) + np.array([0.0, 0.0, 0.03])
        plan = planner.plan_linear(start, target, speed=0.05)

        assert len(plan.joint_targets) > 1
        final = plan.joint_targets[-1]
        np.testing.assert_allclose(
            planner._forward_position(final),
            target,
            atol=0.002,
        )
        assert np.all(final >= planner.joint_ranges[:, 0])
        assert np.all(final <= planner.joint_ranges[:, 1])
    finally:
        planner.close()


def test_trajectory_control_uses_rate_limited_joint_targets(tmp_path) -> None:
    controller = WorkshopController(RuntimeConfig(data_root=tmp_path))
    env = SO101WorkshopEnv(render_mode=None)
    controller.env = env
    try:
        env.reset()
        desired = env.joint_positions.astype(np.float64)
        desired[0] += 0.2
        controller.begin_trajectory_control()
        controller.set_trajectory_joint_targets(desired)
        recorded_action = controller._step_control()

        assert controller._active_control_source == "trajectory"
        assert recorded_action.shape == (6,)
        np.testing.assert_allclose(recorded_action, env.data.ctrl)
        assert 0 < env.data.ctrl[0] <= env.joint_step + 1e-10
    finally:
        controller.end_trajectory_control()
        env.close()


def test_runtime_config_has_hot_mounted_trajectory_root(tmp_path) -> None:
    config = RuntimeConfig(data_root=tmp_path, trajectories_root=Path("programs"))
    assert config.trajectories_root == Path("programs")
