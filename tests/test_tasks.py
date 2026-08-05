from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.controller import WorkshopController
from rospin_workshop.env import ACTION_NAMES, SO101WorkshopEnv
from rospin_workshop.tasks import TaskRegistry, load_task


ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = ROOT / "tasks"


def test_cube_task_loads_from_strict_yaml() -> None:
    registry = TaskRegistry(TASKS_ROOT)
    task = registry.get("cube_in_bowl")

    assert task.title == "Put the cube in the bowl"
    assert task.dataset_description == "Put the red cube into the blue bowl"
    assert task.timeout_seconds == 20
    assert [item.catalog_id for item in task.objects] == [
        "cube_40mm_red",
        "bowl_120mm_blue",
    ]
    assert registry.only() == task


def test_task_loader_rejects_unknown_fields_and_filename_mismatch(tmp_path) -> None:
    source = (TASKS_ROOT / "cube_in_bowl.yaml").read_text(encoding="utf-8")
    bad_field = tmp_path / "cube_in_bowl.yaml"
    bad_field.write_text(source + "\nunknown_field: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_task(bad_field)

    wrong_name = tmp_path / "different_name.yaml"
    wrong_name.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="filename must match"):
        load_task(wrong_name)


def test_task_scene_compiles_and_reset_restores_objects() -> None:
    task = TaskRegistry(TASKS_ROOT).get("cube_in_bowl")
    env = SO101WorkshopEnv(task=task, render_mode=None)
    try:
        env.reset()
        cube_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "task_cube"
        )
        bowl_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "task_bowl"
        )
        assert cube_id >= 0
        assert bowl_id >= 0
        initial_cube = env.data.xpos[cube_id].copy()

        task_collision_ids = [
            geom_id
            for geom_id in range(env.model.ngeom)
            if (
                mujoco.mj_id2name(
                    env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
                or ""
            ).startswith("task_")
            and env.model.geom_contype[geom_id] != 0
        ]
        assert task_collision_ids
        assert all(
            env.model.geom_rgba[geom_id, 3] == 1
            for geom_id in task_collision_ids
        )
        assert all(
            env.model.geom_margin[geom_id] == 0
            for geom_id in task_collision_ids
        )

        env.set_task_object_pose("cube", np.array([0.0, 0.0, 1.1]))
        assert not np.allclose(env.data.xpos[cube_id], initial_cube)
        env.reset()
        np.testing.assert_allclose(env.data.xpos[cube_id], initial_cube)

        snapshot = env.simulation_snapshot()
        assert np.asarray(snapshot["qpos"]).shape == (env.model.nq,)
    finally:
        env.close()


def test_success_requires_containment_release_settling_and_hold() -> None:
    original = TaskRegistry(TASKS_ROOT).get("cube_in_bowl")
    task = replace(original, success_hold_seconds=0.08)
    env = SO101WorkshopEnv(task=task, render_mode=None, control_hz=50)
    idle = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    try:
        env.reset()
        region_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "task_bowl__interior"
        )
        env.set_task_object_pose("cube", env.data.site_xpos[region_id].copy())

        # A closed gripper prevents success even with the cube in the bowl.
        env.data.qpos[env._qpos_indices[-1]] = env.joint_ranges[-1, 0]
        env.data.ctrl[-1] = env.joint_ranges[-1, 0]
        mujoco.mj_forward(env.model, env.data)
        for _ in range(8):
            env.step_dynamics(idle)
        assert env.task_status()["success"] is False

        env.data.qpos[env._qpos_indices[-1]] = env.joint_ranges[-1, 1]
        env.data.ctrl[-1] = env.joint_ranges[-1, 1]
        mujoco.mj_forward(env.model, env.data)
        for _ in range(10):
            env.step_dynamics(idle)
        assert env.task_status()["success"] is True
    finally:
        env.close()


class _FakeRecorder:
    def __init__(self) -> None:
        self.recording = True
        self.saved: bool | None = None

    def stop_episode(self, *, save: bool = True) -> int:
        self.recording = False
        self.saved = save
        return 12


def test_controller_auto_saves_success_and_resets(tmp_path) -> None:
    registry = TaskRegistry(TASKS_ROOT)
    task = registry.get("cube_in_bowl")
    controller = WorkshopController(
        RuntimeConfig(data_root=tmp_path, tasks_root=TASKS_ROOT),
        task_registry=registry,
    )
    env = SO101WorkshopEnv(task=task, render_mode=None)
    recorder = _FakeRecorder()
    try:
        env.reset()
        env._task_success_latched = True
        controller.env = env
        controller._selected_task = task
        controller._recorder = recorder  # type: ignore[assignment]
        controller._episode_started_at = time.monotonic()

        controller._update_episode_lifecycle()

        assert recorder.saved is True
        assert controller._last_episode_outcome == "success"
        assert controller._episode_started_at is None
        assert env.task_status()["success"] is False
        assert controller._render_requested.is_set()
    finally:
        env.close()
