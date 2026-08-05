from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from rospin_workshop.config import RuntimeConfig
from rospin_workshop.controller import WorkshopController
from rospin_workshop.env import ACTION_NAMES, HOME_JOINT_POSITIONS, SO101WorkshopEnv
from rospin_workshop.tasks import TaskRegistry, load_task


ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = ROOT / "tasks"


def test_cube_task_loads_from_strict_yaml() -> None:
    registry = TaskRegistry(TASKS_ROOT)
    task = registry.get("cube_in_bowl")

    assert task.title == "Put the green cube in the bowl"
    assert task.dataset_description == "Put the green cube into the black bowl"
    assert task.timeout_seconds == 20
    assert [item.catalog_id for item in task.objects] == [
        "cube_green_usd",
        "bowl_oala_usd",
    ]
    assert task.objects[0].spawn is not None
    assert task.objects[0].spawn.x == (-0.15, 0.03)
    assert task.objects[0].spawn.y == (0.07, 0.20)
    assert task.objects[1].spawn is None
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

    reversed_spawn = tmp_path / "cube_in_bowl.yaml"
    reversed_spawn.write_text(
        source.replace("x: [-0.15, 0.03]", "x: [0.03, -0.15]"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="minimum must be less"):
        load_task(reversed_spawn)


def test_task_scene_compiles_and_reset_restores_objects() -> None:
    task = TaskRegistry(TASKS_ROOT).get("cube_in_bowl")
    env = SO101WorkshopEnv(task=task, render_mode=None)
    try:
        env.reset(seed=17)
        np.testing.assert_allclose(
            env.joint_positions,
            HOME_JOINT_POSITIONS,
            atol=1e-6,
        )
        cube_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "task_cube"
        )
        bowl_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "task_bowl"
        )
        assert cube_id >= 0
        assert bowl_id >= 0
        initial_cube = env.data.xpos[cube_id].copy()
        assert -0.15 <= initial_cube[0] <= 0.03
        assert 0.07 <= initial_cube[1] <= 0.20
        np.testing.assert_allclose(initial_cube[2], 0.767903)
        np.testing.assert_allclose(
            env.data.xpos[bowl_id],
            [-0.2833, 0.15, 0.7601],
        )

        cube_visual_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "task_cube_visual"
        )
        bowl_visual_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "task_bowl_visual"
        )
        for visual_id in (cube_visual_id, bowl_visual_id):
            assert env.model.geom_type[visual_id] == mujoco.mjtGeom.mjGEOM_MESH
            assert env.model.geom_group[visual_id] == 1
            assert env.model.geom_contype[visual_id] == 0
            assert env.model.geom_rgba[visual_id, 3] == 1

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
        assert len(task_collision_ids) == 130
        assert all(
            env.model.geom_rgba[geom_id, 3] == 0
            for geom_id in task_collision_ids
        )
        assert all(
            env.model.geom_margin[geom_id] == 0
            for geom_id in task_collision_ids
        )
        cube_geom_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "task_cube_geom"
        )
        assert env.model.geom_type[cube_geom_id] == mujoco.mjtGeom.mjGEOM_BOX
        np.testing.assert_allclose(
            env.model.geom_size[cube_geom_id],
            [0.0125, 0.0125, 0.0125],
        )

        env.set_task_object_pose("cube", np.array([0.0, 0.0, 1.1]))
        assert not np.allclose(env.data.xpos[cube_id], initial_cube)
        env.reset(seed=17)
        np.testing.assert_allclose(env.data.xpos[cube_id], initial_cube)
        env.reset(seed=18)
        assert not np.allclose(env.data.xpos[cube_id, :2], initial_cube[:2])

        # The home pose now reaches into the configured workspace. Placement
        # remains deterministic and resamples any location touching the robot.
        env.reset(seed=13)
        assert -0.15 <= env.data.xpos[cube_id, 0] <= 0.03
        assert 0.07 <= env.data.xpos[cube_id, 1] <= 0.20
        assert all(
            cube_id
            not in (
                env.model.geom_bodyid[contact.geom1],
                env.model.geom_bodyid[contact.geom2],
            )
            for contact in env.data.contact
        )

        snapshot = env.simulation_snapshot()
        assert np.asarray(snapshot["qpos"]).shape == (env.model.nq,)
    finally:
        env.close()


def test_task_collisions_are_contained_by_the_source_visual_volumes() -> None:
    def obj_vertices(name: str) -> np.ndarray:
        lines = (ROOT / "assets/objects" / name).read_text(
            encoding="utf-8"
        ).splitlines()
        return np.asarray(
            [
                [float(value) for value in line.split()[1:4]]
                for line in lines
                if line.startswith("v ")
            ],
            dtype=np.float64,
        )

    task = TaskRegistry(TASKS_ROOT).get("cube_in_bowl")
    env = SO101WorkshopEnv(task=task, render_mode=None)
    try:
        env.reset()
        cube_vertices = obj_vertices("cube_green.obj")
        cube_geom_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "task_cube_geom"
        )
        cube_half_size = env.model.geom_size[cube_geom_id]
        np.testing.assert_allclose(cube_vertices.min(axis=0), -cube_half_size)
        np.testing.assert_allclose(cube_vertices.max(axis=0), cube_half_size)

        bowl_vertices = obj_vertices("oala_cuburi.obj")
        heights = np.unique(np.round(bowl_vertices[:, 2], 9))
        np.testing.assert_allclose(heights, [0.0, 0.003, 0.09])
        radii = np.linalg.norm(bowl_vertices[:, :2], axis=1)
        bottom_outer = radii[np.isclose(bowl_vertices[:, 2], heights[0])].min()
        bottom_inner = radii[np.isclose(bowl_vertices[:, 2], heights[1])].max()
        top_radii = radii[np.isclose(bowl_vertices[:, 2], heights[2])]
        top_midpoint = (top_radii.min() + top_radii.max()) / 2
        top_inner = top_radii[top_radii < top_midpoint].max()
        top_outer = top_radii[top_radii > top_midpoint].min()

        base_geom_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "task_bowl_base"
        )
        assert env.model.geom_type[base_geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        base_radius, base_half_height = env.model.geom_size[base_geom_id, :2]
        base_z = env.model.geom_pos[base_geom_id, 2]
        assert base_radius < bottom_inner
        assert base_z - base_half_height >= heights[0]
        assert base_z + base_half_height <= heights[1]

        # Densely sample each compiled cylinder's solid volume. The source bowl
        # consists of linearly tapered inner and outer surfaces, so every proxy
        # sample must remain between those surfaces as well as between its rims.
        clearance = []
        for index in range(128):
            geom_id = mujoco.mj_name2id(
                env.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"task_bowl_wall_{index}",
            )
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, env.model.geom_quat[geom_id])
            rotation = rotation.reshape(3, 3)
            radius, half_length = env.model.geom_size[geom_id, :2]
            samples = []
            for axial in np.linspace(-half_length, half_length, 9):
                for sample_radius in (0.0, radius / 2, radius):
                    for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                        local = np.array(
                            [
                                sample_radius * np.cos(angle),
                                sample_radius * np.sin(angle),
                                axial,
                            ]
                        )
                        samples.append(
                            env.model.geom_pos[geom_id] + rotation @ local
                        )
            samples = np.asarray(samples)
            sample_heights = samples[:, 2]
            sample_radii = np.linalg.norm(samples[:, :2], axis=1)
            inner_surface = bottom_inner + (top_inner - bottom_inner) * (
                (sample_heights - heights[1]) / (heights[2] - heights[1])
            )
            outer_surface = bottom_outer + (top_outer - bottom_outer) * (
                sample_heights / heights[2]
            )
            clearance.extend(sample_radii - inner_surface)
            clearance.extend(outer_surface - sample_radii)
            assert sample_heights.min() >= heights[1]
            assert sample_heights.max() <= heights[2]
        assert min(clearance) > 0.0004
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
