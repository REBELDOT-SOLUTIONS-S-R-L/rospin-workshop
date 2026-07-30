#!/usr/bin/env python3
"""Verify that the MuJoCo model matches the source scene, URDF, and STL files.

This tool is intentionally outside the runtime dependency set. Run it from the
repository root with ``python -m pip install '.[dev]'`` first.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from pxr import Sdf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "assets/scenes/scene.usd"
DEFAULT_ROBOT_DIR = ROOT / "assets/robots/so101"
CAMERA_ROBOT_DIR = ROOT / "assets/robots/SO-101_follower/so101_wrist_mount"
MANIFEST = ROOT / "src/rospin_workshop/models/asset_manifest.json"
MJCF = ROOT / "src/rospin_workshop/models/so101_workshop.xml"


def _bbox(stage: Usd.Stage, prim_path: str) -> list[list[float]]:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    bounds = cache.ComputeWorldBound(
        stage.GetPrimAtPath(prim_path)
    ).ComputeAlignedRange()
    return [list(bounds.GetMin()), list(bounds.GetMax())]


def _assert_quaternion_equivalent(
    actual: np.ndarray, expected: np.ndarray, *, atol: float = 1e-8
) -> None:
    """Compare unit quaternions while accepting the equivalent q/-q forms."""

    if not np.isclose(abs(float(np.dot(actual, expected))), 1.0, atol=atol):
        raise AssertionError(f"Quaternion mismatch: {actual} != {expected}")


def _set_joint_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
    positions: np.ndarray,
) -> None:
    for name, value in zip(joint_names, positions, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    scene = Usd.Stage.Open(str(DEFAULT_SCENE))
    if scene is None:
        raise RuntimeError(f"Could not open {DEFAULT_SCENE}")
    np.testing.assert_allclose(
        _bbox(scene, "/World/GroundPlane"), manifest["scene"]["ground_bbox"]
    )
    np.testing.assert_allclose(
        _bbox(scene, "/World/Table"), manifest["scene"]["table_bbox"]
    )

    robot_usda = CAMERA_ROBOT_DIR / "so101_wrist_mount_sdf_gripper_right_camera.usda"
    robot_layer = Sdf.Layer.FindOrOpen(str(robot_usda))
    camera = robot_layer.GetPrimAtPath(
        "/so101_new_calib/gripper_link/right_wrist_camera"
    )
    position = camera.properties["xformOp:translate"].default
    orientation = camera.properties["xformOp:orient"].default
    np.testing.assert_allclose(position, manifest["wrist_camera"]["source_position"])
    np.testing.assert_allclose(
        [orientation.GetReal(), *orientation.GetImaginary()],
        manifest["wrist_camera"]["source_orientation_wxyz"],
    )

    urdf_path = DEFAULT_ROBOT_DIR / "so101_new_calib.urdf"
    urdf = ET.parse(urdf_path).getroot()
    movable_by_name = {
        joint.attrib["name"]: joint
        for joint in urdf.findall("joint")
        if joint.attrib["type"] != "fixed"
    }
    expected_joints = manifest["robot"]["joints"]
    if set(movable_by_name) != set(expected_joints):
        raise AssertionError(f"URDF joints changed: {sorted(movable_by_name)}")

    mjcf = ET.parse(MJCF).getroot()
    compiler = mjcf.find("compiler")
    if compiler is None or compiler.attrib.get("eulerseq") != "XYZ":
        raise AssertionError("URDF rpy requires MuJoCo's extrinsic eulerseq='XYZ'")
    mjcf_joints = {
        joint.attrib["name"]: joint
        for joint in mjcf.findall(".//joint")
        if "name" in joint.attrib
    }
    for name in expected_joints:
        source_limit = movable_by_name[name].find("limit")
        source_range = [
            float(source_limit.attrib["lower"]),
            float(source_limit.attrib["upper"]),
        ]
        np.testing.assert_allclose(
            np.fromstring(mjcf_joints[name].attrib["range"], sep=" "),
            source_range,
            atol=1e-5,
        )

    mesh_assets = {
        mesh.attrib["name"]: mesh.attrib["file"]
        for mesh in mjcf.findall("./asset/mesh")
    }
    urdf_mesh_files: set[str] = set()
    source_visual_instances: list[tuple[str, str]] = []
    visual_instances = 0
    for source_link in urdf.findall("link"):
        visuals = source_link.findall("visual")
        if not visuals:
            continue
        body = mjcf.find(f".//body[@name='{source_link.attrib['name']}']")
        if body is None:
            raise AssertionError(f"Missing MJCF body: {source_link.attrib['name']}")
        rendered = [
            geom
            for geom in body.findall("geom")
            if geom.attrib.get("class") == "visual"
        ]
        if len(rendered) != len(visuals):
            raise AssertionError(
                f"{source_link.attrib['name']} has {len(visuals)} URDF visuals "
                f"but {len(rendered)} rendered MJCF meshes"
            )
        for source_visual, rendered_geom in zip(visuals, rendered, strict=True):
            origin = source_visual.find("origin")
            source_mesh = source_visual.find("geometry/mesh")
            source_file = Path(source_mesh.attrib["filename"]).name
            urdf_mesh_files.add(source_file)
            source_visual_instances.append((source_link.attrib["name"], source_file))
            if mesh_assets[rendered_geom.attrib["mesh"]] != source_file:
                raise AssertionError(
                    f"Wrong mesh for {source_link.attrib['name']}: {source_file}"
                )
            np.testing.assert_allclose(
                np.fromstring(rendered_geom.attrib.get("pos", "0 0 0"), sep=" "),
                np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" "),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.fromstring(rendered_geom.attrib.get("euler", "0 0 0"), sep=" "),
                np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" "),
                atol=1e-5,
            )
            visual_instances += 1

    # Literal xyz/rpy equality is insufficient: URDF and MJCF use different
    # Euler-sequence notation. Compile the source URDF with MuJoCo's native
    # importer and prove that every link and centered STL geom has the same
    # effective transform in the workshop model.
    native_model = mujoco.MjModel.from_xml_path(str(urdf_path))
    workshop_xml = MJCF.read_text(encoding="utf-8").replace(
        "SO101_MESH_DIR", str(DEFAULT_ROBOT_DIR / "assets")
    )
    workshop_model = mujoco.MjModel.from_xml_string(workshop_xml)

    for native_body_id in range(1, native_model.nbody):
        name = mujoco.mj_id2name(native_model, mujoco.mjtObj.mjOBJ_BODY, native_body_id)
        workshop_body_id = mujoco.mj_name2id(
            workshop_model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if workshop_body_id < 0:
            raise AssertionError(f"Missing compiled MJCF body: {name}")
        np.testing.assert_allclose(
            workshop_model.body_pos[workshop_body_id],
            native_model.body_pos[native_body_id],
            atol=1e-8,
        )
        _assert_quaternion_equivalent(
            workshop_model.body_quat[workshop_body_id],
            native_model.body_quat[native_body_id],
        )

    native_visual_ids = [
        geom_id
        for geom_id in range(native_model.ngeom)
        if native_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    workshop_visual_ids = [
        geom_id
        for geom_id in range(workshop_model.ngeom)
        if workshop_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        and workshop_model.geom_group[geom_id] == 1
    ]
    if not (
        len(native_visual_ids)
        == len(workshop_visual_ids)
        == len(source_visual_instances)
    ):
        raise AssertionError("Compiled visual instance counts differ")

    for (link_name, source_file), native_geom_id, workshop_geom_id in zip(
        source_visual_instances,
        native_visual_ids,
        workshop_visual_ids,
        strict=True,
    ):
        native_mesh_name = mujoco.mj_id2name(
            native_model,
            mujoco.mjtObj.mjOBJ_MESH,
            int(native_model.geom_dataid[native_geom_id]),
        )
        workshop_mesh_name = mujoco.mj_id2name(
            workshop_model,
            mujoco.mjtObj.mjOBJ_MESH,
            int(workshop_model.geom_dataid[workshop_geom_id]),
        )
        if native_mesh_name != Path(source_file).stem:
            raise AssertionError(
                f"Native importer mesh order changed at {link_name}/{source_file}"
            )
        if mesh_assets[workshop_mesh_name] != source_file:
            raise AssertionError(
                f"Workshop compiled wrong mesh at {link_name}/{source_file}"
            )
        np.testing.assert_allclose(
            workshop_model.geom_pos[workshop_geom_id],
            native_model.geom_pos[native_geom_id],
            atol=1e-8,
        )
        _assert_quaternion_equivalent(
            workshop_model.geom_quat[workshop_geom_id],
            native_model.geom_quat[native_geom_id],
        )

    home = np.fromstring(
        mjcf.find("./keyframe/key[@name='home']").attrib["qpos"], sep=" "
    )
    np.testing.assert_allclose(
        home, manifest["robot"]["home_joint_positions"], atol=1e-8
    )
    pose_samples = (
        np.zeros(len(expected_joints)),
        home,
        np.array([0.35, -0.7, 0.8, -0.6, 0.45, 0.4]),
    )
    native_data = mujoco.MjData(native_model)
    workshop_data = mujoco.MjData(workshop_model)
    base_id = mujoco.mj_name2id(workshop_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    for pose in pose_samples:
        _set_joint_positions(native_model, native_data, expected_joints, pose)
        _set_joint_positions(workshop_model, workshop_data, expected_joints, pose)
        base_position = workshop_data.xpos[base_id]
        base_rotation = workshop_data.xmat[base_id].reshape(3, 3)
        for native_geom_id, workshop_geom_id in zip(
            native_visual_ids, workshop_visual_ids, strict=True
        ):
            workshop_position_in_base = base_rotation.T @ (
                workshop_data.geom_xpos[workshop_geom_id] - base_position
            )
            workshop_rotation_in_base = base_rotation.T @ workshop_data.geom_xmat[
                workshop_geom_id
            ].reshape(3, 3)
            np.testing.assert_allclose(
                workshop_position_in_base,
                native_data.geom_xpos[native_geom_id],
                atol=1e-7,
            )
            np.testing.assert_allclose(
                workshop_rotation_in_base,
                native_data.geom_xmat[native_geom_id].reshape(3, 3),
                atol=1e-7,
            )

    expected_meshes = set(manifest["robot"]["mesh_files"])
    if (
        urdf_mesh_files != expected_meshes
        or set(mesh_assets.values()) != expected_meshes
    ):
        raise AssertionError("URDF, manifest, and MJCF mesh sets differ")
    missing_meshes = [
        name
        for name in expected_meshes
        if not (DEFAULT_ROBOT_DIR / "assets" / name).is_file()
    ]
    if missing_meshes:
        raise FileNotFoundError(f"Missing STL files: {missing_meshes}")

    wrist_camera = mjcf.find(".//camera[@name='wrist']")
    np.testing.assert_allclose(
        np.fromstring(wrist_camera.attrib["pos"], sep=" "),
        manifest["wrist_camera"]["mounted_position"],
    )
    np.testing.assert_allclose(
        np.fromstring(wrist_camera.attrib["quat"], sep=" "),
        manifest["wrist_camera"]["source_orientation_wxyz"],
    )
    return {
        "scene": str(DEFAULT_SCENE),
        "robot_urdf": str(urdf_path),
        "table_bbox": manifest["scene"]["table_bbox"],
        "wrist_camera": manifest["wrist_camera"],
        "joints": expected_joints,
        "unique_stl_meshes": len(urdf_mesh_files),
        "rendered_mesh_instances": visual_instances,
        "compiled_pose_samples": len(pose_samples),
        "home_joint_positions": home.tolist(),
        "status": "in sync",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
