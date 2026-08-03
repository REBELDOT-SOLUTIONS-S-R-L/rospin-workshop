from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from rospin_workshop.env import JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = ROOT / "assets/robots/so101"
URDF_PATH = ROBOT_DIR / "so101_new_calib.urdf"
MJCF_PATH = ROOT / "src/rospin_workshop/models/so101_workshop.xml"


def test_hugging_face_so101_assets_are_vendored() -> None:
    assert (ROOT / "assets/scenes/scene.usd").is_file()
    assert URDF_PATH.is_file()
    manifest = json.loads(
        (ROOT / "src/rospin_workshop/models/asset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["sources"]["robot_bucket"] == ("hf://buckets/lerobot/robot-urdfs")
    assert manifest["sources"]["robot_urdf"] == (
        "/assets/robots/so101/so101_new_calib.urdf"
    )
    assert manifest["wrist_camera"]["simulation_resolution"] == [640, 480]
    mesh_files = sorted(path.name for path in (ROBOT_DIR / "assets").glob("*.stl"))
    assert mesh_files == manifest["robot"]["mesh_files"]
    assert all((ROBOT_DIR / "assets" / name).stat().st_size > 0 for name in mesh_files)


def test_every_urdf_visual_is_rendered_with_its_stl_and_transform() -> None:
    urdf = ET.parse(URDF_PATH).getroot()
    mjcf = ET.parse(MJCF_PATH).getroot()
    mesh_assets = {
        mesh.attrib["name"]: mesh.attrib["file"]
        for mesh in mjcf.findall("./asset/mesh")
    }

    source_meshes: set[str] = set()
    rendered_instances = 0
    for source_link in urdf.findall("link"):
        source_visuals = source_link.findall("visual")
        if not source_visuals:
            continue
        body = mjcf.find(f".//body[@name='{source_link.attrib['name']}']")
        assert body is not None
        rendered_visuals = [
            geom
            for geom in body.findall("geom")
            if geom.attrib.get("class") == "visual"
        ]
        assert len(rendered_visuals) == len(source_visuals)

        for source, rendered in zip(source_visuals, rendered_visuals, strict=True):
            source_file = Path(source.find("geometry/mesh").attrib["filename"]).name
            source_meshes.add(source_file)
            assert mesh_assets[rendered.attrib["mesh"]] == source_file
            origin = source.find("origin")
            np.testing.assert_allclose(
                np.fromstring(rendered.attrib.get("pos", "0 0 0"), sep=" "),
                np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" "),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.fromstring(rendered.attrib.get("euler", "0 0 0"), sep=" "),
                np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" "),
                atol=1e-5,
            )
            rendered_instances += 1

    assert source_meshes == set(mesh_assets.values())
    assert len(source_meshes) == 13
    assert rendered_instances == 17


def test_compiled_link_and_mesh_poses_match_mujoco_urdf_importer() -> None:
    mjcf = ET.parse(MJCF_PATH).getroot()
    assert mjcf.find("compiler").attrib["eulerseq"] == "XYZ"

    native_model = mujoco.MjModel.from_xml_path(str(URDF_PATH))
    workshop_xml = MJCF_PATH.read_text(encoding="utf-8").replace(
        "SO101_MESH_DIR", str(ROBOT_DIR / "assets")
    )
    workshop_model = mujoco.MjModel.from_xml_string(workshop_xml)
    robot_mount_id = mujoco.mj_name2id(
        workshop_model, mujoco.mjtObj.mjOBJ_BODY, "robot_mount"
    )
    np.testing.assert_allclose(
        workshop_model.body_pos[robot_mount_id],
        [0.0, 0.23, 0.7545030483],
    )
    expected_mount_quaternion = np.array(
        [np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)]
    )
    assert np.isclose(
        abs(np.dot(workshop_model.body_quat[robot_mount_id], expected_mount_quaternion)),
        1.0,
    )
    perspective_id = mujoco.mj_name2id(
        workshop_model, mujoco.mjtObj.mjOBJ_CAMERA, "perspective"
    )
    np.testing.assert_allclose(
        workshop_model.cam_pos[perspective_id],
        [0.0, 0.72, 1.22],
    )
    perspective_rotation = np.empty(9)
    mujoco.mju_quat2Mat(
        perspective_rotation,
        workshop_model.cam_quat[perspective_id],
    )
    perspective_forward = -perspective_rotation.reshape(3, 3)[:, 2]
    assert abs(perspective_forward[0]) < 1e-8
    assert perspective_forward[1] < 0
    assert perspective_forward[2] < 0

    for native_body_id in range(1, native_model.nbody):
        body_name = mujoco.mj_id2name(
            native_model, mujoco.mjtObj.mjOBJ_BODY, native_body_id
        )
        workshop_body_id = mujoco.mj_name2id(
            workshop_model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        assert workshop_body_id >= 0
        np.testing.assert_allclose(
            workshop_model.body_pos[workshop_body_id],
            native_model.body_pos[native_body_id],
            atol=1e-8,
        )
        assert np.isclose(
            abs(
                np.dot(
                    workshop_model.body_quat[workshop_body_id],
                    native_model.body_quat[native_body_id],
                )
            ),
            1.0,
            atol=1e-8,
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
    assert len(native_visual_ids) == len(workshop_visual_ids) == 17
    for native_geom_id, workshop_geom_id in zip(
        native_visual_ids, workshop_visual_ids, strict=True
    ):
        np.testing.assert_allclose(
            workshop_model.geom_pos[workshop_geom_id],
            native_model.geom_pos[native_geom_id],
            atol=1e-8,
        )
        assert np.isclose(
            abs(
                np.dot(
                    workshop_model.geom_quat[workshop_geom_id],
                    native_model.geom_quat[native_geom_id],
                )
            ),
            1.0,
            atol=1e-8,
        )


def test_urdf_and_mjcf_joint_limits_match() -> None:
    urdf = ET.parse(URDF_PATH).getroot()
    source_joints = {
        joint.attrib["name"]: joint
        for joint in urdf.findall("joint")
        if joint.attrib["type"] != "fixed"
    }
    assert set(source_joints) == set(JOINT_NAMES)

    mjcf = ET.parse(MJCF_PATH).getroot()
    rendered_joints = {
        joint.attrib["name"]: joint
        for joint in mjcf.findall(".//joint")
        if "name" in joint.attrib
    }
    for name in JOINT_NAMES:
        source_limit = source_joints[name].find("limit")
        np.testing.assert_allclose(
            np.fromstring(rendered_joints[name].attrib["range"], sep=" "),
            [
                float(source_limit.attrib["lower"]),
                float(source_limit.attrib["upper"]),
            ],
            atol=1e-5,
        )
