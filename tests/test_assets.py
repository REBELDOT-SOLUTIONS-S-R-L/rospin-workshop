from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

from rospin_workshop.collision import compose_robot_collisions
from rospin_workshop.env import JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = ROOT / "assets/robots/so101"
OBJECT_DIR = ROOT / "assets/objects"
URDF_PATH = ROBOT_DIR / "so101_new_calib.urdf"
MJCF_PATH = ROOT / "src/rospin_workshop/models/so101_workshop.xml"


def test_compose_persists_data_to_explicit_host_bind() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    volumes = compose["services"]["workshop"]["volumes"]
    data_mount = next(
        volume
        for volume in volumes
        if isinstance(volume, dict) and volume.get("target") == "/workspace/data"
    )

    assert data_mount == {
        "type": "bind",
        "source": "${ROSPIN_HOST_DATA_DIR:-./data}",
        "target": "/workspace/data",
        "bind": {"create_host_path": False},
    }
    powershell = (ROOT / "scripts/workshop.ps1").read_text(encoding="utf-8")
    shell = (ROOT / "scripts/workshop.sh").read_text(encoding="utf-8")
    assert '$env:ROSPIN_HOST_DATA_DIR = $hostDataRoot' in powershell
    assert "export ROSPIN_HOST_DATA_DIR" in shell


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


def test_usd_derived_task_object_assets_are_vendored() -> None:
    manifest = json.loads(
        (OBJECT_DIR / "object_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert set(manifest["objects"]) == {"cube_green", "oala_cuburi"}
    expected = {
        "cube_green": {
            "source": "cubes/cube_green.usd",
            "source_sha256": (
                "5c1ad51bba5e3db6e04c9b083c727e4bcf89c947ea4798fa17318f7a8dabfdc1"
            ),
            "bounds_m": [[-0.0125, -0.0125, -0.0125], [0.0125, 0.0125, 0.0125]],
            "triangles": 12,
        },
        "oala_cuburi": {
            "source": "Oala cuburi.usd",
            "source_sha256": (
                "bf80a38d982eafc425147129fa8d7471588fa42f1dd4d3540bae5f977c18837c"
            ),
            "bounds_m": [
                [-0.0875, -0.087458504, 0.0],
                [0.0875, 0.087458504, 0.09],
            ],
            "triangles": 762,
        },
    }
    for name, values in expected.items():
        entry = manifest["objects"][name]
        assert entry["source"] == values["source"]
        assert entry["source_sha256"] == values["source_sha256"]
        assert entry["bounds_m"] == values["bounds_m"]
        assert entry["triangles"] == values["triangles"]
        generated = OBJECT_DIR / entry["generated"]
        assert hashlib.sha256(generated.read_bytes()).hexdigest() == entry[
            "generated_sha256"
        ]


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
        [0.0, 0.35, 0.7545030483],
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
        [0.2, 1.0, 1.38],
    )
    perspective_rotation = np.empty(9)
    mujoco.mju_quat2Mat(
        perspective_rotation,
        workshop_model.cam_quat[perspective_id],
    )
    perspective_forward = -perspective_rotation.reshape(3, 3)[:, 2]
    assert perspective_forward[0] < 0
    assert perspective_forward[1] < 0
    assert perspective_forward[2] < 0

    table_id = mujoco.mj_name2id(
        workshop_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "table",
    )
    np.testing.assert_allclose(workshop_model.geom_pos[table_id, :2], [0.0, 0.0])
    np.testing.assert_allclose(
        workshop_model.geom_size[table_id],
        [0.375, 0.375, 0.375],
    )
    table_material_id = workshop_model.geom_matid[table_id]
    np.testing.assert_allclose(
        workshop_model.mat_rgba[table_material_id],
        [1.0, 1.0, 1.0, 1.0],
    )
    assert workshop_model.mat_specular[table_material_id] == 0
    assert workshop_model.mat_shininess[table_material_id] == 0
    assert workshop_model.mat_reflectance[table_material_id] == 0
    assert workshop_model.mat_emission[table_material_id] == 0.25

    robot_material_id = mujoco.mj_name2id(
        workshop_model,
        mujoco.mjtObj.mjOBJ_MATERIAL,
        "printed_yellow",
    )
    np.testing.assert_allclose(
        workshop_model.mat_rgba[robot_material_id],
        [248 / 255, 139 / 255, 23 / 255, 1.0],
        atol=5e-8,
    )
    assert workshop_model.mat_specular[robot_material_id] == 0
    assert workshop_model.mat_shininess[robot_material_id] == 0
    assert workshop_model.mat_reflectance[robot_material_id] == 0

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


def test_robot_collision_geometry_is_inset_and_visual_meshes_do_not_collide() -> None:
    workshop_xml = MJCF_PATH.read_text(encoding="utf-8").replace(
        "SO101_MESH_DIR", str(ROBOT_DIR / "assets")
    )
    model = mujoco.MjModel.from_xml_string(compose_robot_collisions(workshop_xml))
    robot_bodies = {
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "lower_arm_link",
        "wrist_link",
        "gripper_link",
        "moving_jaw_so101_v1_link",
    }

    visual_ids: list[int] = []
    collision_ids: list[int] = []
    for body_name in robot_bodies:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_geoms = np.flatnonzero(model.geom_bodyid == body_id)
        assert len(body_geoms) > 0
        for geom_id in body_geoms:
            if model.geom_group[geom_id] == 1:
                assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
                assert model.geom_contype[geom_id] == 0
                assert model.geom_conaffinity[geom_id] == 0
                visual_ids.append(int(geom_id))
            elif model.geom_group[geom_id] == 4:
                assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
                assert model.geom_contype[geom_id] == 2
                assert model.geom_conaffinity[geom_id] == 1
                assert model.geom_margin[geom_id] == 0
                assert model.geom_rgba[geom_id, 3] == 0
                collision_ids.append(int(geom_id))
            else:
                raise AssertionError(f"Unexpected robot geom group: {geom_id}")

    catalogue = json.loads(
        (ROOT / "src/rospin_workshop/models/so101_collision_boxes.json").read_text(
            encoding="utf-8"
        )
    )
    source = ET.fromstring(workshop_xml)
    expected_collisions = sum(
        len(catalogue["meshes"][geom.attrib["mesh"]]["boxes"])
        for geom in source.findall(".//geom[@class='visual']")
    )
    assert len(visual_ids) == 17
    assert len(collision_ids) == expected_collisions
    assert len(collision_ids) > len(visual_ids)


def test_collision_catalogue_matches_the_rendered_stl_assets() -> None:
    catalogue = json.loads(
        (ROOT / "src/rospin_workshop/models/so101_collision_boxes.json").read_text(
            encoding="utf-8"
        )
    )
    assert catalogue["schema_version"] == 1
    assert catalogue["pitch_m"] == 0.002
    source_meshes = {
        mesh.attrib["name"]: mesh.attrib["file"]
        for mesh in ET.parse(MJCF_PATH).getroot().findall("./asset/mesh")
    }
    assert set(catalogue["meshes"]) == set(source_meshes)

    for name, filename in source_meshes.items():
        entry = catalogue["meshes"][name]
        path = ROBOT_DIR / "assets" / filename
        assert entry["source"] == filename
        assert entry["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["boxes"]
        for box in entry["boxes"]:
            assert len(box["center"]) == 3
            assert len(box["half_size"]) == 3
            assert all(value >= catalogue["pitch_m"] / 2 for value in box["half_size"])
