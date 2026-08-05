from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


TASK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


@dataclass(frozen=True)
class SpawnWorkspace:
    x: tuple[float, float]
    y: tuple[float, float]


@dataclass(frozen=True)
class TaskObject:
    name: str
    catalog_id: str
    pose: Pose
    spawn: SpawnWorkspace | None


@dataclass(frozen=True)
class SuccessCondition:
    type: str
    values: Mapping[str, Any]


@dataclass(frozen=True)
class TaskDefinition:
    schema_version: int
    id: str
    title: str
    instruction: str
    objects: tuple[TaskObject, ...]
    success_hold_seconds: float
    success_conditions: tuple[SuccessCondition, ...]
    timeout_seconds: float
    dataset_description: str

    def public_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "instruction": self.instruction,
            "timeout_seconds": self.timeout_seconds,
            "success_hold_seconds": self.success_hold_seconds,
        }


@dataclass(frozen=True)
class ObjectCatalogEntry:
    dynamic: bool
    half_extents: tuple[float, float, float] | None = None


OBJECT_CATALOG: dict[str, ObjectCatalogEntry] = {
    "cube_green_usd": ObjectCatalogEntry(
        dynamic=True,
        half_extents=(0.0125, 0.0125, 0.0125),
    ),
    "bowl_oala_usd": ObjectCatalogEntry(dynamic=False),
}


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return dict(value)


def _only_keys(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{context} has unknown fields: {names}")


def _required_string(value: Mapping[str, Any], name: str, context: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{context}.{name} must be a non-empty string")
    return result.strip()


def _finite_float(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "a positive finite number" if positive else "finite"
        raise ValueError(f"{context} must be {qualifier}")
    return result


def _vector(value: Any, length: int, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{context} must be a {length}-element list")
    return tuple(_finite_float(item, f"{context}[{index}]") for index, item in enumerate(value))


def _parse_pose(value: Any, context: str) -> Pose:
    pose = _mapping(value, context)
    _only_keys(pose, {"position", "quaternion"}, context)
    position = _vector(pose.get("position"), 3, f"{context}.position")
    quaternion = _vector(pose.get("quaternion"), 4, f"{context}.quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError(f"{context}.quaternion cannot be zero")
    if not math.isclose(norm, 1.0, abs_tol=1e-4):
        raise ValueError(f"{context}.quaternion must be normalized")
    return Pose(
        position=(position[0], position[1], position[2]),
        quaternion=(quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
    )


def _parse_spawn_workspace(value: Any, context: str) -> SpawnWorkspace:
    workspace = _mapping(value, context)
    _only_keys(workspace, {"x", "y"}, context)
    x = _vector(workspace.get("x"), 2, f"{context}.x")
    y = _vector(workspace.get("y"), 2, f"{context}.y")
    if x[0] >= x[1]:
        raise ValueError(f"{context}.x minimum must be less than its maximum")
    if y[0] >= y[1]:
        raise ValueError(f"{context}.y minimum must be less than its maximum")
    return SpawnWorkspace(x=(x[0], x[1]), y=(y[0], y[1]))


def _parse_object(value: Any, index: int) -> TaskObject:
    context = f"objects[{index}]"
    item = _mapping(value, context)
    _only_keys(item, {"name", "catalog_id", "pose", "spawn"}, context)
    name = _required_string(item, "name", context)
    if not TASK_ID_PATTERN.fullmatch(name):
        raise ValueError(f"{context}.name must use lowercase letters, digits, and underscores")
    catalog_id = _required_string(item, "catalog_id", context)
    if catalog_id not in OBJECT_CATALOG:
        raise ValueError(f"{context}.catalog_id references unknown object {catalog_id!r}")
    spawn = (
        _parse_spawn_workspace(item["spawn"], f"{context}.spawn")
        if "spawn" in item
        else None
    )
    if spawn is not None and not OBJECT_CATALOG[catalog_id].dynamic:
        raise ValueError(f"{context}.spawn requires a dynamic catalogue object")
    return TaskObject(
        name=name,
        catalog_id=catalog_id,
        pose=_parse_pose(item.get("pose"), f"{context}.pose"),
        spawn=spawn,
    )


def _parse_condition(value: Any, index: int, object_names: set[str]) -> SuccessCondition:
    context = f"success.conditions[{index}]"
    item = _mapping(value, context)
    condition_type = _required_string(item, "type", context)
    allowed: dict[str, set[str]] = {
        "object_fully_inside_region": {"type", "object", "region"},
        "gripper_open": {"type", "minimum_fraction"},
        "body_speed_below": {"type", "object", "linear_mps", "angular_rps"},
    }
    if condition_type not in allowed:
        raise ValueError(f"{context}.type is unsupported: {condition_type!r}")
    _only_keys(item, allowed[condition_type], context)

    values: dict[str, Any] = {}
    if condition_type in {"object_fully_inside_region", "body_speed_below"}:
        object_name = _required_string(item, "object", context)
        if object_name not in object_names:
            raise ValueError(f"{context}.object references unknown instance {object_name!r}")
        values["object"] = object_name
    if condition_type == "object_fully_inside_region":
        region = _required_string(item, "region", context)
        owner = region.split(".", 1)[0]
        if "." not in region or owner not in object_names:
            raise ValueError(f"{context}.region must reference an object region")
        values["region"] = region
    elif condition_type == "gripper_open":
        minimum = _finite_float(item.get("minimum_fraction"), f"{context}.minimum_fraction")
        if not 0 <= minimum <= 1:
            raise ValueError(f"{context}.minimum_fraction must be between 0 and 1")
        values["minimum_fraction"] = minimum
    elif condition_type == "body_speed_below":
        values["linear_mps"] = _finite_float(item.get("linear_mps"), f"{context}.linear_mps", positive=True)
        values["angular_rps"] = _finite_float(item.get("angular_rps"), f"{context}.angular_rps", positive=True)
    return SuccessCondition(type=condition_type, values=values)


def load_task(path: Path) -> TaskDefinition:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    root = _mapping(raw, str(path))
    _only_keys(
        root,
        {"schema_version", "id", "title", "instruction", "objects", "success", "episode", "dataset"},
        str(path),
    )
    if root.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    task_id = _required_string(root, "id", str(path))
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(f"{path}: id must use lowercase letters, digits, and underscores")
    if path.stem != task_id:
        raise ValueError(f"{path}: filename must match task id {task_id!r}")

    objects_raw = root.get("objects")
    if not isinstance(objects_raw, list) or not objects_raw:
        raise ValueError(f"{path}: objects must be a non-empty list")
    objects = tuple(_parse_object(item, index) for index, item in enumerate(objects_raw))
    object_names = [item.name for item in objects]
    if len(set(object_names)) != len(object_names):
        raise ValueError(f"{path}: object instance names must be unique")

    success = _mapping(root.get("success"), "success")
    _only_keys(success, {"hold_seconds", "conditions"}, "success")
    conditions_raw = success.get("conditions")
    if not isinstance(conditions_raw, list) or not conditions_raw:
        raise ValueError("success.conditions must be a non-empty list")
    conditions = tuple(
        _parse_condition(item, index, set(object_names))
        for index, item in enumerate(conditions_raw)
    )

    episode = _mapping(root.get("episode"), "episode")
    _only_keys(episode, {"timeout_seconds"}, "episode")
    dataset = _mapping(root.get("dataset"), "dataset")
    _only_keys(dataset, {"description"}, "dataset")

    return TaskDefinition(
        schema_version=1,
        id=task_id,
        title=_required_string(root, "title", str(path)),
        instruction=_required_string(root, "instruction", str(path)),
        objects=objects,
        success_hold_seconds=_finite_float(success.get("hold_seconds"), "success.hold_seconds", positive=True),
        success_conditions=conditions,
        timeout_seconds=_finite_float(episode.get("timeout_seconds"), "episode.timeout_seconds", positive=True),
        dataset_description=_required_string(dataset, "description", "dataset"),
    )


class TaskRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {self.root}")
        definitions: dict[str, TaskDefinition] = {}
        for path in sorted(self.root.glob("*.yaml")):
            task = load_task(path)
            if task.id in definitions:
                raise ValueError(f"Duplicate task id: {task.id}")
            definitions[task.id] = task
        if not definitions:
            raise ValueError(f"Task directory contains no YAML tasks: {self.root}")
        self._definitions = definitions

    def get(self, task_id: str) -> TaskDefinition:
        try:
            return self._definitions[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task: {task_id}") from exc

    def list(self) -> list[dict[str, Any]]:
        return [task.public_metadata() for task in self._definitions.values()]

    def only(self) -> TaskDefinition | None:
        return next(iter(self._definitions.values())) if len(self._definitions) == 1 else None


def _numbers(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _cube_body(instance: TaskObject) -> ET.Element:
    body = ET.Element(
        "body",
        {
            "name": f"task_{instance.name}",
            "pos": _numbers(instance.pose.position),
            "quat": _numbers(instance.pose.quaternion),
        },
    )
    ET.SubElement(body, "freejoint", {"name": f"task_{instance.name}_free"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"task_{instance.name}_visual",
            "type": "mesh",
            "mesh": "task_cube_green_mesh",
            "material": "task_cube_green",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "mass": "0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"task_{instance.name}_geom",
            "type": "box",
            "size": "0.0125 0.0125 0.0125",
            "mass": "0.02",
            "friction": "1.2 0.02 0.002",
            "rgba": "0 0 0 0",
            "contype": "1",
            "conaffinity": "1",
            "group": "4",
        },
    )
    return body


def _bowl_body(instance: TaskObject) -> ET.Element:
    body = ET.Element(
        "body",
        {
            "name": f"task_{instance.name}",
            "pos": _numbers(instance.pose.position),
            "quat": _numbers(instance.pose.quaternion),
        },
    )
    common = {
        "contype": "1",
        "conaffinity": "1",
        "friction": "1.0 0.01 0.001",
        "rgba": "0 0 0 0",
        "group": "4",
    }
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"task_{instance.name}_visual",
            "type": "mesh",
            "mesh": "task_bowl_oala_mesh",
            "material": "task_bowl_black",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )

    # The bowl floor occupies z=0..3 mm and tapers from 70 mm to 67.5 mm.
    # Keep this contact cylinder inset from both rendered faces and its edge.
    ET.SubElement(
        body,
        "geom",
        {
            **common,
            "name": f"task_{instance.name}_base",
            "type": "cylinder",
            "pos": "0 0 0.0015",
            "size": "0.0665 0.0013",
        },
    )

    # The USD bowl is a 3 mm thick tapered shell. These cylinders follow its
    # wall centreline, inset from the visual surfaces and both rims, so contact
    # geometry never extends beyond the rendered mesh.
    segments = 128
    bottom_radius = 0.0693483
    top_radius = 0.0856784
    bottom_height = 0.0045
    top_height = 0.0885
    for index in range(segments):
        angle = 2 * math.pi * index / segments
        cosine = math.cos(angle)
        sine = math.sin(angle)
        ET.SubElement(
            body,
            "geom",
            {
                **common,
                "name": f"task_{instance.name}_wall_{index}",
                "type": "cylinder",
                "fromto": (
                    f"{bottom_radius * cosine:.9g} "
                    f"{bottom_radius * sine:.9g} {bottom_height:.9g} "
                    f"{top_radius * cosine:.9g} "
                    f"{top_radius * sine:.9g} {top_height:.9g}"
                ),
                "size": "0.0011",
            },
        )
    ET.SubElement(
        body,
        "site",
        {
            "name": f"task_{instance.name}__interior",
            "type": "cylinder",
            "pos": "0 0 0.0265",
            "size": "0.07 0.0335",
            "rgba": "0.1 0.5 1 0.08",
            "group": "3",
        },
    )
    return body


def compose_task_model(base_xml: str, task: TaskDefinition) -> str:
    root = ET.fromstring(base_xml)
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Base MuJoCo model is missing asset or worldbody")
    if asset.find("material[@name='task_cube_green']") is None:
        ET.SubElement(
            asset,
            "material",
            {"name": "task_cube_green", "rgba": "0.05 0.8 0.05 1"},
        )
    if asset.find("material[@name='task_bowl_black']") is None:
        ET.SubElement(
            asset,
            "material",
            {"name": "task_bowl_black", "rgba": "0.035 0.035 0.04 1"},
        )
    if asset.find("mesh[@name='task_cube_green_mesh']") is None:
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": "task_cube_green_mesh",
                "file": "WORKSHOP_OBJECT_DIR/cube_green.obj",
            },
        )
    if asset.find("mesh[@name='task_bowl_oala_mesh']") is None:
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": "task_bowl_oala_mesh",
                "file": "WORKSHOP_OBJECT_DIR/oala_cuburi.obj",
            },
        )

    builders = {
        "cube_green_usd": _cube_body,
        "bowl_oala_usd": _bowl_body,
    }
    for instance in task.objects:
        worldbody.append(builders[instance.catalog_id](instance))
    return ET.tostring(root, encoding="unicode")
