from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from importlib.resources import files
from typing import Any


COLLISION_CATALOGUE = "models/so101_collision_boxes.json"


@lru_cache(maxsize=1)
def _catalogue() -> dict[str, Any]:
    resource = files("rospin_workshop").joinpath(COLLISION_CATALOGUE)
    value = json.loads(resource.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("meshes"), dict):
        raise ValueError("Invalid SO-101 collision catalogue")
    return value


def compose_robot_collisions(base_xml: str) -> str:
    """Add conservative collision boxes in each rendered mesh's local frame."""

    root = ET.fromstring(base_xml)
    catalogue = _catalogue()["meshes"]
    instance_index = 0
    for body in root.findall(".//body"):
        visuals = list(body.findall("./geom[@class='visual']"))
        for visual in visuals:
            mesh_name = visual.attrib.get("mesh")
            if mesh_name not in catalogue:
                raise ValueError(
                    f"Rendered robot mesh has no conservative collisions: {mesh_name!r}"
                )
            frame_attributes = {
                key: visual.attrib[key]
                for key in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis")
                if key in visual.attrib
            }
            frame = ET.SubElement(body, "frame", frame_attributes)
            boxes = catalogue[mesh_name].get("boxes")
            if not isinstance(boxes, list) or not boxes:
                raise ValueError(f"Collision mesh has no boxes: {mesh_name!r}")
            for box_index, box in enumerate(boxes):
                center = box["center"]
                half_size = box["half_size"]
                ET.SubElement(
                    frame,
                    "geom",
                    {
                        "name": (
                            f"robot_collision_{instance_index}_{mesh_name}_{box_index}"
                        ),
                        "class": "collision",
                        "type": "box",
                        "pos": " ".join(str(value) for value in center),
                        "size": " ".join(str(value) for value in half_size),
                    },
                )
            instance_index += 1
    if instance_index == 0:
        raise ValueError("SO-101 model contains no rendered robot meshes")
    return ET.tostring(root, encoding="unicode")
