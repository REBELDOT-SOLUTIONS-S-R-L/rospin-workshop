#!/usr/bin/env python3
"""Generate conservative SO-101 collision boxes from the rendered STL solids.

This is an offline asset-generation tool. Install ``trimesh``, ``scipy``,
``rtree``, and ``embreex`` before running it. The application itself only reads
the generated JSON and does not need those packages.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
MJCF_PATH = ROOT / "src/rospin_workshop/models/so101_workshop.xml"
MESH_DIR = ROOT / "assets/robots/so101/assets"
OUTPUT_PATH = ROOT / "src/rospin_workshop/models/so101_collision_boxes.json"
DEFAULT_PITCH = 0.002


def _mesh_files() -> dict[str, str]:
    root = ET.parse(MJCF_PATH).getroot()
    return {
        element.attrib["name"]: element.attrib["file"]
        for element in root.findall("./asset/mesh")
    }


def _solid_mesh(path: Path) -> trimesh.Trimesh:
    raw = trimesh.load_mesh(path, process=True, validate=True)
    components = [
        component
        for component in raw.split(only_watertight=True)
        if abs(float(component.volume)) > 1e-10
    ]
    if not components:
        raise ValueError(f"{path} has no closed, positive-volume component")
    return trimesh.util.concatenate(components)


def _greedy_boxes(
    cells: np.ndarray,
    order: tuple[int, int, int],
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    remaining = {tuple(int(value) for value in cell) for cell in cells}
    boxes: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    while remaining:
        seed = min(remaining, key=lambda point: tuple(point[axis] for axis in order))
        low = list(seed)
        high = [value + 1 for value in seed]
        for axis in order:
            while True:
                ranges = [range(low[index], high[index]) for index in range(3)]
                ranges[axis] = range(high[axis], high[axis] + 1)
                if all(tuple(point) in remaining for point in itertools.product(*ranges)):
                    high[axis] += 1
                else:
                    break
        for point in itertools.product(
            *(range(low[index], high[index]) for index in range(3))
        ):
            remaining.remove(tuple(point))
        boxes.append((tuple(low), tuple(high)))
    return boxes


def _interior_boxes(path: Path, pitch: float) -> list[dict[str, list[float]]]:
    mesh = _solid_mesh(path)
    candidate_points = mesh.voxelized(pitch).fill().points
    inside = mesh.contains(candidate_points)
    triangle_tree = mesh.triangles_tree
    half_pitch = pitch / 2

    # A cell is accepted only when its centre is inside the solid and its full
    # AABB is disjoint from every surface-triangle AABB. A continuous path from
    # that centre to anywhere in the cell therefore cannot cross the mesh
    # boundary, so the complete box is contained by the rendered STL volume.
    safe_points = []
    for point in candidate_points[inside]:
        bounds = np.concatenate((point - half_pitch, point + half_pitch))
        if not any(triangle_tree.intersection(bounds)):
            safe_points.append(point)
    if not safe_points:
        raise ValueError(f"{path} has no collision cells at pitch {pitch}")

    points = np.asarray(safe_points)
    origin = points.min(axis=0)
    cells = np.rint((points - origin) / pitch).astype(np.int64)
    decompositions = (
        _greedy_boxes(cells, order)
        for order in itertools.permutations((0, 1, 2))
    )
    boxes = min(decompositions, key=len)

    result: list[dict[str, list[float]]] = []
    for low, high in boxes:
        low_corner = origin + (np.asarray(low, dtype=np.float64) - 0.5) * pitch
        high_corner = origin + (np.asarray(high, dtype=np.float64) - 0.5) * pitch
        result.append(
            {
                "center": np.round((low_corner + high_corner) / 2, 9).tolist(),
                "half_size": np.round((high_corner - low_corner) / 2, 9).tolist(),
            }
        )
    return result


def generate(pitch: float) -> dict[str, object]:
    meshes: dict[str, object] = {}
    for name, filename in sorted(_mesh_files().items()):
        path = MESH_DIR / filename
        boxes = _interior_boxes(path, pitch)
        meshes[name] = {
            "source": filename,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "boxes": boxes,
        }
        print(f"{name}: {len(boxes)} boxes")
    return {
        "schema_version": 1,
        "method": "fully-contained interior voxels merged into boxes",
        "pitch_m": pitch,
        "meshes": meshes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitch", type=float, default=DEFAULT_PITCH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.pitch <= 0:
        parser.error("--pitch must be positive")

    generated = json.dumps(generate(args.pitch), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text(encoding="utf-8") != generated:
            raise SystemExit(f"Collision catalogue is stale: regenerate {OUTPUT_PATH}")
        print(f"Collision catalogue is current: {OUTPUT_PATH}")
        return
    OUTPUT_PATH.write_text(generated, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
