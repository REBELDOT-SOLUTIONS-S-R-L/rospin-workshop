#!/usr/bin/env python3
"""Convert the workshop's source USD objects into deterministic MuJoCo OBJs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(
    "/home/roboticslab/IsaacTools/IsaacTasks/single_so101_cubes_leisaac/"
    "source/single_so101_cubes_leisaac/single_so101_cubes_leisaac/assets/objects"
)
OUTPUT_ROOT = ROOT / "assets/objects"


@dataclass(frozen=True)
class ObjectSource:
    name: str
    relative_path: str
    prim_path: str
    scale: float
    orient_outward: bool = False


SOURCES = (
    ObjectSource(
        "cube_green",
        "cubes/cube_green.usd",
        "/Cube",
        1.0,
        orient_outward=True,
    ),
    # The bowl's source stage is centimetre-authored/Y-up, while its original
    # scene reference resolves to this net scale with axes unchanged.
    ObjectSource("oala_cuburi", "Oala cuburi.usd", "/World/node_/mesh_", 0.001),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _obj_text(
    mesh: UsdGeom.Mesh,
    scale: float,
    *,
    orient_outward: bool,
) -> tuple[str, np.ndarray, int]:
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64) * scale
    counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get()]
    indices = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get()]
    lines = ["# Generated from the USD source recorded in object_manifest.json"]
    lines.extend(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}" for point in points)

    cursor = 0
    triangle_count = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        cursor += count
        for index in range(1, count - 1):
            # OBJ indices are one-based. Fan triangulation preserves the USD
            # polygon winding and is deterministic for both source meshes.
            triangle = (face[0], face[index], face[index + 1])
            if orient_outward:
                vertices = points[np.asarray(triangle)]
                normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                if np.dot(normal, vertices.mean(axis=0) - points.mean(axis=0)) < 0:
                    triangle = (triangle[0], triangle[2], triangle[1])
            triangle = tuple(vertex + 1 for vertex in triangle)
            lines.append(f"f {triangle[0]} {triangle[1]} {triangle[2]}")
            triangle_count += 1
    if cursor != len(indices):
        raise ValueError("USD face counts do not consume every face index")
    return "\n".join(lines) + "\n", points, triangle_count


def generate(source_root: Path) -> tuple[dict[str, str], dict[str, object]]:
    outputs: dict[str, str] = {}
    objects: dict[str, object] = {}
    for source in SOURCES:
        source_path = source_root / source.relative_path
        stage = Usd.Stage.Open(str(source_path))
        if stage is None:
            raise FileNotFoundError(source_path)
        prim = stage.GetPrimAtPath(source.prim_path)
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            raise ValueError(f"USD prim is not a mesh: {source.prim_path}")
        text, points, triangle_count = _obj_text(
            mesh,
            source.scale,
            orient_outward=source.orient_outward,
        )
        output_name = f"{source.name}.obj"
        output_bytes = text.encode("utf-8")
        outputs[output_name] = text

        mass = None
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        objects[source.name] = {
            "source": source.relative_path,
            "source_sha256": _sha256(source_path.read_bytes()),
            "source_prim": source.prim_path,
            "applied_scale": source.scale,
            "generated": output_name,
            "generated_sha256": _sha256(output_bytes),
            "bounds_m": [
                np.round(points.min(axis=0), 9).tolist(),
                np.round(points.max(axis=0), 9).tolist(),
            ],
            "triangles": triangle_count,
            "mass_kg": float(mass) if mass is not None else None,
        }
        print(f"{source.name}: {len(points)} vertices, {triangle_count} triangles")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_root": str(source_root),
        "objects": objects,
    }
    return outputs, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    outputs, manifest = generate(args.source_root)
    outputs["object_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if args.check:
        stale = [
            name
            for name, expected in outputs.items()
            if not (OUTPUT_ROOT / name).is_file()
            or (OUTPUT_ROOT / name).read_text(encoding="utf-8") != expected
        ]
        if stale:
            raise SystemExit(f"Generated object assets are stale: {', '.join(stale)}")
        print(f"Generated object assets are current: {OUTPUT_ROOT}")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        (OUTPUT_ROOT / name).write_text(text, encoding="utf-8")
    print(f"Wrote {len(outputs)} files to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
