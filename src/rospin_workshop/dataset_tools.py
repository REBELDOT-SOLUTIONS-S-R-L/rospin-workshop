from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def inspect_dataset(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobot v3.0, found {info.get('codebase_version')!r}"
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # LeRobot v3 info.json intentionally does not serialize repo_id. This
    # project records under a deterministic local/<directory> namespace.
    repo_id = f"local/{root.name}"
    dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")
    required = {
        "observation.state",
        "observation.eef_position",
        "observation.eef_orientation",
        "observation.images.wrist",
        "action",
    }
    missing = required.difference(dataset.features)
    if missing:
        raise ValueError(f"Dataset is missing required features: {sorted(missing)}")
    if dataset.num_episodes < 1 or dataset.num_frames < 1:
        raise ValueError("Dataset has no saved frames or episodes")
    sample = dataset[0]
    camera_key = "observation.images.wrist"
    expected_h, expected_w, _ = dataset.features[camera_key]["shape"]
    if tuple(sample[camera_key].shape) != (3, expected_h, expected_w):
        raise ValueError(f"Could not decode {camera_key} at its declared resolution")
    return {
        "root": str(root.resolve()),
        "repo_id": dataset.repo_id,
        "codebase_version": info["codebase_version"],
        "episodes": dataset.num_episodes,
        "frames": dataset.num_frames,
        "fps": dataset.fps,
        "features": dataset.features,
        "decoded_frame_shapes": {
            key: list(value.shape)
            for key, value in sample.items()
            if key.startswith("observation.images.")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local LeRobot v3 dataset")
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_dataset(args.dataset_root), indent=2, default=str))


if __name__ == "__main__":
    main()
