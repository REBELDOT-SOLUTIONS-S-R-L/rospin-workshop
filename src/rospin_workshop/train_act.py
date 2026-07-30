from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _dataset_repo_id(root: Path) -> str:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"{root} is not a finalized LeRobot dataset (missing meta/info.json)"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("ACT training requires a LeRobot v3.0 dataset")
    # LeRobot v3 does not persist repo_id in info.json. Recording uses the
    # dataset directory name as its stable local-only namespace.
    return f"local/{root.name}"


def build_train_command(args: argparse.Namespace) -> list[str]:
    dataset_root = args.dataset_root.resolve()
    repo_id = _dataset_repo_id(dataset_root)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (
            dataset_root.parent.parent / "outputs" / f"act_{dataset_root.name}"
        ).resolve()
    )
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        "--dataset.video_backend=pyav",
        "--policy.type=act",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        f"--output_dir={output_dir}",
        f"--job_name=act_{dataset_root.name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
    ]
    if args.no_pretrained_backbone:
        command.append("--policy.pretrained_backbone_weights=null")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a local ACT policy from a workshop LeRobot v3 dataset"
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Avoid downloading pretrained ResNet weights",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved LeRobot command without running it",
    )
    args = parser.parse_args()
    command = build_train_command(args)
    if args.print_command:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
