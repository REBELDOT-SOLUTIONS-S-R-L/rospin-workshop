from __future__ import annotations

import argparse
import json
import sys

from rospin_workshop.train_act import build_train_command


def test_local_act_command_never_pushes(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text(
        json.dumps({"codebase_version": "v3.0"}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        dataset_root=dataset,
        output_dir=tmp_path / "output",
        device="cpu",
        steps=12,
        batch_size=2,
        num_workers=0,
        no_pretrained_backbone=True,
    )
    command = build_train_command(args)
    assert command[:3] == [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
    ]
    assert "--policy.type=act" in command
    assert "--policy.push_to_hub=false" in command
    assert "--wandb.enable=false" in command
    assert "--dataset.repo_id=local/dataset" in command
    assert "--dataset.video_backend=pyav" in command
    assert "--policy.pretrained_backbone_weights=null" in command
