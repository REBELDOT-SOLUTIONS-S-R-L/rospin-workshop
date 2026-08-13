from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _request(base_url: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=130) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        raise RuntimeError(str(body.get("detail") or exc.reason)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy a trained ACT policy in the live workshop simulator"
    )
    parser.add_argument(
        "checkpoint",
        help="Checkpoint path relative to data/outputs, ending in pretrained_model",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    status = _request(
        args.url,
        "/api/policy/start",
        {
            "checkpoint": args.checkpoint,
            "episodes": args.episodes,
            "seed": args.seed,
            "device": args.device,
        },
    )
    last_progress: tuple[Any, ...] | None = None
    try:
        while status.get("running"):
            progress = (
                status.get("phase"),
                status.get("current_episode"),
                status.get("completed_episodes"),
                status.get("successful_episodes"),
                status.get("timed_out_episodes"),
            )
            if progress != last_progress:
                print(
                    " ".join(
                        (
                            f"phase={progress[0]}",
                            f"episode={progress[1]}",
                            f"completed={progress[2]}",
                            f"success={progress[3]}",
                            f"timeout={progress[4]}",
                        )
                    ),
                    flush=True,
                )
                last_progress = progress
            time.sleep(0.25)
            status = _request(args.url, "/api/policy/status")
    except KeyboardInterrupt:
        _request(args.url, "/api/policy/stop", {})
        print("Policy deployment stop requested", file=sys.stderr)
        raise SystemExit(130) from None

    print(json.dumps(status, indent=2))
    if status.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
