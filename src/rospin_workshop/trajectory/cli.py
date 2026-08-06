from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _request(
    server: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
) -> Any:
    data = None
    headers: dict[str, str] = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(
        f"{server.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.load(exc).get("detail", str(exc))
        except Exception:  # noqa: BLE001 - best-effort API error parsing
            detail = str(exc)
        raise RuntimeError(str(detail)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview a Python trajectory or generate synthetic episodes"
    )
    parser.add_argument("program", help="Python filename under trajectories/")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-name", default="synthetic_trajectory")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Execute one visible episode without recording",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Record without first validating the same seeded episode",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("ROSPIN_SERVER_URL", "http://127.0.0.1:8000"),
    )
    args = parser.parse_args()

    try:
        status = _request(
            args.server,
            "/api/trajectory/start",
            payload={
                "program": args.program,
                "episodes": args.episodes,
                "seed": args.seed,
                "preview": args.preview,
                "dataset_name": args.dataset_name,
                "preflight": not args.no_preflight,
            },
        )
        previous_summary: tuple[Any, ...] | None = None
        while status["running"]:
            summary = (
                status.get("current_seed"),
                status.get("phase"),
                status.get("completed_episodes"),
                status.get("saved_episodes"),
                status.get("discarded_episodes"),
            )
            if summary != previous_summary:
                print(
                    f"seed={summary[0]} phase={summary[1]} "
                    f"completed={summary[2]} saved={summary[3]} "
                    f"discarded={summary[4]}",
                    flush=True,
                )
                previous_summary = summary
            time.sleep(0.4)
            status = _request(args.server, "/api/trajectory/status")
    except KeyboardInterrupt:
        _request(args.server, "/api/trajectory/stop", payload={})
        raise SystemExit(130) from None
    except (OSError, RuntimeError) as exc:
        print(f"trajectory error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(status, indent=2))
    if status.get("error"):
        raise SystemExit(1)
    if not args.preview and status.get("saved_episodes", 0) == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
