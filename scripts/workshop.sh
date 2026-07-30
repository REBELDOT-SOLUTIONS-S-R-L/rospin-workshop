#!/usr/bin/env sh
set -eu

command_name="${1:-start}"
dataset_name="${2:-}"
compose_files="-f compose.yaml"
device="cpu"

if [ "${ROSPIN_GPU:-0}" = "1" ]; then
  compose_files="$compose_files -f compose.gpu.yaml"
  device="cuda"
fi

case "$command_name" in
  start)
    docker compose $compose_files up --build
    ;;
  stop)
    docker compose $compose_files down
    ;;
  check)
    test -n "$dataset_name"
    docker compose $compose_files run --rm --entrypoint rospin-check-dataset \
      workshop "/workspace/data/datasets/$dataset_name"
    ;;
  train)
    test -n "$dataset_name"
    docker compose $compose_files run --rm --entrypoint rospin-train-act \
      workshop "/workspace/data/datasets/$dataset_name" --device "$device"
    ;;
  *)
    echo "Usage: $0 {start|stop|check|train} [dataset-directory]" >&2
    exit 2
    ;;
esac

