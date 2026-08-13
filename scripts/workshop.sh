#!/usr/bin/env sh
set -eu

command_name="${1:-start}"
dataset_name="${2:-}"
numeric_argument="${3:-}"
seed="${4:-0}"
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
      workshop "/workspace/data/datasets/$dataset_name" --device "$device" \
      --steps "${numeric_argument:-100000}"
    ;;
  deploy)
    test -n "$dataset_name"
    docker compose $compose_files up -d --build --wait workshop
    docker compose $compose_files exec -T workshop \
      rospin-deploy "$dataset_name" --episodes "${numeric_argument:-1}" \
      --seed "$seed" --device "$device"
    ;;
  preview)
    test -n "$dataset_name"
    docker compose $compose_files exec -T workshop \
      rospin-generate "$dataset_name" --preview --seed "$seed"
    ;;
  generate)
    test -n "$dataset_name"
    docker compose $compose_files exec -T workshop \
      rospin-generate "$dataset_name" --episodes "${numeric_argument:-10}" \
      --seed "$seed"
    ;;
  *)
    echo "Usage: $0 {start|stop|check|train|deploy|preview|generate} [name] [count] [seed]" >&2
    exit 2
    ;;
esac
