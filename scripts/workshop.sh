#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(dirname "$script_dir")
cd "$repo_root"
mkdir -p "$repo_root/data"
ROSPIN_HOST_DATA_DIR=${ROSPIN_HOST_DATA_DIR:-"$repo_root/data"}
export ROSPIN_HOST_DATA_DIR

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

verify_data_mount() {
  probe_name=".rospin-mount-probe-$$"
  probe_value="rospin-mount-ok-$$"
  probe_path="$repo_root/data/$probe_name"

  docker compose $compose_files exec -T workshop sh -c \
    "printf '%s' '$probe_value' > '/workspace/data/$probe_name'"
  probe_attempt=0
  while [ "$probe_attempt" -lt 50 ]; do
    if [ -f "$probe_path" ] && [ "$(cat "$probe_path")" = "$probe_value" ]; then
      break
    fi
    probe_attempt=$((probe_attempt + 1))
    sleep 0.1
  done
  if [ ! -f "$probe_path" ] || [ "$(cat "$probe_path")" != "$probe_value" ]; then
    docker compose $compose_files exec -T workshop \
      rm -f "/workspace/data/$probe_name" >/dev/null 2>&1 || true
    echo "Docker cannot write through to the host data directory: $repo_root/data" >&2
    echo "Refusing to run because datasets would remain inside Docker." >&2
    exit 1
  fi
  rm -f "$probe_path"
}

case "$command_name" in
  start)
    docker compose $compose_files up -d --build --force-recreate --wait workshop
    verify_data_mount
    echo "SO-101 workshop is running at http://localhost:8000"
    docker compose $compose_files logs -f workshop
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
    verify_data_mount
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
