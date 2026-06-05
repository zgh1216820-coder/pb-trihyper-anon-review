#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-42}"
TEST_INTERVAL="${TEST_INTERVAL:-1000}"

if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

prepare_partition() {
  local setting="$1"
  if [[ "${PREPARE_DATA:-0}" != "1" ]]; then
    return 0
  fi
  case "$setting" in
    cifar100_dir03_20c)
      python generate_data.py -d cifar100 -cn 20 -sp sample -tr 0.25 -vr 0 -a 0.3 -c 0 -ms 100 -pd 0 --seed "$SEED" --use_cuda 0
      ;;
    cifar100_path20_20c)
      python generate_data.py -d cifar100 -cn 20 -sp sample -tr 0.25 -vr 0 -a 0 -c 20 -ms 100 -pd 0 --seed "$SEED" --use_cuda 0
      ;;
    cifar100_dir03_10c)
      python generate_data.py -d cifar100 -cn 10 -sp sample -tr 0.25 -vr 0 -a 0.3 -c 0 -ms 100 -pd 0 --seed "$SEED" --use_cuda 0
      ;;
    cifar100_path20_10c)
      python generate_data.py -d cifar100 -cn 10 -sp sample -tr 0.25 -vr 0 -a 0 -c 20 -ms 100 -pd 0 --seed "$SEED" --use_cuda 0
      ;;
    cifar10_dir03_100c)
      python generate_data.py -d cifar10 -cn 100 -sp sample -tr 0.25 -vr 0 -a 0.3 -c 0 -ms 10 -pd 0 --seed "$SEED" --use_cuda 0
      ;;
    cifar10_dir03_50c)
      python generate_data.py -d cifar10 -cn 50 -sp sample -tr 0.25 -vr 0 -a 0.3 -c 0 -ms 10 -pd 0 --seed "$SEED" --use_cuda 0
      ;;
    cifar10_path2_100c)
      python generate_data.py -d cifar10 -cn 100 -sp sample -tr 0.25 -vr 0 -a 0 -c 2 -ms 10 -pd 0 --seed "$SEED" --use_cuda 0
      ;;
    cifar10_path2_50c)
      python generate_data.py -d cifar10 -cn 50 -sp sample -tr 0.25 -vr 0 -a 0 -c 2 -ms 10 -pd 0 --seed "$SEED" --use_cuda 0
      ;;
    emnist_dir03_1000c)
      python generate_data.py -d emnist -cn 1000 -sp sample -tr 0.25 -vr 0 -a 0.3 -c 0 -ms 100 -pd 0 --emnist_split byclass --seed "$SEED" --use_cuda 0
      ;;
    emnist_path20_1000c)
      python generate_data.py -d emnist -cn 1000 -sp sample -tr 0.25 -vr 0 -a 0 -c 20 -ms 100 -pd 0 --emnist_split byclass --seed "$SEED" --use_cuda 0
      ;;
    *) echo "Unknown setting for data preparation: $setting" >&2; exit 2 ;;
  esac
}

main_config_for() {
  local setting="$1"
  case "$setting" in
    cifar100_dir03_20c) echo "icdm/main_grid/cifar100_dir03_20c" ;;
    cifar100_path20_20c) echo "icdm/main_grid/cifar100_path20_20c" ;;
    cifar100_dir03_10c) echo "icdm/main_grid/cifar100_dir03_10c" ;;
    cifar100_path20_10c) echo "icdm/main_grid/cifar100_path20_10c" ;;
    cifar10_dir03_100c) echo "icdm/main_grid/cifar10_dir03_100c" ;;
    cifar10_dir03_50c) echo "icdm/main_grid/cifar10_dir03_50c" ;;
    cifar10_path2_100c) echo "icdm/main_grid/cifar10_path2_100c" ;;
    cifar10_path2_50c) echo "icdm/main_grid/cifar10_path2_50c" ;;
    emnist_dir03_1000c) echo "icdm/main_grid/emnist_dir03_1000c" ;;
    emnist_path20_1000c) echo "icdm/main_grid/emnist_path20_1000c" ;;
    *) echo "Unknown setting: $setting" >&2; exit 2 ;;
  esac
}

run_config() {
  local config_name="$1"
  shift || true
  echo "===== Running $config_name at $(date) ====="
  local config_args=()
  if [[ "$config_name" == */* ]]; then
    config_args=(--config-path="config/${config_name%/*}" --config-name="${config_name##*/}")
  else
    config_args=(--config-name="$config_name")
  fi
  python main.py "${config_args[@]}" \
    common.seed="$SEED" \
    common.test.client.interval="$TEST_INTERVAL" \
    common.monitor=null "$@"
}
