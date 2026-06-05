#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

SETTINGS="${SETTINGS:-cifar100_dir03_20c cifar100_path20_20c cifar100_dir03_10c cifar100_path20_10c cifar10_dir03_100c cifar10_dir03_50c cifar10_path2_100c cifar10_path2_50c emnist_dir03_1000c emnist_path20_1000c}"
GRANULARITIES="${GRANULARITIES:-block layer_tied model_tied}"
for setting in $SETTINGS; do
  prepare_partition "$setting"
  for gran in $GRANULARITIES; do
    run_config "icdm/routing_granularity/${setting}_${gran}" "$@"
  done
done
