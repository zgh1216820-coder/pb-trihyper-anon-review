#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

prepare_partition cifar100_dir03_20c
QS="${QS:-128 256 512}"
for q in $QS; do
  run_config "icdm/block_size/cifar100_dir03_20c_q${q}" "$@"
done
