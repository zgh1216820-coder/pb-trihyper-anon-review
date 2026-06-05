#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

SETTING="${SETTING:-cifar100_dir03_20c}"
prepare_partition "$SETTING"
run_config "${CONFIG:-icdm/diagnostics/routing_export_cifar100_dir03_20c}" "$@"

echo "Routing diagnostics are saved under model_saved/new_model/pb_trihyper/*/routing_evidence/ when the final client evaluation runs."
