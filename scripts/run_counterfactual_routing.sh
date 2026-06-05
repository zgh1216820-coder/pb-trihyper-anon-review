#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_common.sh

if [[ -z "${CKPT_DIR:-}" ]]; then
  echo "CKPT_DIR is required. Example: CKPT_DIR=model_saved/new_model/pb_trihyper/20_0.3_cifar100_block" >&2
  exit 2
fi
SETTING="${SETTING:-cifar100_dir03_20c}"
prepare_partition "$SETTING"
run_config "${CONFIG:-icdm/diagnostics/counterfactual_routing_cifar100_dir03_20c}"   pb_trihyper.eval_checkpoint_dir="$CKPT_DIR" "$@"
