# PB-TriHyper Anonymous Review Release

This repository contains an anonymous implementation of PB-TriHyper for generated personalized federated learning with parameter-block routing.

## Scope

This release provides:

- PB-TriHyper implementation;
- block-wise, layer-tied, and model-tied routing variants;
- benchmark configuration files;
- partition generation scripts;
- scripts for routing diagnostics and table generation.

Raw datasets and large checkpoints are not included.

## Installation

```bash
conda create -n pbtrihyper python=3.10 -y
conda activate pbtrihyper
pip install -r requirements.txt
```

The default `requirements.txt` includes the CUDA 12.1 PyTorch wheel index. Adjust the PyTorch install line if your review environment uses CPU-only PyTorch or a different CUDA version.

## Dataset Preparation

Datasets are generated with the FL-bench partitioning entry point. The generated partition metadata is saved under `data/<dataset>/` as `args.json`, `partition.pkl`, and related files.

Examples:

```bash
# CIFAR-100 Dirichlet-0.3 / 20 clients
python generate_data.py -d cifar100 -cn 20 -sp sample -tr 0.25 -vr 0 -a 0.3 -c 0 -ms 100 -pd 0 --seed 42 --use_cuda 0

# CIFAR-10 Path-2 / 50 clients
python generate_data.py -d cifar10 -cn 50 -sp sample -tr 0.25 -vr 0 -a 0 -c 2 -ms 10 -pd 0 --seed 42 --use_cuda 0
```

The run scripts can prepare partitions automatically by setting `PREPARE_DATA=1`.

## Smoke Test

This command prepares a small CIFAR-10 partition and runs a two-round CPU smoke test:

```bash
python generate_data.py -d cifar10 -cn 5 -sp sample -tr 0.25 -vr 0 -a 0.3 -c 0 -ms 5 -pd 0 --seed 42 --use_cuda 0
python main.py --config-path=config/icdm/diagnostics --config-name=smoke_test
```

## Main Benchmark Grid

```bash
# Run all ten block-routing settings, preparing each partition first.
PREPARE_DATA=1 GPU=0 bash scripts/run_main_grid.sh

# Run a subset.
PREPARE_DATA=1 SETTINGS="cifar100_dir03_20c cifar10_path2_50c" GPU=0 bash scripts/run_main_grid.sh
```

## Routing-Granularity Grid

```bash
PREPARE_DATA=1 GPU=0 bash scripts/run_granularity_grid.sh

# Optional subset.
PREPARE_DATA=1 SETTINGS="cifar100_dir03_20c" GRANULARITIES="block layer_tied model_tied" bash scripts/run_granularity_grid.sh
```

## Block-Size Sensitivity

The legacy implementation uses `effective block size q = pb_trihyper.chunk_size / 2`. The block-size sensitivity configs set `pb_trihyper.effective_block_size_q` directly for `q in {128, 256, 512}` while leaving the legacy default path unchanged for other configs.

```bash
PREPARE_DATA=1 GPU=0 bash scripts/run_block_size_sensitivity.sh
```

## Routing Diagnostics

```bash
PREPARE_DATA=1 GPU=0 bash scripts/export_routing_diagnostics.sh
```

Expected routing diagnostics appear under:

```text
model_saved/new_model/pb_trihyper/<setting>_<granularity>/routing_evidence/
```

The exported files include `routing_weights.npz`, `client_metadata.csv`, and `block_metadata.csv` when the final client evaluation runs.

## Counterfactual Routing Interventions

First train/export a block-routing checkpoint, then run:

```bash
CKPT_DIR=model_saved/new_model/pb_trihyper/20_0.3_cifar100_block PREPARE_DATA=1 GPU=0 bash scripts/run_counterfactual_routing.sh
```

The output CSV is written to the Hydra run directory as `counterfactual_routing_results.csv`.

## Paper Artifact Mapping

| Paper item | Script/config |
| --- | --- |
| Table I main benchmark | `scripts/run_main_grid.sh`, `config/icdm/main_grid/*.yaml` |
| Table II routing granularity | `scripts/run_granularity_grid.sh`, `config/icdm/routing_granularity/*.yaml` |
| Block-size table | `scripts/run_block_size_sensitivity.sh`, `config/icdm/block_size/*.yaml` |
| Routing diagnostics figure/table | `scripts/export_routing_diagnostics.sh`, `model_saved/.../routing_evidence/` |
| Counterfactual routing table | `scripts/run_counterfactual_routing.sh` |
| Component ablations | `config/icdm/diagnostics/component_ablation_*.yaml` |

Small expected CSVs are included in `results/expected/`. Convert them to Markdown with:

```bash
python scripts/make_tables.py --expected-dir results/expected --output-dir outputs/tables
```

See `docs/table_mapping.md` for a longer mapping.

## Output Locations

- Hydra logs: `out/pb_trihyper/<dataset>/<timestamp>/`
- Generated checkpoints and routing evidence: `model_saved/new_model/pb_trihyper/`
- Expected CSV summaries: `results/expected/`
- Derived Markdown tables: `outputs/tables/`

Generated outputs are ignored by git.

## Anonymity

This release is prepared for anonymous review. It contains no raw datasets, generated partitions, checkpoints, logs, local notebooks, author names, emails, institution names, or personal absolute paths. See `docs/anonymity_checklist.md` for audit commands.
