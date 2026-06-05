# Final Release Report

## Included implementation

- `src/client/pb_trihyper.py`
- `src/server/pb_trihyper.py`
- `src/client/trimoe_nofish_nofuzzy.py`
- `src/server/trimoe_nofish_nofuzzy.py`
- FL-bench base classes required by the method: `src/client/fedavg.py`, `src/server/fedavg.py`

The `pb_trihyper` files expose the anonymous method name through the FL-bench dynamic loader. The implementation backend is retained to avoid changing numerical behavior.

## Included configs

- Table I main benchmark grid: `config/icdm/main_grid/*.yaml` (10 configs)
- Table II routing-granularity grid: `config/icdm/routing_granularity/*.yaml` (30 configs)
- Block-size sensitivity: `config/icdm/block_size/*.yaml` (q = 128, 256, 512)
- Diagnostics: `config/icdm/diagnostics/*.yaml`
- Smoke test: `config/icdm/diagnostics/smoke_test.yaml`

## Included scripts

- `scripts/run_main_grid.sh`
- `scripts/run_granularity_grid.sh`
- `scripts/run_block_size_sensitivity.sh`
- `scripts/export_routing_diagnostics.sh`
- `scripts/run_counterfactual_routing.sh`
- `scripts/make_tables.py`
- `scripts/audit_budget.py`

## Validation performed

- Python compile check for the main entry point, data generation scripts, PB-TriHyper client/server files, FL-bench base client/server files, and utility scripts.
- `scripts/make_tables.py` successfully parsed the expected CSV files in `results/expected/`.
- A one-round CPU smoke test completed on a temporary CIFAR-10 5-client partition. The run logged `pb_trihyper Max Accuracy` at epoch 1.
- `scripts/audit_budget.py --root .` passed after removing temporary smoke-test data and outputs.
- Identity grep checks found no author placeholders, institution placeholders, home paths, Windows user paths, emails, or private server strings.

## Known limitations

- Raw datasets and large checkpoints are intentionally excluded.
- The smoke test downloads CIFAR-10 if it is not already cached by torchvision.
- Full benchmark reproduction requires the reviewer to run the provided scripts on a CUDA environment.
- On Windows PowerShell, `conda run` may fail while printing rich terminal output because of console encoding. The underlying smoke-test log still completed successfully; Linux shell execution is recommended for reproduction.
- Some internal compatibility class names are preserved in the implementation backend to avoid changing algorithm behavior. User-facing configs, scripts, and documentation use `PB-TriHyper` and `pb_trihyper`.
