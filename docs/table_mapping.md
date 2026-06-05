# Paper Table/Figure Mapping

| Paper item | Purpose | Configs | Script | Expected outputs |
| --- | --- | --- | --- | --- |
| Table I | Ten-setting PB-TriHyper main benchmark | `config/icdm/main_grid/*.yaml` | `scripts/run_main_grid.sh` | `out/pb_trihyper/<dataset>/<timestamp>/metrics.csv`, `results/expected/main_grid_seed42.csv` |
| Table II | Block/layer/model routing-granularity comparison | `config/icdm/routing_granularity/*_{block,layer_tied,model_tied}.yaml` | `scripts/run_granularity_grid.sh` | `out/pb_trihyper/<dataset>/<timestamp>/metrics.csv` |
| Block-size table | Effective block size q sensitivity on CIFAR-100 Dirichlet-0.3/20 clients | `config/icdm/block_size/cifar100_dir03_20c_q*.yaml` | `scripts/run_block_size_sensitivity.sh` | `out/pb_trihyper/cifar100/<timestamp>/metrics.csv` |
| Routing diagnostics | Export routing weights, client metadata, and block metadata | `config/icdm/diagnostics/routing_export_cifar100_dir03_20c.yaml` | `scripts/export_routing_diagnostics.sh` | `model_saved/new_model/pb_trihyper/*/routing_evidence/` |
| Counterfactual routing | Post-hoc routing interventions from a saved checkpoint | `config/icdm/diagnostics/counterfactual_routing_cifar100_dir03_20c.yaml` | `scripts/run_counterfactual_routing.sh` | `out/pb_trihyper/<dataset>/<timestamp>/counterfactual_routing_results.csv` |
| Component ablations | Disable one prior branch while keeping the same training scaffold | `config/icdm/diagnostics/component_ablation_cifar100_dir03_20c_*.yaml` | `python main.py --config-name <config>` | `out/pb_trihyper/cifar100/<timestamp>/metrics.csv` |
