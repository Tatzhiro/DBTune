# DBTune - Database Configuration Tuning System

## Overview
DBTune automatically finds optimal configuration knobs for database systems (MySQL, PostgreSQL).
It uses black-box optimization: propose knob configs -> run benchmark -> observe performance -> repeat.

## Entry Point
```
cd scripts && python optimize.py --config=config_performance.ini
```
Flow: `optimize.py` -> `DBTuner.tune()` -> `PipleLine.run()` -> iteratively calls `iterate()` which gets a config suggestion from the optimizer and evaluates it via `DBEnv.step()`.

## Architecture

### Core Pipeline (`autotune/pipleline/pipleline.py`)
`PipleLine(BOBase)` is the main tuning loop. Each iteration:
1. **Knob selection** - optionally prunes the configuration space (increase/decrease/none)
2. **Get suggestion** - asks the optimizer for the next config to try
3. **Evaluate** - applies config to DB, runs benchmark, records result in `HistoryContainer`
4. Supports optional **space transfer** (prune search space using source task data) and **auto optimizer selection** (switch optimizers dynamically)

### Optimizers (`autotune/optimizer/`)
All optimizers implement `get_suggestion(history_container, compact_space)` and are registered in `PipleLine.__init__`:

| Type | Class | File | Description |
|------|-------|------|-------------|
| `MBO`/`SMAC` | `BO_Optimizer` | `bo_optimizer.py` | Bayesian Optimization with surrogate model + acquisition function |
| `GA` | `GA_Optimizer` | `ga_optimizer.py` | Evolutionary/Genetic Algorithm with tournament selection + mutation |
| `DDPG` | `DDPG_Optimizer` | `ddpg_optimizer.py` | Deep RL (actor-critic), uses internal metrics as state |
| `TurBO` | `TURBO_Optimizer` | `turbo_optimizer.py` | Trust Region BO |
| `TPE` | `TPE_Optimizer` | `tpe_optimizer.py` | Tree-structured Parzen Estimator |

### Surrogate Models (`autotune/optimizer/surrogate/`)
Built via `build_surrogate()` in `surrogate/core.py`:
- `gp` / `gp_rbf` / `gp_mcmc` - Gaussian Process variants (`surrogate/base/`)
- `prf` - Probabilistic Random Forest (pyrfr or sklearn fallback)
- `lightgbm` - LightGBM surrogate
- `context_prf` - RF with context features (internal metrics)
- `tlbo_*` - Transfer learning surrogates (RGPE, workload mapping, etc.)

### Acquisition Functions (`autotune/optimizer/acquisition_function/`)
Built via `build_acq_func()` in `optimizer/core.py`:
- Single-obj: `ei`, `pi`, `lcb`, `logei`, `eips`, `lpei`
- With constraints: `eic`
- Multi-obj: `ehvi`, `mesmo`, `usemo`, `parego`
- Monte Carlo variants: `mcei`, `mcehvi`, `mcparego`, etc.

### Acquisition Maximizers (`autotune/optimizer/acq_maximizer/`)
- `local_random` (InterleavedLocalAndRandomSearch) - default for mixed spaces
- `random_scipy` (RandomScipyOptimizer) - for continuous-only spaces
- Others: `scipy_global`, `cma_es`, `batchmc`, etc.

### Knob Selection (`autotune/selector/selector.py`)
`KnobSelector` ranks knobs by importance to reduce search space:
- `shap` - SHAP values via LightGBM (default)
- `fanova` - Functional ANOVA
- `gini` - Random Forest feature importance
- `ablation` - Ablation path analysis
- `lasso` - LASSO regularization path

### Transfer Learning (`autotune/transfer/tlbo/`)
Leverages historical tuning data from other workloads:
- `RGPE` - Ranking-weighted GP Ensemble
- `WorkloadMapping` - Maps similar source workloads (supports two methods, see below)
- `MFGPE` - Multi-fidelity GP Ensemble
- `SGPR` - Stacking GPR
- `TOPO_V3` - Topology-based transfer

#### WorkloadMapping Methods (`autotune/transfer/tlbo/workload_map.py`)
The `WorkloadMapping` class supports two configurable mapping methods via `mapping_method`:

**OtterTune (default, `mapping_method = ottertune`)**
Implements the original OtterTune workload mapping algorithm (Van Aken et al., SIGMOD 2017, Section 6.1).
Directly compares observed internal metrics at configurations that exactly match between target and source workloads.

Algorithm:
1. For each source workload, find configs that exactly match any target config (normalized config array, atol=1e-6)
2. At matched configs, compare internal metrics using decile binning (values binned into 1-10 via `np.percentile` deciles)
3. Compute Euclidean distance between binned metric vectors for each matched pair
4. Score = average distance across all matched config pairs
5. Select source with lowest score, concatenate its data with target data to build the surrogate

The first iteration always has at least one match because all tuning sessions start from the default configuration. As tuning progresses, more config matches may accumulate and the score is recalculated using all accumulated matches.

Optional: `mapping_prune_metrics = true` enables FA+KMeans metric pruning (OtterTune Section 4.2) — reduces internal metrics to a representative subset via Factor Analysis dimensionality reduction followed by K-Means clustering (K selected by Pham-Dimov-Nguyen 2005 heuristic).

**DBTune (`mapping_method = dbtune`)**
The original DBTune GP-based approach. For each source workload, trains a Gaussian Process per metric dimension to predict what the target's internal metrics would look like, then compares GP predictions with actual target metrics. More expensive but handles non-overlapping configs.

Config example:
```ini
[tune]
transfer_framework = workload_map
mapping_method = ottertune          # ottertune (default) | dbtune
mapping_prune_metrics = false       # true | false (only for ottertune)
data_repo = ./DBTune_history/csv_source_no_tpcc
```

Surrogate type encoding (internal): `mapping_method` is encoded into the `surrogate_type` string that flows through the pipeline:
- `tlbo_ottertune_prf` / `tlbo_ottertune_gp` — OtterTune without pruning
- `tlbo_ottertune_pruned_prf` / `tlbo_ottertune_pruned_gp` — OtterTune with FA+KMeans pruning
- `tlbo_mapping_prf` / `tlbo_mapping_gp` — DBTune GP-based (original)

### Database Environment (`autotune/dbenv.py`)
`DBEnv.step(config)` is the objective function:
1. Converts config to knob values
2. Applies knobs to DB (online or offline with restart)
3. Runs benchmark (sysbench/oltpbench/JOB/TPC-H)
4. Collects external metrics (tps, lat, qps) + internal metrics (65 DB stats) + resource usage (cpu, IO, mem)
5. Returns `(objs, constraints, external_metrics, resource, internal_metrics, info, trial_state)`

### Database Backends (`autotune/database/`)
- `MysqlDB` (`mysqldb.py`) - MySQL connector, knob application, restart, internal metrics collection
- `PostgresqlDB` (`postgresqldb.py`) - PostgreSQL equivalent

### Configuration Space (`autotune/utils/config_space/`)
Uses ConfigSpace library. Knob definitions loaded from JSON files (e.g., `scripts/experiment/gen_knobs/OLTP_8.0.json`).
Types: `UniformIntegerHyperparameter`, `CategoricalHyperparameter`, `UniformFloatHyperparameter`.

### History Container (`autotune/utils/history_container.py`)
`HistoryContainer` stores all observations (config, objs, constraints, metrics, etc.).
Supports save/load JSON, tracks incumbents, and provides transformed performance values.
`MOHistoryContainer` extends it for multi-objective.

## Config File Format (`scripts/config_performance.ini`)
Two sections:
- `[database]` - DB connection, knob config file, workload settings
- `[tune]` - Task ID, performance metric, optimizer method, knob selection, transfer settings

Key settings:
- `optimize_method` = MBO | SMAC | TPE | DDPG | TurBO | GA
- `transfer_framework` = none | workload_map | rgpe | finetune | context
- `mapping_method` = ottertune (default) | dbtune — only when `transfer_framework = workload_map`
- `mapping_prune_metrics` = false (default) | true — FA+KMeans metric pruning, only for ottertune
- `selector_type` = shap | fanova | gini | ablation | lasso
- `incremental` = none | increase | decrease

## Adding a New Optimizer
To add a new optimization method:
1. Create `autotune/optimizer/your_optimizer.py` with a class that implements:
   - `__init__(self, config_space, history_container, ...)`
   - `get_suggestion(self, history_container, compact_space=None)` -> returns a `Configuration`
   - `update(self, observation)` (optional, for stateful optimizers like GA/DDPG)
2. Register it in `PipleLine.__init__()` (`autotune/pipleline/pipleline.py`) with an `elif optimizer_type == 'YOUR_TYPE':` block
3. Add it to the config file's `optimize_method` option

## Key Types
- `Configuration` - a point in the config space (dict-like, from ConfigSpace)
- `Observation` - named tuple with config, objs, constraints, trial_state, elapsed_time, EM, resource, IM, info, context
- `HistoryContainer` - stores list of Observations, tracks incumbents

## Notes
- Performance is minimized internally (tps is negated: `-tps`)
- The `auto_optimizer` mode dynamically switches between SMAC/MBO/DDPG/GA using an XGBoost ranker
- Knob values exceeding `sys.maxsize` are scaled by 1000 (e.g., `innodb_buffer_pool_size`)
- The `pipleline` directory name is a typo (should be "pipeline") - preserved for compatibility

## DML Model Training Data
The DML (Deep Metric Learning) optimizer uses a triplet embedding model trained on the `DBMSTransferLearning` dataset. Training inputs:
- **Triplet data**: `DBMSTransferLearning/dataset/full_triplet_data_concordance.csv` — generates (anchor, positive, negative) triplets from historical tuning runs on different hardware/workloads
- **Context metrics**: `DBMSTransferLearning/dataset/context_default_metrics_all.csv` — 11 DBMS/OS metrics (memory usage, InnoDB hit rate, dirty pages, QPS, CPU, row ops, disk IOPS) collected at the **default configuration** for each `{hardware}_{workload}` context
- **Training script**: `scripts/train_dml_model.py`
- **Input dim**: 11 features → 64 → 32 → 16-dim embedding (L2 normalized)
- **Outputs**: `context_model.pth`, `scaler.pkl` (MinMaxScaler), and a copy of the context metrics CSV

To train a model that excludes a specific workload (for "unseen workload" evaluation), filter both CSVs before running `train_dml_model.py` and point `--output_dir` to `autotune/optimizer/dml_models_no_{workload}/`. See `.claude/skills/run-experiment.md` for details.

## Known Performance Issues
- **MySQL slow restart after force kill is likely a disk I/O bottleneck.** In controlled tests, graceful shutdown + restart takes ~26-30s for any config (including 5GB redo logs). But in real tuning runs, InnoDB crash recovery after `kill -9` can take **60-120s** for the same data and configs. Test data: standalone `scripts/test_restart_slowdown.sh` shows ~30s; real runs consistently show 60-120s. The server hosts Elasticsearch (128GB heap) and other processes competing for disk — this is the suspected cause, though not definitively proven. Increasing `TIMEOUT_CLOSE` in `mysqldb.py` to allow graceful shutdown (rather than force kill) avoids crash recovery and is faster.
- Tuning iterations with sysbench RW can take ~10 minutes. Changing `innodb_log_file_size` (e.g. 5GB -> 48MB) was hypothesized as the cause but **disproven**: a standalone test (`scripts/test_restart_slowdown.sh`) showed the 5GB->48MB restart adds only ~33s (30s shutdown + 3s startup), far less than the observed ~10min delays. The real bottleneck is likely disk I/O contention as described above.
