# pa_moap_rl

`pa_moap_rl` implements preference-aware multi-objective teaching-method
assignment. It consumes the knowledge points selected by an upstream
resource-combination stage and assigns one of 11 teaching methods to each
selected point.

The assignment objective combines teaching effect, student preference,
teacher preference, method diversity, and method-share penalties. The
repository includes random, greedy, local-search, exact Gurobi, hybrid, and
masked PPO solvers, together with versioned experiment and audit workflows.

## Current research baseline

The current four-topology development baseline was frozen on 2026-09-21:

- objective: `main_no_reference__h0.65__c0.35__be1.00__bc1.00`;
- effect weight `0.35`, student/teacher weights `0.325` each;
- PPO model: `legacy_separate_v1`;
- seed-0 reference: stabilization run03, checkpoint update 60;
- deterministic inference: `max_steps=512`, `patience=32`.

The baseline is an engineering and experiment-protocol freeze, not a claim
that all preference-response requirements have been met. In particular,
teacher-only preference changes remain a known zero-response limitation.

See [the current project snapshot](docs/project_status_20260922.md),
[the task ledger](docs/mission_list.md), and
[the frozen inference configuration](pa_moap_rl/configs/ppo_inference_frozen_four_topology_v2.yaml)
for the authoritative status and scope.

## Installation and validation

Python 3.10 or newer is required. On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -m compileall pa_moap_rl
.\.venv\Scripts\python.exe -m pytest -q
```

Exact-solver experiments additionally require a licensed Gurobi installation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-exact.txt
```

The repository baseline currently contains 115 passing tests.

## Data preparation and smoke workflow

Raw input data and generated instances are intentionally not committed. With
the local inputs available, the formal pipeline starts with:

```powershell
.\.venv\Scripts\python.exe -m pa_moap_rl.data.convert_pre_data `
  --instance-root pre_data/instance `
  --selection-root pre_data/x `
  --output-root data_processed/base_instances

.\.venv\Scripts\python.exe -m pa_moap_rl.data.build_dataset `
  --manifest data_processed/base_instances/manifest.csv `
  --output-dir data_processed/splits

.\.venv\Scripts\python.exe -m pa_moap_rl.experiments.formal_training `
  --smoke-test --device auto
```

Read [the formal training guide](docs/formal_training_guide.md) before running
pilot or long training jobs. Monitoring fields and checkpoint artifacts are
documented in [training_monitoring_data.md](docs/training_monitoring_data.md).

## Repository map

- `pa_moap_rl/configs/`: objective, profile, training, inference, and experiment
  specifications.
- `pa_moap_rl/data/`: parsing, instance conversion, split construction, and
  scenario generation.
- `pa_moap_rl/envs/`: the masked single-replacement assignment environment.
- `pa_moap_rl/models/`: actor-critic encoders and versioned preference-repair
  models.
- `pa_moap_rl/solvers/`: heuristic, exact, hybrid, and PPO solvers.
- `pa_moap_rl/experiments/`: training, evaluation, diagnostics, ablations,
  service benchmarks, and report preparation.
- `tests/`: unit and protocol-regression tests.
- `docs/`: mathematical definitions, protocols, reports, and the research task
  ledger.

See [repo_structure.md](docs/repo_structure.md) for more detail.

## Artifact policy

The repository tracks source code, lightweight configuration, tests, and
protocol documentation. Raw data, converted instances, checkpoints,
TensorBoard logs, and experiment outputs are excluded by `.gitignore` and must
be retained separately when reproducing historical results. Published claims
must remain traceable to the hashes recorded in the frozen configurations and
experiment reports.
