# Repository Structure

This document describes the current high-level layout. Individual experiment
entrypoints and versioned YAML files are grouped by role instead of being
exhaustively listed.

```text
Resource_Allocation/
  README.md
  pyproject.toml
  requirements-exact.txt

  pa_moap_rl/
    checkpointing.py
    objective.py

    configs/
      default.yaml
      method_config.yaml
      profile_config.yaml
      objective_*.yaml
      ppo_*.yaml
      preference_*.yaml
      service_*.yaml

    data/
      parse_sm.py
      selection_loader.py
      build_assignment_instance.py
      convert_pre_data.py
      build_dataset.py
      build_four_topology_protocol.py
      build_service_scenario_bank_v1.py
      scenarios.py

    envs/
      method_assignment_env.py

    models/
      encoders.py
      actor_critic.py
      preference_repair_v2.py

    solvers/
      random_solver.py
      greedy_solver.py
      local_search_solver.py
      one_point_local_search.py
      gurobi_exact_solver.py
      hybrid_solver.py
      ppo_solver.py

    experiments/
      train_*.py
      formal_training.py
      evaluate_*.py
      run_*.py
      summarize_*.py
      diagnose_*.py
      prepare_publication_results.py

    utils/
      scoring.py
      masks.py
      metrics.py
      seed.py
      io.py

  tests/
    test_*.py

  docs/
    project_status_20260922.md
    mission_list.md
    model_definition.md
    formal_training_guide.md
    *_protocol_*.md
    *_report.md

  examples/
    instance_standardization_rule.md
    *.sm
    *.csv

  pre_data/                 # local raw inputs; ignored
  data_processed/           # generated datasets; ignored
  instance*/                # generated assignment instances; ignored
  results/                  # checkpoints and experiment outputs; ignored
  figures/                  # generated figures; ignored
  tables/                   # generated tables except .gitkeep; ignored
```

## Responsibilities

- `configs/` stores versioned and hash-audited objective, scenario, training,
  inference, and evaluation specifications. Frozen files must not be silently
  overwritten by exploratory configurations.
- `data/` parses upstream `.sm` and selection CSV files, creates assignment
  instances, constructs leakage-aware splits, and generates controlled
  topology and preference scenarios.
- `envs/` implements the masked single-node replacement MDP and preserves the
  best feasible assignment seen within an episode.
- `models/` contains the original actor-critic and separately versioned
  preference-repair architectures.
- `solvers/` exposes random, scalarized greedy, local search, Gurobi exact,
  hybrid, and PPO inference paths under common scoring semantics.
- `experiments/` contains reproducible command-line workflows for calibration,
  training, checkpoint selection, controlled response evaluation, ablation,
  service efficiency, dynamic preference analysis, and publication tables.
- `tests/` covers mathematical scoring, masks, environments, model execution,
  exact/hybrid consistency, checkpoint contracts, frozen-mainline invariants,
  and experiment-protocol gates.
- `docs/` records mathematical definitions, preregistered protocols, decision
  boundaries, milestone reports, and the authoritative task ledger.

## Implementation boundaries

- The assignment stage uses the upstream-selected nodes; it does not re-solve
  precedence or upstream resource-capacity decisions.
- Node `type` maps to `category_id`, and `q_2` supplies cognitive load.
- Feasibility masks remain supported even though the current research data
  generally permits all teaching methods.
- Batch padding is controlled by `node_mask`; action feasibility is controlled
  separately by static and dynamic masks.
- Generated data and results are not source-controlled. Reproducibility relies
  on the recorded configuration, code, input, checkpoint, and output hashes.
