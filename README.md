# pa_moap_rl

`pa_moap_rl` implements the PA-MOAP teaching method assignment optimizer described in `docs/scheme_final.md`.

The project takes selected knowledge points from the upstream resource-combination stage and assigns one teaching method to each selected point. The upstream stage already handles precedence and resource capacity constraints, so this package uses only `category_id` and `cognitive_load` to build the content-two assignment instance.

## Phase 0 Setup

Create or reuse the local virtual environment, then install the package and development dependencies:

```bash
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Phase 0 validation:

```bash
.\.venv\Scripts\python -m compileall pa_moap_rl
.\.venv\Scripts\python -c "import pa_moap_rl; print(pa_moap_rl.__version__)"
.\.venv\Scripts\python -m pytest -q
```

## PPO Training Entrypoints

Single-instance PPO training:

```bash
.\.venv\Scripts\python -m pa_moap_rl.experiments.train_ppo --instance-json instance/group1/inst_V60_w6-10_s670487.assignment.json --device cpu --updates 20
```

Shared multi-instance PPO smoke training:

```bash
.\.venv\Scripts\python -m pa_moap_rl.experiments.train_ppo_batch --smoke-test --device cpu
```

Pilot shared PPO training over representative group instances:

```bash
.\.venv\Scripts\python -m pa_moap_rl.experiments.train_ppo_batch --group group1 --limit 10 --eval-limit 2 --updates 20 --instances-per-update 2 --rollout-steps 64 --device auto --output-dir results/ppo_batch_group1_pilot
```

TensorBoard live monitoring:

```bash
.\.venv\Scripts\python -m pa_moap_rl.experiments.train_ppo_batch --instance-root instance_profile_v2 --group group1 --limit 10 --eval-limit 2 --updates 20 --instances-per-update 2 --rollout-steps 64 --device auto --output-dir results/ppo_batch_group1_pilot --tensorboard-dir results/tensorboard/ppo_batch_group1_pilot
.\.venv\Scripts\tensorboard.exe --logdir results/tensorboard --port 6006
```

Open `http://localhost:6006` to watch `train/*` and `eval/<solver_name>/*` scalars during training. CSV logs are also refreshed after the initial evaluation and after every update.

Regenerate assignment instances after changing `profile_config.yaml`:

```bash
.\.venv\Scripts\python -m pa_moap_rl.data.convert_pre_data --output-root instance_profile_v2 --overwrite
```

Implementation should follow `docs/implementation_plan.md`; the target structure is documented in `docs/repo_structure.md`.
