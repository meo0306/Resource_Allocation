# Repo Structure

```text
Resource_Allocation/
  pyproject.toml
  README.md

  pa_moap_rl/
    __init__.py

    configs/
      __init__.py
      loader.py
      default.yaml
      method_config.yaml
      profile_config.yaml

    data/
      __init__.py
      parse_sm.py
      selection_loader.py
      build_assignment_instance.py
      convert_pre_data.py
      instance_schema.py
      loader.py

    envs/
      __init__.py
      method_assignment_env.py

    models/
      __init__.py
      encoders.py
      actor_critic.py

    solvers/
      __init__.py
      random_solver.py
      greedy_solver.py
      local_search_solver.py
      ppo_solver.py

    utils/
      __init__.py
      scoring.py
      masks.py
      metrics.py
      seed.py
      io.py

    experiments/
      __init__.py
      train_ppo.py
      run_batch.py
      run_ablation.py
      analyze_results.py

  tests/
    test_parse_sm.py
    test_instance_builder.py
    test_scoring.py
    test_masks.py
    test_env.py
    test_baselines.py
    test_encoders.py
    test_actor_critic.py

  docs/
    scheme_final.md
    implementation_plan.md
    model_definition.md
    repo_structure.md
    GlobalAcceptanceCommand.txt

  examples/
    instance_standardization_rule.md
    inst_V60_w6-10_s670487.sm
    summary_ga_own_group1_260417_210154.csv

  pre_data/
    instance/
      group*/
        *.sm
    x/
      summary_*.csv

  instance/
    group*/
      *.assignment.json
      manifest.csv
    manifest.csv

  results/
    *.csv

  figures/
    *.png
    *.mmd

  tables/
    *.csv
```

## 目录职责

- `pa_moap_rl/configs/`：固定 PA-MOAP 第一版配置，包括知识点类别、教学方法、矩阵参数、画像、目标分布、权重和训练参数。
- `pa_moap_rl/data/`：解析 `.sm`、读取内容一选择结果、构造和保存 `assignment_instance`。
- `pa_moap_rl/envs/`：封装类似 Gymnasium 的 `MethodAssignmentEnv`。
- `pa_moap_rl/models/`：实现 Node Encoder、Method Encoder、Global Encoder、Pair Feature Builder、Actor-Critic。
- `pa_moap_rl/solvers/`：实现 random、greedy、local search、PPO 等求解器。
- `pa_moap_rl/utils/`：实现 scoring、mask、metrics、seed、I/O 等共享工具。
- `pa_moap_rl/experiments/`：提供训练、批量运行、消融和结果分析脚本。
- `tests/`：覆盖 parser、instance builder、scoring、mask、env、baseline、encoder、actor-critic。
- `docs/`：保留数学方案、实施计划、模型定义、架构说明和验收命令。
- `examples/`：保留当前已确认的 `.sm` 标准格式和内容一 CSV 输出示例。
- `pre_data/`：保留内容一输出和原始 `.sm` 批量输入。
- `instance/`：保存转换后的内容二 assignment instance，可直接供后续优化求解使用。
- `results/`、`figures/`、`tables/`：保存实验输出、图和表。

## 当前实现边界

- 内容二只使用 `.sm` 中的 `type` 和 `q_2` 构造实例。
- 内容一 CSV 的 `x` 字段按 `x[j-1] == 1 -> nodnr=j` 解释。
- 第一版 `feasible_mask` 默认为全 `True`。
- PPO 网络需要支持 batch；smoke test 同时覆盖单实例和 batch。
