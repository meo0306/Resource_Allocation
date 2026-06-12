# pa_moap_rl 中文实施计划

## 总体说明

第一版实现严格围绕 `docs/scheme_final.md` 的 PA-MOAP 求解方案展开：输入来自内容一已经选中的知识点集合，本阶段只负责为每个已选知识点分配一种教学方法，不重复处理先修约束、`TeachingTime`、`ExternalResourceDemand` 等容量约束。

核心数据流：

1. `.sm` 原始算例提供知识点类别与认知负荷。
2. 内容一结果 CSV 提供知识点选择向量 `x`。
3. 根据 `category_id + cognitive_load` 构造 `concept_need: [N, 6]`。
4. 从 YAML 配置读取方法矩阵、画像、目标方法分布和权重。
5. 构造 `assignment_instance`，包含静态评分矩阵与静态可行性掩码。
6. solver/env 在动态状态中维护 `assignment`、`method_distribution`、`current_scores`。
7. 动作是单点替换 `(i, m)`。
8. reward 定义为 `J(next_assignment) - J(current_assignment)`。

已确认输入规则：

- `.sm` 文件以 `examples/inst_V60_w6-10_s670487.sm` 为参照。
- 节点属性区格式为：`nodnr. <type> <p_1> <p_2> <p_3> <q_1> <q_2> <q_3>`。
- `.sm` 中 `type -> category_id`。
- `.sm` 中 `q_2 -> cognitive_load`。
- 内容一结果 CSV 以 `examples/summary_ga_own_group1_260417_210154.csv` 为参照。
- CSV 中 `x` 字段是内容一决策变量向量，`x[j-1] == 1` 表示 `.sm` 原始节点 `nodnr=j` 被选中。
- `student_profile`、`teacher_profile`、`target_distribution`、`weights` 先放在可修改 YAML 配置中，后期可替换为算例级配置。
- `feasible_mask` 第一版默认为全 `True`。
- PPO 网络默认支持 batch，smoke test 同时覆盖单实例和 batch。

## Phase 0：工程骨架与依赖

生成 Python 包结构、项目配置和测试入口。

新增：

- `pa_moap_rl/`
- `pyproject.toml`
- `README.md`
- `tests/`

默认依赖：

- runtime：`numpy`, `pyyaml`, `torch`, `pandas`, `matplotlib`, `tqdm`
- dev/test：`pytest`

验收：

- `python -m compileall pa_moap_rl`
- `python -c "import pa_moap_rl"`
- `pytest -q` 可发现测试

## Phase 1：配置与数学定义冻结

生成配置文件和工程版数学说明，作为后续代码唯一公式依据。

生成：

- `pa_moap_rl/configs/default.yaml`
- `pa_moap_rl/configs/method_config.yaml`
- `pa_moap_rl/configs/profile_config.yaml`
- `docs/model_definition.md`

实现内容：

- 固定 6 类知识点类别：基础、理论、算法、实验、案例、扩展。
- 固定 11 类教学方法。
- 写入 `D_C: [6,5]`、`R: [11,6]`、`A_E: [11,6]`。
- 写入 `lambda_q=0.6`、`lambda_r=0.4`、`alpha_stu=alpha_tea=0.75`、`H_min=0.55`、`pi_cap=0.35`、`lambda_div=lambda_cap=1.0`。
- 标量目标统一为：`J = w_E F_E + w_S F_S + w_T F_T + w_G F_G - w_soft V_soft`。
- 配置中显式提供默认 `student_profile`、`teacher_profile`、`target_distribution`、`weights`，后期可直接修改。

验收：

- 配置 loader 校验矩阵 shape。
- scoring 代码不写死权重、画像和目标分布。
- 不引入随机扰动、隐藏归一化、0.1 截断或未声明修正。

## Phase 2：输入适配与 Assignment Instance

实现从 `.sm + 内容一 CSV` 到内容二 instance 的构造链路。

生成：

- `pa_moap_rl/data/parse_sm.py`
- `pa_moap_rl/data/selection_loader.py`
- `pa_moap_rl/data/build_assignment_instance.py`
- `pa_moap_rl/data/convert_pre_data.py`
- `pa_moap_rl/data/instance_schema.py`
- `pa_moap_rl/data/loader.py`

实现内容：

- `.sm` parser 读取 instance 元信息、节点属性、容量信息；内容二只使用 `type` 和 `q_2`。
- `type` 按配置映射到 `category_id`。
- `q_2` 直接作为 `cognitive_load`。
- CSV loader 过滤 `row_type == "instance"`，解析带换行的 NumPy 风格 `x` 字段。
- `x` 的位置映射为 `.sm` 原始节点编号：`index 0 -> nodnr 1`。
- builder 抽取被选中节点，重新编码为 `0..N-1`。
- 支持 JSON 输入：`{"selected_ids": [...]}` 和 `{"x_star": [0,1,...]}`。
- 提供批量转换脚本，将 `pre_data/instance` 与 `pre_data/x` 中按 group 配对的 `.sm + x` 结果转换为处理后的内容二算例。
- 处理后算例默认保存到 `instance/group*/<instance_name>.assignment.json`，每组生成 `manifest.csv`，根目录生成总 `manifest.csv`。
- 保存格式使用可读 JSON；只保留后续优化所需的 assignment instance 字段，不保存完整先修关系、容量和原始 p/q 字段。

构造字段：

- `category_id: [N]`
- `cognitive_load: [N]`
- `concept_need: [N,6]`
- `method_attr: [M,6]`
- `effect_matrix: [N,M]`
- `student_pref: [N,M]`
- `teacher_pref: [N,M]`
- `target_distribution: [M]`
- `feasible_mask: [N,M]`

验收：

- `concept_need[:, :5] == D_C[category_id]`。
- `concept_need[:, 5] == cognitive_load`。
- `effect_matrix[i,m] == A_E[m, category_id[i]]`。
- 每个知识点至少一个 feasible method。
- JSON/NPZ 保存加载后数值一致。
- 示例 `.sm` 可解析出 60 个节点，示例 CSV 第一行可解析出对应选择向量。
- `pre_data` 全量转换可完成，输出 group 数量、算例数量和每个 JSON 的 shape 与 schema 校验通过。

## Phase 3：Scoring 与 Mask 内核

实现目标函数、软约束和动作掩码。

生成：

- `pa_moap_rl/utils/scoring.py`
- `pa_moap_rl/utils/masks.py`

评分函数：

- `method_distribution(assignment, M)`
- `F_E`
- `F_S`
- `F_T`
- `F_G = 1 - 0.5 * sum(abs(pi - rho))`
- `H`
- `V_div = max(0, H_min - H)^2`
- `V_cap = sum(max(0, pi - pi_cap)^2)`
- `V_soft = lambda_div * V_div + lambda_cap * V_cap`
- `J`
- 单点替换 delta scoring

mask 规则：

- `feasible_mask` 是静态方法可行性掩码，只来自 instance/config。
- `action_mask` 是动态替换动作掩码。
- 单实例：`action_mask[i,m] = feasible_mask[i,m] and m != assignment[i]`。
- batch：`action_mask[b,i,m] = node_mask[b,i] and feasible_mask[b,i,m] and m != assignment[b,i]`。

验收：

- 所有局部目标使用平均值，不随 `N` 放大。
- 同一 assignment 重复评分完全一致。
- 单点 delta 与完整重算差值一致。
- 当前方法永远不能作为替换动作。
- padding 节点在 batch mask 中永远不可选。

## Phase 4：Gymnasium-like 环境

封装教学方法分配环境。

生成：

- `pa_moap_rl/envs/method_assignment_env.py`

接口：

```python
env.reset(instance, initial_assignment=None) -> obs
env.step(action: tuple[int, int]) -> obs, reward, done, info
env.get_action_mask() -> np.ndarray
```

行为：

- `assignment: np.ndarray[int], shape=[N]`。
- action 表示单点替换 `(node_index, method_index)`。
- 非法 action 抛出清晰异常。
- reward = `J(next_assignment) - J(current_assignment)`。
- 维护 `best_assignment`、`best_score`、`steps_to_best`。
- done 条件为 `step_count >= max_steps` 或 `no_improve_steps >= patience`。

验收：

- reset 后 obs 字段完整。
- step 合法动作后动态状态同步更新。
- best-so-far 不因负 reward 丢失。
- max steps 和 patience 两类终止条件均有测试。

## Phase 5：Baseline Solvers

实现对比算法，作为 PPO 的基础参照。

生成：

- `pa_moap_rl/solvers/random_solver.py`
- `pa_moap_rl/solvers/greedy_solver.py`
- `pa_moap_rl/solvers/local_search_solver.py`
- `pa_moap_rl/utils/metrics.py`

实现：

- random legal
- effect-only greedy
- scalarized independent greedy
- scalarized one-point greedy improvement
- local search best-improvement
- local search first-improvement
- random restart local search

验收：

- 所有 baseline 输出满足 `feasible_mask`。
- effect-only greedy 对每个节点选择 `effect_matrix` 最大的可行方法。
- one-point greedy 每步 `J` 单调不降。
- local search 终止时不存在正收益单点替换。
- metrics 输出完整：可行性、mask violation、各项 score、runtime、improvement。

## Phase 6：Actor-Critic 网络

实现 `docs/scheme_final.md` 中的网络结构。

生成：

- `pa_moap_rl/models/encoders.py`
- `pa_moap_rl/models/actor_critic.py`

shape 要求：

- Node Encoder 输出 `h: [B, N_max, H]`
- Method Encoder 输出 `mu: [B, M, H]`
- Global Encoder 输出 `g: [B, H]`
- Pair Feature Builder 输出 `xi: [B, N_max, M, 3H + 9]`
- Actor Head 输出 `logits: [B, N_max, M]`
- Critic Head 输出 `value: [B, 1]`

强制规则：

- Actor 必须执行 `masked_logits = action_logits.masked_fill(~action_mask, -1e9)`。
- Critic 只使用 global representation `g`。
- 动作 flatten 映射固定为：
  - `flat = i * M + m`
  - `i = flat // M`
  - `m = flat % M`

验收：

- batch 前向传播 shape 全部正确。
- masked action 不会被采样。
- `log_prob`、`entropy`、`value` 可反向传播。
- 全 mask 状态抛出明确错误。

## Phase 7：Minimal PPO 闭环

实现最小可运行 masked PPO。

生成：

- `pa_moap_rl/solvers/ppo_solver.py`
- `pa_moap_rl/experiments/train_ppo.py`

实现：

- rollout buffer
- GAE
- clipped PPO loss
- value loss
- entropy bonus
- gradient clipping
- checkpoint 保存
- training log CSV

第一版目标：

- 单实例 smoke test 跑通。
- batch smoke test 跑通。
- reward 不爆炸。
- illegal action rate = 0。
- 不要求一开始超过所有 baseline。

验收：

- `python -m pa_moap_rl.experiments.train_ppo --smoke-test`
- 产生训练日志。
- episode 正常结束。
- best-so-far assignment 可导出。

## Phase 8：批量实验、消融与分析

实现实验脚本和结果分析。

生成：

- `pa_moap_rl/experiments/run_batch.py`
- `pa_moap_rl/experiments/run_ablation.py`
- `pa_moap_rl/experiments/analyze_results.py`

实现：

- 批量读取 `.sm + 内容一 CSV`。
- 批量运行 baseline 与 PPO。
- 保存 `results/*.csv`。
- 生成 PPO reward curve。
- 生成 total score baseline 对比柱状图。
- 生成 effect/preference trade-off 散点图。
- 生成方法分布 vs 目标分布图。

消融：

- 无偏好 vs 有偏好。
- 无全局分布项 vs 有全局分布项。
- 启发式初解 vs 随机初解。

验收：

- 结果 CSV 字段完整。
- 图表脚本可仅从 CSV 复现图。
- 同 seed 结果可复现。


## 明确假设

- 内容二默认不排除 `.sm` 中任何节点，除非后续 `.sm` 明确标注虚拟源点/汇点。
- `.sm` 中的类别字符串必须能映射到 6 类配置，否则 fail fast。
- 第一版 `feasible_mask` 全 True。
- 配置文件中的默认 profiles、target distribution、weights 是可运行默认值，不代表最终实验参数锁死。
- PPO 第一版目标是闭环正确、mask 正确、训练稳定，而不是立即优于所有 baseline。
