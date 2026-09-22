# 新对话交接：研究口径调整与后续实验（2026-09-12）

## 1. 先读什么、先做什么

主需求文件：[research_requirements_and_plan_20260912.md](research_requirements_and_plan_20260912.md)。它区分用户已确认要求、助手建议和未验证假设，是本轮后续设计的入口；旧训练手册描述的是旧版，不自动适用于新版。

**当前不应立即启动训练。** 先只读核对实现和数据，提出目标函数校准与四拓扑数据协议，等待用户确认关键项。此次仅新增这两份文档，没有修改代码、权重、数据划分或运行实验。

项目位置：`D:\project\Resource_Allocation`；PowerShell；项目解释器：`.\.venv\Scripts\python.exe`。不要切换到系统 Python。Gurobi 12.0.3 曾在该虚拟环境通过许可证及求解校验，新运行前可轻量复核；历史验证不等于当前已再次验证。

工作区已有大量修改和未跟踪文件，属于此前工作，禁止重置/清理或覆盖。先检查 `git status --short` 和实际存在的 `AGENTS.md`。用户要求关键问题必须显式答复：用 `【需要你回复，未回复不会继续】` 标出当前阻塞决策，不以默认同意代替回答。

## 2. 已完成的旧版结果，不要重跑或误判为新版

### PPO 训练

- 五拓扑；旧权重 `0.30/0.25/0.25/0.10/0.10`。
- joint 同步多环境采样，B=4，rollout=128，联合 PPO 更新；lr=3e-4，epochs=4，minibatch=64，entropy coefficient=0.01，hidden=64，target_kl 关闭。
- seed0/1/2 均完成 1500 updates；候选参数搜索和可选 target-KL 回退已实施，不应当成待从零补齐的旧计划。
- 旧版 validation 选择模型：`results/formal_joint_seed1_run01/checkpoints/best.pt`，update 1450，validation J=0.67109445。
- 历史 seed0 update 725 best 是旧阶段参照，不是当前最终选出的旧版部署模型；现有 checkpoint 均应保留。
- 三种子日志中 PPO update 计时合计约 1.88 小时、validation 推理约 1.43 小时，总计 3.31 小时；不是完整墙钟耗时，不含启动、写盘、中断等开销。

旧版测试，每类 675 个配对样本：

| 场景 | 标量 greedy | PPO seed1 | 单点 best-improvement local search | PPO 三种子均值 ± 总体 SD |
|---|---:|---:|---:|---:|
| interpolation | 0.655647 | 0.671166 | 0.687444 | 0.669342 ± 0.001292 |
| held-out 画像组合 | 0.656164 | 0.671739 | 0.688414 | 0.670409 ± 0.000944 |

两类测试已运行并被查看。旧报告中“封存”指当时选参阶段的流程，不代表现在仍未查看。held-out 不是未见拓扑测试。

历史分项分析发现，PPO 相对标量 greedy 的 J 提升主要来自全局分布项，不能只凭 J 高就声称学生、教师偏好均改善。新版必须复核分项与偏好响应。

证据入口：

- [旧版参数与最终评估报告](ppo_parameter_search_and_final_evaluation_20260908.md)
- `results/ppo_parameter_search/`：候选、冻结与审计产物。
- `results/final_evaluation/`：正式旧版测试及多种子表。
- `results/publication_ready_20260908/README.md`：整理包入口。
- 该整理包 `figures/` 中有 training_dynamics、quality_speed_comparison、generalization_by_topology 三组 SVG/PDF/PNG，`source_data/` 和 `tables/` 保存数据。
- `results/formal_joint_seed1_run01/tensorboard_rebuilt`：清理无效恢复段后的 TensorBoard。

### Gurobi / 混合方案 validation pilot

- 45 个实例 = 五拓扑 × 九规模组，各格一个中位选中节点数实例。
- 原始规模组 V=60/80/100/150/200/300/500/800/1000；实际分配节点 N=40–736，M=11。V 与 N 不可混用。
- 45 个均在首轮 120 秒预算内证得最优，无需追加 600 秒；最大目标重算误差 `3.786e-14`。
- pilot 结果均值：PPO J≈0.673091，局部搜索 J≈0.690365，Gurobi J≈0.691422；PPO 平均逐实例相对 gap≈2.659%。
- PPO+4/8/16/32 accepted moves 已比较；没有候选同时达到“恢复至少 50% PPO—局部搜索差距且时间不超过局部搜索 20%”的原门禁，不能说混合预算已冻结。
- 计时复测中 Gurobi 平均约 1.869 秒、PPO 约 1.091 秒、局部搜索约 21.604 秒；但 Gurobi 是独立重测，PPO 初始化边界仍有差异，且缺少统一重复测量。这里只用于定位历史记录，不是最终公平加速结论。

关键路径：

- `results/exact_hybrid_validation_pilot/paired_results.csv` 与 `.jsonl`
- `results/exact_hybrid_validation_pilot/run_metadata.json`
- `results/exact_hybrid_validation_pilot/gurobi_logs/`
- `results/exact_hybrid_validation_pilot/fair_runtime/paired_results_fair_runtime.csv`
- `results/exact_hybrid_validation_pilot/fair_runtime/gurobi_runtime_remeasurement.csv`
- `results/exact_hybrid_validation_pilot/summary_fair_runtime/decision_report.md`
- 同目录 `algorithm_summary.csv`、`topology_summary.csv`、`hybrid_budget_decision.json`

[旧精确求解状态文档](exact_hybrid_status_20260908.md)停留在 smoke 后，仍写“大规模 pilot 待决定”，已滞后；以上实际 pilot 产物优先。历史局部搜索比 PPO 质量高，不意味着局部搜索给出全局最优上界。

## 3. 新版方向速记

已确认：排除 staircase（训练、validation、测试），保留四拓扑并接受重新训练；教学效果可以为偏好让步，不设效果下限；学生教师等权；偏好为重点；全局项倾向防止方案过于单一的辅助作用；软惩罚单独校准。

未冻结：正向系数归一化的含义、实际权重、F_G/rho 设计、软阈值/惩罚扫描、新数据划分及独立确认来源、新训练预算与验收阈值。`0.35/0.30/0.30/0.05` 只是建议起点，不可直接写入配置。

研究顺序：目标/数据校准 → 新版 pilot 与偏好响应/搜索预算诊断 → 重点批量服务比较 → 多种子与独立确认 → 可选动态调整/消融。完整约束和处理方案见主需求文件。

## 4. 实现地图

以下为项目根目录相对路径；使用前读取实际文件和接口，不假设参数可由 checkpoint 自动补齐。

| 功能 | 入口 |
|---|---|
| 方法效果、画像/权重、默认配置、场景配置 | `pa_moap_rl/configs/{method_config,profile_config,default,scenario_config}.yaml` |
| 数据转换、划分、画像与实例构造 | `pa_moap_rl/data/{convert_pre_data,build_dataset,scenarios,build_assignment_instance}.py` |
| 统一评分与结果重算 | `pa_moap_rl/utils/scoring.py`、`pa_moap_rl/utils/metrics.py` |
| 环境与增量改进 | `pa_moap_rl/envs/method_assignment_env.py` |
| PPO 核心及批量训练 | `pa_moap_rl/solvers/ppo_solver.py`、`pa_moap_rl/experiments/train_ppo_batch.py` |
| 正式训练与 checkpoint 评估 | `pa_moap_rl/experiments/formal_training.py`、`evaluate_checkpoint.py` |
| 现有单点最优改进局部搜索 | `pa_moap_rl/solvers/local_search_solver.py` 的 `local_search_best_improvement` |
| 局部搜索规范名称兼容入口 | `pa_moap_rl/solvers/one_point_local_search.py`，不新增重复算法 |
| 精确/有界模型及混合精炼 | `pa_moap_rl/solvers/gurobi_exact_solver.py`、`hybrid_solver.py` |
| 精确/混合 pilot、汇总、时间复测 | `pa_moap_rl/experiments/{compare_exact_hybrid,summarize_exact_hybrid,remeasure_gurobi_runtime}.py` |
| 监控与结果整理 | `docs/training_monitoring_data.md`、`pa_moap_rl/experiments/prepare_publication_results.py` |

旧数据入口：`pre_data/`、`data_processed/base_instances/manifest.csv`、`data_processed/splits/`、`instance_profile_v2/`。需要从 manifest/生成脚本追溯实际对应关系，而非直接复制旧 split 并删除 staircase 后声称完成独立新测试。

## 5. 优先核查的工程与方法风险

1. **非默认软参数重算一致性。** `solve_gurobi` 接受自定义软参数，但当前 `make_solver_result` / `evaluate_assignment` 调用 `score_assignment` 时没有传入这些自定义参数。历史默认参数重算正确，不能据此认定惩罚扫描也正确。先统一配置传递、环境增量评分和各求解器，并对非默认值做枚举/一致性回归。
2. `default.yaml` 当前有重复的 `target_kl: null`；旧数学文档权重与实际配置不一致。列入后续获准实施时的清理，不能让文档修订偷偷变更实验配置。
3. PPO 当前从 effect-only greedy 初解做单点替换，返回 best-so-far；评估一般为 masked argmax，max_steps=128、patience=32。延长推理预算、改变初解均需验证，不能保证必然提升。
4. 现有局部搜索核心不是新写的算法；别名和混合诊断已补充，不要假设局部搜索已有全部拟议的扫描/early-stop 诊断字段。
5. Gurobi “追加 600 秒”实现是用 incumbent 重新建模热启动，不是接续同一搜索树。新协议需明确计时含义。
6. 历史 seed1 曾因 CSV 锁冲突及一次恢复默认参数错误产生无效段，已隔离到 `results/formal_joint_seed1_run01/audit_invalid_resume_defaults_20260907`。不要混入正式统计；恢复需显式核对所有运行参数。
7. 旧 tests 曾报告全项目 68 passed、精确专项 3 passed；这是历史结果，此次文档交接未重跑测试。新版参数传递与评分修改后需重新验证。
8. 保留 mask 能力，但当前应用基本无禁用关系，不能用人为添加禁用关系制造算法优势。
9. 不承诺 Gurobi 慢、PPO 偏好适应独有、混合方案必胜或排除 staircase 后 PPO 一定更好。偏好变化本身对 Gurobi 是可更新的目标系数；效率优势需要合理线程/并发和端到端质量匹配测量。

## 6. 给新对话的启动提示词

可直接复制以下内容：

> 请接续 D:\project\Resource_Allocation 项目。先读取 AGENTS.md（如存在）、docs/research_requirements_and_plan_20260912.md 和 docs/handoff_research_restart_20260912.md，再只读核对相关实现、配置、数据划分和实验产物。当前先做“四拓扑新版研究的目标函数校准与数据协议设计”，不要修改代码/权重/数据，也不要启动训练或实验。请区分用户已确认要求、建议参数和未验证假设，给我一个可以落实的下一阶段方案：澄清正向系数归一化的含义，说明 F_G/rho 与熵/占比惩罚的分工，提出候选权重、惩罚扫描、样本选择、分项评价及选型准则，并审计旧测试已查看之后独立确认数据如何安排。指出目标评分传参等必须先修的问题，列出实施顺序和产物。仅把本阶段真正阻塞的关键选择集中问我，必须等待我明确回复再执行依赖这些选择的修改或实验。不要重新运行已经完成的旧实验，也不要把旧版参数冻结当成新版已冻结。

新对话第一轮的预期产物是“只读核对结论 + 目标校准/数据协议方案 + 必须确认的问题”，不是新的 checkpoint。用户确认后再转入实现。
