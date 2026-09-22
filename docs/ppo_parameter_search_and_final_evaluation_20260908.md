# PPO 参数冻结与最终评估报告

日期：2026-09-08

## 最终结论

主模型参数已经冻结，不再继续搜索：

| 参数 | 冻结值 |
|---|---:|
| rollout mode | joint |
| instances per update | 4 |
| rollout steps | 128 |
| minibatch size | 64 |
| learning rate | 3e-4 |
| PPO epochs | 4 |
| entropy coefficient | 0.01 |
| hidden dimension | 64 |
| target KL | disabled |
| 正式训练预算 | 1500 updates |

三种子均完成 1500 updates。部署模型按固定 validation 选择为：

`results/formal_joint_seed1_run01/checkpoints/best.pt`

该 checkpoint 对应 update 1450，固定 validation 分数为 0.67109445。`latest.pt` 只用于恢复训练，不能替代 `best.pt` 做部署或算法比较。

## 300-update 参数初筛

所有候选均使用 seed0、同一批 1–300 update 实例路径及 scenario ID；逐 update、逐环境核对结果完全一致。测试集在此阶段保持封存。

| 候选 | lr | epochs | S | 最佳验证 | KL P90 | clip mean | entropy min/end | 最弱拓扑差 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B | 3e-4 | 4 | 0.654349 | 0.662251 | 0.044205 | 0.204583 | 0.624863 / 0.681387 | 0 | 合格，胜出 |
| C1 | 3e-4 | 3 | 0.657075 | 0.658264 | 0.040854 | 0.194583 | 0.588167 / 0.710507 | -0.006885 | 最佳质量、拓扑未过门槛 |
| C2 | 3e-4 | 2 | 0.655388 | 0.657917 | 0.036500 | 0.163125 | 0.597648 / 0.690052 | -0.006810 | 最佳质量、拓扑未过门槛 |
| C3 | 2e-4 | 4 | 0.650257 | 0.658985 | 0.036014 | 0.174479 | 0.470734 / 0.498890 | -0.006830 | 质量、拓扑、熵未过门槛 |
| C4 | 3e-4 | 4 + target KL 0.03 | 0.652867 | 0.657824 | 0.045285 | 0.212969 | 0.555855 / 0.649896 | -0.007834 | 最佳质量、拓扑未过门槛 |

C1/C2 的更新更保守，确实降低了 KL 和 clip fraction，但没有换来足够的最佳解质量，且最弱拓扑退化超过 0.005。C3 出现明显熵衰减。按回退规则运行的 C4 在 300 updates 中触发 target-KL early stop 55 次，实际 epochs 均值 3.753、范围 1–4，但仍未改善质量。因此停止扩展参数搜索，采用 B。

完整初筛产物位于：

- `results/ppo_parameter_search/round_300_final/candidate_summary.csv`
- `results/ppo_parameter_search/round_300_final/training_schedule_pairs.csv`
- `results/ppo_parameter_search/round_300_final/validation_pairs.csv`
- `results/ppo_parameter_search/round_300_final/topology_comparison.csv`
- `results/ppo_parameter_search/round_300_final/decision_report.md`

## update 1000 冻结门禁

| seed | 最佳验证 | 最佳 update | 尾四点均值 | 峰值减尾段 | KL P90 | entropy min | 可行率 | 非法动作 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.667432 | 725 | 0.656588 | 0.010844 | 0.044691 | 0.538824 | 100% | 0 |
| 1 | 0.665086 | 900 | 0.664109 | 0.000977 | 0.041409 | 0.564379 | 100% | 0 |
| 2 | 0.668755 | 875 | 0.665055 | 0.003700 | 0.042346 | 0.524959 | 100% | 0 |

三种子最佳分数中位数 0.667432，总体标准差 0.001517；KL P90 中位数 0.042346、最大值 0.044691。全部冻结门禁通过。seed1 与 seed2 的最佳点均在 update 800 之后，因此满足统一延长到 1500 的条件。

对应产物位于 `results/ppo_parameter_search/freeze_u1000`。

## update 1500 最终长程结果

| seed | 最佳验证 | 最佳 update | 尾四点均值 | 峰值减尾段 | KL P90 | entropy min |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.669582 | 1150 | 0.664751 | 0.004831 | 0.044660 | 0.538824 |
| 1 | 0.671094 | 1450 | 0.667256 | 0.003839 | 0.043168 | 0.550995 |
| 2 | 0.668755 | 875 | 0.658211 | 0.010544 | 0.043282 | 0.524959 |

最终最佳分数中位数 0.669582，总体标准差 0.000969；全部种子硬可行且非法动作数为 0。seed0 和 seed1 在延长段刷新最佳，seed2 保持 update 875 的最佳。

五拓扑在各 seed 最佳点的跨种子统计全部通过。尤其 staircase 的中位数为 0.609140，相对 seed0 参考高 0.001766，不存在系统性退化。

对应产物位于 `results/ppo_parameter_search/final_u1500`。

## 冻结后封存测试与算法对比

参数冻结后才运行 test interpolation 和 test held-out。每类测试均覆盖 675 个相同实例；seed1 同时运行随机可行、Effect-only greedy、标量 greedy、局部搜索和 PPO。三种子 PPO 的 instance path 与 scenario ID 配对完全一致，均为硬可行且 mask violation 为 0。

### 部署模型与基线

| 场景 | random | Effect-only | scalar greedy | PPO seed1 | local search |
|---|---:|---:|---:|---:|---:|
| interpolation | 0.603897 | 0.619974 | 0.655647 | 0.671166 | 0.687444 |
| held-out | 0.604979 | 0.620868 | 0.656164 | 0.671739 | 0.688414 |

PPO 在两类测试中分别比标量 greedy 高 0.015519 和 0.015575；未见画像组合测试没有下降，反而比 interpolation 高约 0.000573。PPO 尚未超过完整 best-improvement local search，差距分别为 0.016278 和 0.016675。

速度差异很明显：

| 场景 | PPO mean runtime | local-search mean runtime | local-search max runtime | PPO 加速比 |
|---|---:|---:|---:|---:|
| interpolation | 0.3317 s | 22.1447 s | 134.8307 s | 约 66.8 倍 |
| held-out | 0.3265 s | 22.2246 s | 129.0832 s | 约 68.1 倍 |

因此当前结果支持如下叙事：PPO 学习从 Effect-only greedy 初解出发进行快速邻域改进，显著优于不搜索或独立贪婪方案；完整局部搜索仍提供更高的解质量上界，但推理成本高两个数量级左右。PPO 的主要优势是质量与部署时延的折中，而不是在当前版本中声称全面超过局部搜索。

### 三种子 PPO 泛化

| 场景 | PPO 三种子均值 | 种子总体标准差 | staircase 均值 | staircase 标准差 |
|---|---:|---:|---:|---:|
| interpolation | 0.669342 | 0.001292 | 0.613729 | 0.002150 |
| held-out | 0.670409 | 0.000944 | 0.616328 | 0.002549 |

五类拓扑在两类测试中均为 100% 硬可行、0 mask violation。staircase 仍是绝对分数最低的拓扑，应在论文中单列，但它在 held-out 上没有恶化。

正式测试产物位于：

- `results/final_evaluation/seed1_best_test_interpolation_with_baselines.csv`
- `results/final_evaluation/seed1_best_test_heldout_with_baselines.csv`
- `results/final_evaluation/interpolation_tables`
- `results/final_evaluation/heldout_tables`
- `results/final_evaluation/interpolation_multiseed`
- `results/final_evaluation/heldout_multiseed`

## 工程修正与审计说明

本轮增加了 target-KL 可选回退、候选选择器、三种子冻结器、测试配对汇总器、TensorBoard 重建器、评估 offset 分块，以及 Windows CSV 原子替换的短锁重试。完整回归测试通过。

seed1 首次训练时，实时读取活跃 CSV 与 Windows 原子替换短暂冲突，训练在 update 548 退出；有效 checkpoint 为 update 525。随后一次简写恢复意外使用 CLI 默认运行参数，已在 checkpoint 节奏异常时识别。无效 update 526–690 的全部文件均移动到：

`results/formal_joint_seed1_run01/audit_invalid_resume_defaults_20260907`

随后恢复有效的 update 350 best 和 update 525 latest，并用完整原参数从 update 525 重新训练。最终 seed1 的 CSV 包含 1500 个 update 汇总、6000 个训练实例条目和 60 个验证点，无效段不在正式数据中。干净的完整 TensorBoard 重建目录为：

`results/formal_joint_seed1_run01/tensorboard_rebuilt`

恢复训练时必须显式传入完整训练参数；当前接口不能把省略的运行级参数全部自动继承为原值。

## 尚未执行的部分

现有 `run_ablation.py` 是启发式目标项消融，不训练或评估 PPO，不能直接作为主模型消融。根据此前允许消融后置，本轮没有把它误作为正式 PPO 消融。下一阶段应先明确并实现至少以下受控版本，再从随机初始化训练：去掉 Effect-only greedy 初解、去掉 action mask 或改成可行性惩罚、去掉偏好条件输入，以及必要时 PPO 后接短程局部搜索。消融不能复用主模型权重后直接改目标来替代重新训练。
