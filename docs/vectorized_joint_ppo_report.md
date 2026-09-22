# 多环境同步采样与联合 PPO 改造验收

## 结论

本次改造达到工程采用条件：默认训练流程可以改用 `rollout-mode=joint`。在同一起点、
同一训练实例和场景序列下，joint 将训练吞吐提高约 2.04 倍，同时短程验证质量没有退化，
update 500 的逐实例验证结果整体优于 sequential。

这个结论只回答“改造是否值得采用”。它不等价于主模型已经完成训练，也不构成论文中的
最终算法优越性证据；后者仍需从零开始的独立随机种子训练与冻结测试集评估。

## 实现范围

- 按规模 group 构造训练桶，单批优先抽取不同 topology 和不同 source identity；
- 支持变长节点 observation padding，padding 节点及动作始终被 mask；
- 每一步对多个环境做一次批量策略前向，环境状态转换仍保持独立；
- 每个环境单独计算和标准化 GAE，再合并所有 transition 做一次联合 PPO 更新；
- 保留 `sequential` 模式作为消融和回归基准；
- 在 CSV、JSON 与 TensorBoard 中记录模式、批次组成、吞吐、耗时、显存和 PPO 稳定性指标。

## 正确性与容量验收

- 全量自动化测试：51 passed；
- B=1 时 joint 与原单环境采样的动作、log probability、reward、return、advantage 和 best assignment 完全一致；
- 变长 B=2 测试确认 padding mask 有效，非法动作数为 0；
- 大规模 B=4 容量探针使用 n=695–732 的四种拓扑，512 transitions 全部有限，非法动作数为 0；
- 容量探针峰值 allocated 显存约 1046 MB、reserved 显存约 1390 MB。

## 受控效率比较

两个分支均从原 seed0 的 update 400 `best.pt` 开始，使用相同的 batch sampler、相同模型和
PPO 超参数；逐 update 核对的 20 个 instance/scenario 条目完全一致。

| 指标 | joint | sequential | 对比 |
|---|---:|---:|---:|
| 平均训练吞吐 | 418.08 transitions/s | 208.06 transitions/s | 2.01x |
| 平均 update 耗时 | 1.267 s | 2.493 s | -49.2% |

该 5-update 门禁的验证集仅限 1 个实例，所以只用于性能测量，不用于模型选择或质量判断。

## 受控质量比较

质量门禁继续从同一 update 400 权重开始，两个分支运行到 update 500。400 个训练
instance/scenario 条目逐项一致；每 25 updates 在固定的 90 个验证实例上评估。

| 指标（update 401–500） | joint | sequential |
|---|---:|---:|
| 平均训练吞吐 | 419.03 transitions/s | 205.79 transitions/s |
| 训练总耗时 | 229.72 s | 352.12 s |
| update 425–500 验证均值 | 0.65454 | 0.65200 |
| update 500 验证分数 | 0.65773 | 0.65106 |
| update 500 配对胜率 | 94.44% | — |
| update 500 平均配对差 | +0.00667 | 基准 |
| 非法动作数 | 0 | 0 |

update 500 的五种拓扑均为正差：bottleneck +0.00520、hourglass +0.00747、
random +0.00559、staircase +0.00715、uniform +0.00793。因此短程收益不是由单一拓扑驱动。

两个分支的最佳 checkpoint 仍是继承的 update 400（0.65901）。这说明 100-update 门禁支持
“joint 不劣且更快”，但尚未证明继续精炼一定超过原最佳点。

## 稳定性边界

joint 的更新幅度明显高于 sequential：

| KL 指标 | joint | sequential |
|---|---:|---:|
| 中位数 | 0.01726 | 0.00635 |
| P90 | 0.04078 | 0.01250 |
| 最大值 | 0.06630 | 0.02324 |
| KL > 0.03 | 22/100 | 0/100 |

joint 的 clip fraction 中位数为 0.1406、最大值为 0.6563。当前没有伴随验证崩溃或非法动作，
因此不因单次尖峰否定改造；但正式长跑必须监控 KL、clip fraction、标准化熵和验证分数。
若 KL 连续异常或同时出现验证下降，应把学习率或 PPO epochs 作为第一优先级复核参数。

## 科研解释限制

这次比较从同一已训练 checkpoint 续跑，适合隔离工程改造本身。最终实验仍应至少包括：

1. joint 模式从随机初始化开始的 seed0 正式 pilot；
2. 按预定里程碑检查验证分数、标准化熵、KL、clip fraction 和五拓扑分数；
3. 方案冻结后运行至少 3 个独立种子并报告均值与标准差；
4. 最后才在未参与调参的 interpolation/held-out 测试场景上比较各基线。

原始门禁数据保存在 `results/quality_gate_joint_u400_u500` 与
`results/quality_gate_sequential_u400_u500`。这些目录用于方法比较，不应覆盖或冒充正式主训练目录。
