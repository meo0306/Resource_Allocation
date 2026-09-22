# Joint PPO seed0：update 1000 阶段报告

## 结论

原 joint seed0 已从 update 600 继续训练至 update 1000，运行状态为 completed。新最佳点
出现在 update 725，固定验证分数为 0.6674321，高于 update 450 的 0.6636497。

用于后续评估的主模型是 update 725 `best.pt`；update 1000 `latest.pt` 只用于恢复训练，
不能代替最佳模型进行算法效果比较。

| 项目 | 结果 |
|---|---:|
| 累计 updates | 1000 |
| 固定验证点 | 40 |
| 训练实例记录 | 4000 |
| 验证实例记录 | 3600 |
| 最佳 update | 725 |
| 最佳验证分数 | 0.6674321 |
| update 1000 验证分数 | 0.6601790 |
| 非法动作总数 | 0 |

## Update 600–1000 验证轨迹

| Update | 验证分数 |
|---:|---:|
| 600 | 0.6584034 |
| 625 | 0.6596914 |
| 650 | 0.6542593 |
| 675 | 0.6636752 |
| 700 | 0.6641921 |
| 725 | **0.6674321** |
| 750 | 0.6626542 |
| 775 | 0.6642528 |
| 800 | 0.6625102 |
| 825 | 0.6648404 |
| 850 | 0.6628471 |
| 875 | 0.6515490 |
| 900 | 0.6513815 |
| 925 | 0.6527336 |
| 950 | 0.6538076 |
| 975 | 0.6596321 |
| 1000 | 0.6601790 |

继续训练确实找到了优于 update 450 的模型，说明扩展到 1000 有价值。但 update 725 后的
11 个验证点均未再次刷新最佳，875–950 还出现一段明显回撤，随后在 975/1000 恢复。
当前表现属于带振荡的平台期，而不是单调收敛。

## 五拓扑表现

update 725 的分拓扑验证均值：

| Topology | Update 450 | Update 725 | 差值 |
|---|---:|---:|---:|
| bottleneck | 0.68969 | 0.69354 | +0.00385 |
| hourglass | 0.67060 | 0.67827 | +0.00768 |
| random | 0.67140 | 0.67510 | +0.00370 |
| staircase | 0.60702 | 0.60673 | -0.00029 |
| uniform | 0.67955 | 0.68352 | +0.00397 |

整体提升由四类拓扑共同贡献；staircase 相比 update 450 基本持平，且仍高于 update 25。
后续应把 staircase 作为鲁棒性重点观察对象，不能只报告 overall 均值。

## Update 600–1000 稳定性

| 指标 | 数值 |
|---|---:|
| KL mean | 0.02336 |
| KL median | 0.02005 |
| KL P90 | 0.04791 |
| KL P95 | 0.05492 |
| KL max | 0.09029 |
| KL > 0.03 | 121/400 |
| KL > 0.05 | 34/400 |
| clip fraction mean | 0.22520 |
| clip fraction max | 0.59375 |
| normalized entropy min | 0.53882 |
| normalized entropy at update 1000 | 0.64810 |
| 平均训练吞吐 | 403.60 transitions/s |

与前 600 updates 相比，KL 尾部更重、clip fraction 略高；但熵没有坍缩、非法动作仍为 0，
因此这不是数值性失败。它说明继续延长同一轨迹的边际收益和策略漂移风险都在增加。

## Checkpoint 与数据

- 主模型：`results/formal_joint_seed0_run01/checkpoints/best.pt`，update 725；
- 恢复断点：`results/formal_joint_seed0_run01/checkpoints/latest.pt`，update 1000；
- 完整训练曲线：`results/formal_joint_seed0_run01/metrics/train_updates.csv`；
- 固定验证曲线：`results/formal_joint_seed0_run01/metrics/validation_updates.csv`；
- 逐实例验证：`results/formal_joint_seed0_run01/metrics/validation_instances.csv`；
- TensorBoard：`results/formal_joint_seed0_run01/tensorboard`。

训练期 validation 只评估 PPO 本身，不含随机、Effect-only greedy、标量 greedy 或局部搜索
结果。因此 0.6674321 只能用于同一模型轨迹的 checkpoint 选择，不能单独证明 PPO 优于
对比算法。冻结方案后的独立基线评估仍需另外运行。

## 阶段建议

不建议立刻把 seed0 继续延长到 1500。更有科研价值的下一步是先固定 update 1000 的训练
预算和当前超参数，运行额外独立 seed，确认 update 725 的提升不是单 seed 偶然；同时可在
固定 validation 上运行一次独立基线评估。若多 seed 普遍在 700–1000 仍持续刷新最佳，再
统一决定是否把所有种子延长到 1500。
