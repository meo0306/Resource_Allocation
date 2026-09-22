# Joint PPO seed0：update 600 阶段报告

## 运行结论

从随机初始化开始的首个 joint seed0 已完整训练至 update 600，运行状态为 completed。
本阶段证明正式流程能够覆盖多规模、五拓扑和动态偏好场景持续训练，并生成完整 checkpoint、
CSV、JSON 与 TensorBoard 记录。

用于后续评估和分支训练的模型必须是 update 450 的 `best.pt`，不能误用 update 600 的
`latest.pt`。

| 项目 | 结果 |
|---|---:|
| 训练 updates | 600 |
| 每次更新实例数 | 4 |
| 每个环境 rollout steps | 128 |
| 每次联合更新 transitions | 512 |
| 固定验证间隔 | 25 updates |
| 每次验证实例数 | 90 |
| 训练实例记录 | 2400 |
| 验证实例记录 | 2160 |
| 最佳 update | 450 |
| 最佳验证分数 | 0.6636497 |
| update 600 验证分数 | 0.6584034 |
| 非法动作总数 | 0 |

## 验证趋势

关键验证点如下：

| Update | 验证分数 |
|---:|---:|
| 25 | 0.6589543 |
| 150 | 0.6622512 |
| 300 | 0.6435978 |
| 400 | 0.6595966 |
| 450 | **0.6636497** |
| 550 | 0.6629122 |
| 600 | 0.6584034 |

训练存在明显振荡，但 update 450 刷新全程最佳，update 550 也再次接近该水平。这说明中途
回撤并非不可恢复的训练崩溃，按固定验证集选择 `best.pt` 是必要的。

update 450 相比 update 25，五类拓扑验证均值均提高：

| Topology | Update 25 | Update 450 | 差值 |
|---|---:|---:|---:|
| bottleneck | 0.68606 | 0.68969 | +0.00362 |
| hourglass | 0.66775 | 0.67060 | +0.00285 |
| random | 0.66387 | 0.67140 | +0.00753 |
| staircase | 0.60523 | 0.60702 | +0.00179 |
| uniform | 0.67186 | 0.67955 | +0.00769 |

因此最佳点的提升并非由牺牲某一种拓扑换取。

## PPO 稳定性

| 指标（600 updates） | 数值 |
|---|---:|
| KL mean | 0.02084 |
| KL median | 0.01695 |
| KL P90 | 0.04351 |
| KL P95 | 0.05212 |
| KL max | 0.08234 |
| KL > 0.03 | 135/600 |
| KL > 0.05 | 34/600 |
| clip fraction mean | 0.20732 |
| clip fraction max | 0.79688 |
| normalized entropy min | 0.62486 |
| normalized entropy at update 600 | 0.75510 |

熵没有坍缩、硬约束始终满足，但 KL 和 clip fraction 存在较重尾部，说明联合更新仍偏激进。
这不否定当前最佳模型，但在继续训练和多 seed 实验时必须保留监控，不能只看 episode return。

## 性能与资源

- 平均训练吞吐 317.16 transitions/s，中位数 353.55 transitions/s；
- 日志中累计 update 耗时约 1655.10 s，其中 rollout 约 1153.45 s、PPO 更新约 453.26 s；
- 逐实例验证累计 runtime 约 1250.95 s；
- 峰值 allocated 显存约 1099.81 MB，峰值 reserved 显存约 5144 MB。

恢复训练后 `run_summary.json` 的 `training_elapsed_sec` 只表示最后一次 resume 段；科研统计应
优先使用 `train_updates.csv` 和 `validation_instances.csv` 中的逐项耗时，不把该字段误当作
从 update 1 开始的完整墙钟时间。

## 学习率受控分支

从 update 150 `best.pt` 建立的 `lr=2e-4` 分支运行至 update 225，训练实例和场景序列与
原 `3e-4` 分支 300/300 条完全一致。该分支 update 225 验证分数为 0.63465，低于原分支的
0.65783；75 updates 的平均 KL 也没有实质改善。因此该分支不继续训练，保留为调参负结果。

## 文件使用约定

- 主模型：`results/formal_joint_seed0_run01/checkpoints/best.pt`，update 450；
- 恢复轨迹：`results/formal_joint_seed0_run01/checkpoints/latest.pt`，update 600；
- 完整训练曲线：`results/formal_joint_seed0_run01/metrics/train_updates.csv`；
- 固定验证曲线：`results/formal_joint_seed0_run01/metrics/validation_updates.csv`；
- 逐实例配对分析：`results/formal_joint_seed0_run01/metrics/validation_instances.csv`；
- 未通过门禁的分支：`results/formal_joint_seed0_lr2e4_gate`。

当前只完成 seed0 阶段训练，不能据此报告多随机种子均值或最终测试集结论。测试场景库仍应
保持冻结，待训练更新数和 PPO 稳定性方案确定后再统一运行。
