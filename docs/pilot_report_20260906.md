# Masked PPO 短程 pilot 报告（2026-09-06）

## 运行配置

- 主 pilot：seed 0，40 updates，每次 2 个实例，每实例 64 transitions，minibatch 64，4 epochs。
- 学习率 3e-4，熵系数 0.01，hidden dim 64。
- CUDA 设备：RTX 4060 Laptop GPU。
- 训练场景：30 个非留出学生-教师模板组合内连续扰动。
- 验证只使用 validation split，没有使用 test split。

## 训练资源与稳定性

| 指标 | 主 pilot | 4实例×128 容量探针 |
|---|---:|---:|
| 平均 transitions/s | 169.1 | 197.7 |
| 中位 transitions/s | 184.1 | 200 左右 |
| 平均 update 训练时间 | 0.85 s | 2.61 s |
| CUDA 峰值 allocated | 1010 MB | 993 MB |
| CUDA 峰值 reserved | 2996 MB | 1906 MB |
| 非法动作 | 0 | 0 |
| 平均 approx KL | 0.0051 | 0.0026 |

主 pilot 的标准化动作熵从前 5 次更新均值 0.9999 降至最后 5 次的
0.6974，最低 0.6851。策略已经开始形成偏好，但没有出现探索完全塌缩。

## 修正后的跨五拓扑验证曲线

原始 limit 逻辑会取 manifest 前若干行，导致首次 pilot 日志的 30 个验证
实例全部来自 hourglass。训练抽样不受影响，但原 validation_log 不用于最终
判断。该问题已修复为按 group、topology 轮转抽样，并对现有 checkpoint
使用完全相同的 45 个实例重新评估。

| checkpoint | 平均 J | 标准差 |
|---|---:|---:|
| update 10 | 0.6328 | 0.0449 |
| update 20 | 0.6373 | 0.0414 |
| update 25 / 当时 best | 0.6508 | 0.0388 |
| update 30 | 0.6509 | 0.0388 |
| update 40 | 0.6553 | 0.0375 |

截至 update 40 仍有改善，40 updates 明显不足以视为收敛。

## 与基线的关系

在修正后的 45 个验证实例上，当时 best checkpoint 的平均 J 为 0.6508：

- Effect-only greedy：0.6240，PPO 平均高 0.0268，逐实例胜率 100%。
- Random legal：0.6062，PPO 平均高 0.0447，逐实例胜率 97.8%。
- Scalarized independent greedy：0.6582，PPO 平均低 0.0073，胜率 33.3%。

update 40 的 PPO 均值已达到 0.6553，与 scalarized independent greedy 的
平均差距缩小到约 0.0029。短训练说明 PPO 确实在学习全局分布与软约束，
但尚不能据此宣称超过最强贪婪基线。

分拓扑看，staircase 的绝对 J 最低，但 PPO 相对 Effect-only greedy 的提升
最大。各拓扑上的 PPO 都超过 Effect-only greedy；相对 scalarized greedy
的差距介于约 0.0027 到 0.0114。

## 下一次正式长训练参数

第一条正式长跑建议固定如下：

    --total-updates 1500
    --instances-per-update 4
    --rollout-steps 128
    --minibatch-size 64
    --update-epochs 4
    --learning-rate 0.0003
    --entropy-coef 0.01
    --hidden-dim 64
    --eval-interval 25
    --eval-limit 90
    --checkpoint-interval 25

依据如下：

- 4实例×128 配置已通过 CUDA 容量探针，吞吐约 198 transitions/s，显存余量充足。
- KL 很低且稳定，没有证据要求降低学习率。
- 标准化熵仍约 0.70，没有证据要求提高熵系数。
- 验证分数在 update 40 仍上升，因此将第一条长跑目标设为 1500，并保留断点续训。
- eval limit 90 可使 9 个规模组×5 个拓扑各有 2 个实例，并让 30 个插值画像组合循环 3 次。

预计单个 seed 约需 2 小时上下，实际取决于 Windows 后台负载和验证耗时。
先运行 seed 0；在 update 300、600、1000 检查曲线，若健康则完成到 1500，
再以相同配置运行 seed 1 和 seed 2。

## 中途判据

- 非法动作必须持续为 0。
- 平均 approx KL 建议保持在 0.03 以下；若持续超过，学习率降到 1e-4。
- 标准化熵若在验证尚未提升时持续低于 0.35，可将熵系数提高到 0.02 后开新实验，不应直接污染当前 run。
- 若验证分数连续 200 updates 无提升，可停止该 seed 并比较 best checkpoint。
- 不使用 test interpolation 或 test held-out 选择超参数。两类 test 只在训练方案冻结后运行。

本次 pilot 的 best.pt 是按修复前的单拓扑限量验证选出的，只用于诊断；
正式长跑将使用已经修复的跨 topology 验证抽样重新选择 best.pt。
