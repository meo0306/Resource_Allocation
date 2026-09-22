# Seed 0 正式训练里程碑报告

## 执行状态

- 已按固定配置完成 update 1–1000。
- update 300、600、1000 均执行了独立健康检查。
- 因 update 1000 未通过正常续训条件，未自动继续到 1500。
- 全程非法动作数为 0，三个阶段 stderr 均为空。
- test interpolation 和 test held-out 均未用于训练或选模。

## 里程碑验证结果

| update | validation 平均 J | 判断 |
|---:|---:|---|
| 300 | 0.657319 | 新高，训练健康 |
| 400 | 0.659013 | 全程最佳 |
| 600 | 0.656483 | 有波动但仍健康 |
| 1000 | 0.637182 | 明显性能漂移，不应从 latest 原样续训 |

best.pt 对应 update 400，latest.pt 对应 update 1000。选模必须使用 best.pt。

## 熵和 KL 诊断

- update 376–400：标准化熵均值 0.612，KL 均值 0.0066。
- update 601–700：标准化熵均值 0.614，KL 均值 0.0073。
- update 901–975：标准化熵均值 0.737，KL 均值 0.0144。
- update 976–1000：标准化熵均值 0.711，KL 均值 0.0143。

后期熵重新升高且五个拓扑同步下降，表明固定 entropy coef 0.01 和
learning rate 3e-4 使策略在找到较优确定性解后继续扩散。没有发现数据错误、
非法动作、梯度爆炸或单拓扑异常。

## update 400 best 与基础基线

统一使用 90 个 validation 实例和固定插值画像场景：

| solver | 平均 J | 标准差 |
|---|---:|---:|
| Masked PPO best | 0.659013 | 0.035611 |
| Scalarized independent greedy | 0.656873 | 0.035347 |
| Effect-only greedy | 0.621475 | 0.050836 |
| Random legal | 0.603925 | 0.031328 |

- PPO 相对 Scalarized independent greedy 平均 +0.002140，逐实例胜率 62.2%。
- PPO 相对 Effect-only greedy 平均 +0.037538，逐实例胜率 100%。
- PPO 相对 Random legal 平均 +0.055088，逐实例胜率 100%。

分拓扑相对 Scalarized independent greedy：

| topology | PPO - scalarized | PPO 胜率 |
|---|---:|---:|
| bottleneck | +0.006834 | 94.4% |
| staircase | +0.008132 | 94.4% |
| uniform | -0.000083 | 44.4% |
| hourglass | -0.000963 | 44.4% |
| random | -0.003222 | 33.3% |

## 后续建议

不建议从 latest.pt 按原参数继续到 1500。推荐从 update 400 best.pt
创建独立精炼分支，并通过受控实验降低后期策略扩散：

1. 只降低 entropy coef：0.01 → 0.003，学习率保持 3e-4。
2. 只降低 learning rate：3e-4 → 1e-4，entropy coef 保持 0.01。
3. 两个分支各从 update 400 训练 150–300 updates，使用同一 validation 比较。
4. 选择更好的分支后再决定是否续至累计 update 1000 或 1500。

为了保证学习率分支真实生效，恢复 checkpoint 后需要显式把 CLI 指定的
learning rate 写回 optimizer parameter groups；当前 PyTorch optimizer
state 恢复会带回 checkpoint 中的旧学习率。
