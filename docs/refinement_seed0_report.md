# Seed 0：update 400 受控精炼分支结果

## 设计

两个分支均从 results/formal_ppo_run01/checkpoints/best.pt（update 400，
validation J=0.6590128105）恢复，使用相同随机状态、训练实例、固定验证集、每次 90 个
验证实例和 update 600 终点。

- control：learning_rate=3e-4，entropy_coef=0.01（原轨迹）；
- entropy003：只把 entropy_coef 改为 0.003；
- lr1e4：只把 learning_rate 改为 1e-4。

## 验证结果

| update | control | entropy003 | lr1e4 |
|---:|---:|---:|---:|
| 400 | 0.659013 | 0.659013 | 0.659013 |
| 425 | 0.656799 | 0.650123 | 0.658791 |
| 450 | 0.658482 | 0.657128 | 0.658260 |
| 475 | 0.650600 | 0.651769 | 0.656508 |
| 500 | 0.658558 | 0.650080 | 0.656573 |
| 525 | 0.652419 | 0.650309 | 0.655499 |
| 550 | 0.658613 | 0.652028 | 0.650954 |
| 575 | 0.657260 | 0.653279 | 0.653602 |
| 600 | 0.656483 | 0.658009 | 0.652177 |

update 425–600 的验证均值分别为 control 0.656152、entropy003 0.652840、
lr1e4 0.655295。三个轨迹都没有超过共同起点 0.659013，因此两个新分支的 best.pt
均保留为 update 400，且已逐张量核对与源模型一致。

低学习率分支的平均 KL 为 0.002130，低于 control 的 0.007690 和 entropy003 的
0.005179，说明更新更保守；但更小的漂移没有产生更优验证分数。entropy003 在 update
600 达到 0.658009，比同点 control 高 0.001526，并在 hourglass、random、staircase、
uniform 四类拓扑上高于同点 control，但它在完整八个验证点上的均值更低。

三个轨迹在全部验证实例上的 hard_feasible_rate 都为 1，mask violation 总数都为 0。
两个新分支 stderr 均为空，分别耗时约 1014 秒和 983 秒。

## 当前结论

不把任一常数超参数分支替换为新的主模型；当前模型选择仍为 update 400 best.pt。
降低学习率显著抑制 KL，但只带来更平滑而非更优的解；降低熵系数的末点恢复值得保留
为后续调度型实验的依据，但单次 seed 不能证明优势。若继续精炼，更合理的下一轮是
验证从 update 400 开始进行学习率/熵系数退火或早停，而不是继续延长这两个常数分支。

完整数据和 TensorBoard event 分别保存在：

- results/formal_ppo_refine_entropy003_seed0
- results/formal_ppo_refine_lr1e4_seed0
