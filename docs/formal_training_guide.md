# 正式训练操作与实验约定

## 已固定的第一版研究设定

- 共享模型覆盖五种拓扑、不同内容实例及连续扰动后的学生/教师偏好。
- 目标权重固定为教学效果 0.30、学生偏好 0.25、教师偏好 0.25、方法分布 0.10、软约束 0.10。
- 目标方法分布固定为 0.18/0.12/0.10/0.10/0.10/0.08/0.08/0.08/0.05/0.05/0.06。
- 学生和教师各六类教学画像；训练使用 30 个模板组合，另外 6 个组合只用于组合外泛化测试。画像按标准差 0.05、最大绝对扰动 0.10 连续采样。
- manual_bottleneck 只作为输入目录名，实验元数据统一写成 bottleneck。
- 当前不加入教学效果下限。主算法为 Effect-only greedy 初始化后的 Masked PPO；局部搜索作为强基线，PPO+局部搜索不列为第一版主模型。

## 数据准备

全部五拓扑转换命令如下。输出按 topology/group/instance 分层，因此跨拓扑同名文件不会覆盖。

    .\.venv\Scripts\python.exe -m pa_moap_rl.data.convert_pre_data --instance-root pre_data/instance --selection-root pre_data/x --output-root data_processed/base_instances

然后构造 70/15/15 划分和固定场景库：

    .\.venv\Scripts\python.exe -m pa_moap_rl.data.build_dataset --manifest data_processed/base_instances/manifest.csv --output-dir data_processed/splits --seed 20260906

划分以 group + 原始随机种子为单位，同一个底层实例的五种拓扑变体一定进入同一个 split。检查 data_processed/splits/dataset_summary.json 中的 leakage_check、行数和各拓扑计数。

## 训练前最小验收

    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe -m pa_moap_rl.experiments.formal_training --smoke-test --device auto --output-dir results/formal_ppo_smoke

验收条件：测试全部通过；smoke 生成 train_log.csv、validation_log.csv、checkpoints/latest.pt、checkpoints/best.pt，且日志中非法动作数为零。

## 正式训练与恢复

以下参数已经通过短程 pilot 与 CUDA 容量探针验证。依据见 pilot_report_20260906.md。

    .\.venv\Scripts\python.exe -m pa_moap_rl.experiments.formal_training --device auto --seed 0 --total-updates 1500 --instances-per-update 4 --rollout-steps 128 --minibatch-size 64 --update-epochs 4 --learning-rate 0.0003 --entropy-coef 0.01 --hidden-dim 64 --eval-interval 25 --eval-limit 90 --checkpoint-interval 25 --rollout-mode joint --output-dir results/formal_ppo_run01

断点恢复时，total-updates 表示最终累计 update 数，不是新增数：

    .\.venv\Scripts\python.exe -m pa_moap_rl.experiments.formal_training --resume-from results/formal_ppo_run01/checkpoints/latest.pt --total-updates 1500 --output-dir results/formal_ppo_run01

checkpoint 同时保存模型、优化器、Python/NumPy/PyTorch/CUDA 随机状态、历史日志和最佳验证分数。小批量和大批量使用同一流程；instances-per-update 控制每次更新抽取的实例数，minibatch-size 控制 PPO 优化批量，两者不是同一概念。

`rollout-mode=joint` 是当前默认和推荐模式：4 个同规模桶实例同步采样，变长节点维先 padding，再进行一次批量策略推理；4 条轨迹分别计算并标准化 GAE，随后合并为 512 条 transition 做一次联合 PPO 更新。训练批次优先保证内容实例不重复、拓扑多样。`rollout-mode=sequential` 只用于受控回归比较，不作为正式训练默认值。

改造验收报告见 `vectorized_joint_ppo_report.md`。从同一 update 400 起点、使用完全一致的 100-update 实例与场景序列时，joint 的训练吞吐是 sequential 的 2.04 倍，update 500 配对验证平均提高 0.00667；但 joint 的 KL 更高，因此这只是工程采用依据，不能替代从零开始、多随机种子的最终实验。

首个从零训练的 joint seed0 已完成至 update 1000，最佳 checkpoint 位于 update 725，固定验证分数为 0.66743。update 600 中间报告见 `formal_joint_seed0_u600_report.md`，当前阶段结论和 `best.pt`/`latest.pt` 使用约定见 `formal_joint_seed0_u1000_report.md`。

## 独立测试与对比

插值场景测试：

    .\.venv\Scripts\python.exe -m pa_moap_rl.experiments.evaluate_checkpoint --checkpoint results/formal_ppo_run01/checkpoints/best.pt --scenario-bank data_processed/splits/scenarios/test_interpolation.json --output-csv results/formal_ppo_run01/test_interpolation.csv

未见画像组合测试：

    .\.venv\Scripts\python.exe -m pa_moap_rl.experiments.evaluate_checkpoint --checkpoint results/formal_ppo_run01/checkpoints/best.pt --scenario-bank data_processed/splits/scenarios/test_heldout.json --output-csv results/formal_ppo_run01/test_heldout.csv

独立评估默认同时运行随机可行、Effect-only greedy 和标量独立 greedy。追加 --include-local-search 才运行耗时更高的增量局部搜索强基线；调试时可用 --limit，也可用 --skip-baselines 只检查 PPO。

## 正式长跑中的监控与复核

- 在 update 300、600、1000 检查验证趋势、标准化策略熵和 approx KL。
- joint 模式短程门禁中 KL 的中位数为 0.0173，22/100 次超过 0.03，最大值 0.0663；正式长跑除里程碑外还应实时观察 KL 和 clip fraction。若出现连续异常而非孤立尖峰，应暂停并先复核学习率或 PPO 更新轮数，不要仅因单次尖峰删除运行。
- 当前验证配置为每 25 updates 评估 90 个跨 group/topology 实例。
- 局部搜索的迭代上限、随机重启次数及是否只在测试子集运行。
- 是否增加多个训练随机种子。论文主结果建议至少三个独立种子，报告均值和标准差。

不要用测试集挑超参数或选择 checkpoint。best.pt 只根据固定验证集和固定插值场景库更新；插值测试与留出画像组合测试只在方案冻结后运行。
