# 训练监控与科研数据说明

正式训练默认在输出目录下开启 TensorBoard，并同时保存机器可读的完整训练数据。
每个实验必须使用独立的 output-dir；不要让两个进程写入同一目录。

## 实时查看

训练启动后，在另一个 PowerShell 窗口运行：

    .\.venv\Scripts\tensorboard.exe --logdir results --port 6006

然后访问 http://localhost:6006 。如果只查看单次实验，可把 logdir 指向该实验的
tensorboard 子目录。正式训练命令可用 --tensorboard-dir 指定其他位置，或用
--disable-tensorboard 关闭。

建议重点关注：

- validation/total_score_mean、validation/best_total_score_mean：模型选择主指标；
- validation_by_topology/*：五类拓扑是否同步改善；
- train/normalized_entropy、train/approx_kl、train/clip_fraction：策略探索和更新幅度；
- train/policy_loss、train/value_loss、train/explained_variance、train/grad_norm：优化稳定性；
- system/transitions_per_sec、system/gpu_peak_*：吞吐和显存；
- system/rollout_elapsed_sec、system/ppo_elapsed_sec、system/optimizer_minibatches：联合采样与 PPO 两部分耗时；
- train/illegal_action_rate、validation/hard_feasible_rate_mean：约束正确性。

TensorBoard 用于及时诊断，论文绘图应优先读取 CSV，不直接把 event 文件作为唯一数据源。

## 输出目录

每次运行形成以下结构：

    output_dir/
      run_metadata.json
      run_summary.json
      checkpoint_index.csv
      train_log.csv
      validation_log.csv
      tensorboard/
      metrics/
        train_instances.csv
        train_updates.csv
        validation_instances.csv
        validation_updates.csv
      checkpoints/
        best.pt
        latest.pt
        update_*.pt

- train_instances.csv：每个 update 抽到的每个实例及其精确画像、目标分布、权重、
  分数分解、回报和 PPO 诊断；
- train_updates.csv：每个 update 一行，适合直接画训练曲线；
- validation_instances.csv：每次验证的逐实例完整结果；
- validation_updates.csv：按 overall 和 topology 聚合的均值、标准差、最小值和最大值；
- run_metadata.json：参数、软件/硬件环境、输入文件绝对路径及 SHA-256；
- run_summary.json：当前状态、最近/最佳验证分数、行数和耗时；
- checkpoint_index.csv：继承起点、刷新 best 和定期 checkpoint 的索引。

CSV 和 JSON 通过原子替换写出，训练过程中可由只读分析程序安全读取。恢复训练时，
历史行继续保留；若从另一个目录的 best.pt 开分支，新目录会先保存 inherited_best，
因此即使后续没有改善也不会丢失分支起点。

## 科研绘图约定

训练曲线以 update 为横轴。主结果优先使用 validation_updates.csv 中
scope=overall 的 total_score_mean；拓扑鲁棒性使用 scope=topology。训练稳定性使用
train_updates.csv。逐实例显著性检验、置信区间和配对比较使用 validation_instances.csv，
并按 instance_path、scenario_id 和 update 配对。

调参只使用固定验证集，冻结方案后再运行测试集。图表和统计结果应记录 output-dir、
best checkpoint 对应 update、输入哈希和随机种子。不要手工改写原始 CSV；派生数据放在
新的 analysis 子目录。

联合采样实验还应保留 `rollout_mode`、`batch_group`、`batch_max_n`、
`unique_sources`、`unique_topologies` 和 `transitions` 字段。比较 joint 与 sequential 时，
先按 update 核对 `train_instances.csv` 中的 instance_path 和 scenario_id 完全一致；否则只能
视为非受控运行，不能把分数差直接归因于采样/更新方式。
