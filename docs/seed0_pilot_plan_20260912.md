# 四拓扑 b075 新版 seed0 短程 pilot 方案

状态：已确认；首轮与稳定化运行均已完成。该方案只验证训练、监控和恢复链，并为后续 PPO/推理设置调整提供证据；不构成正式多种子结论，也不冻结 PPO 超参数。

## 固定输入

- 目标：冻结的 b075，语义哈希 `b075006e79dc73a06ce76049ac0e3350c2d6b9b3ceb79db6fec57294e5b51606`。
- 训练池：四拓扑 `dev_train`，2,520 个实例、630 个源家族。
- 训练画像：Scenario v2 的非 held-out 插值组合，每次采样扰动种子并保存完整画像与哈希。
- 固定 validation：`calibration_core_72` 的九规模组 × 两源家族 × 四拓扑，沿用 `screen_pairs_72` 的受控画像，并严格连接 b075 已证最优结果。冲突画像虽带 `held_out=true`，但访问状态是已查看的 `controlled_calibration`，只通过显式 opt-in 用于开发 validation，绝不解释为独立测试。
- 数据仍属于探索性开发数据；不把 `dev_seen_diagnostic` 当作独立测试，也不生成新底层内容。

## 起始训练设置

共享一个策略覆盖四拓扑、九规模组和多偏好画像。seed=0，joint B=4，rollout=128，100 updates，epochs=4，minibatch=64，学习率 `3e-4`，策略熵系数 `0.01`，hidden=64，环境最大 128 步、patience=32，target-KL 关闭。使用 RTX 4060 Laptop GPU 和确定性算法。

这些值只是从旧版经验出发的 pilot 起点，不是新版冻结参数。validation 在 update 0、10、20、…、100 执行；每 10 updates 保存编号 checkpoint 和 latest，best 以 validation 平均 J 为主、P90 精确相对 gap 为并列判据。

## 监控与停止

逐实例和逐 update 保存 J、F_E、F_S、F_T、软惩罚、策略熵、方案方法分布熵、KL、clip fraction、梯度、可行性、非法动作、吞吐、GPU 峰值显存和耗时；固定 validation 另保存绝对/相对精确 gap，并按拓扑分层。

以下情况立即停止并修复，不把结果用于选参：目标/方法矩阵/数据/scenario/精确结果哈希不一致，评分重算误差超过 `1e-9`，出现 NaN/Inf，硬约束可行性低于 100%，非法动作不为零，或 checkpoint 不能严格恢复。

完成 100 updates 且恢复链、数值与可行性均通过，视为工程 pilot 成功。若 validation 最优均值或 P90 gap 相对 update 0 有改善且训练指标稳定，可进入偏好响应和搜索预算诊断；若链路稳定但无改善，先调整学习率、rollout、epochs、minibatch 或 target-KL；若重复出现数值、可行性或恢复错误，则停止 PPO 扩展。1%/2%/3%/5% gap 只作为计算评价档位，不作为未经验证的教学阈值。

## 运行产物

正式有效运行独立输出到 `results/four_topology_b075_seed0_pilot_v1_run02/`，包含 run metadata/summary、逐实例与逐 update CSV、TensorBoard、checkpoint 索引、best/latest/编号 checkpoint。不得覆盖任何旧版 seed0 或校准结果。首次启动目录 `results/four_topology_b075_seed0_pilot_v1/` 因发现新版 manifest 的 `size_group` 未被旧采样字段识别而在 update 10 终止，标记为失败预检，不得参与分析或恢复。

## 首轮结果与下一决策

run02 已完成 100 updates，工程、数值、可行性和恢复链全部通过。best 为 update 50：平均 J=0.764410，平均精确 gap=0.654%，P90 gap=1.160%；latest 为 update 100：平均 J=0.712049，平均 gap=7.449%，P90 gap=11.069%。update 70 后出现离散策略退化，且未自行恢复，因此首轮 PPO 起点不稳定，不得冻结。

审计和决策报告位于 `results/four_topology_b075_seed0_pilot_v1_run02/pilot_decision_report.md`。建议的下一步是从头运行一个独立稳定化候选，优先降低学习率并启用 target-KL；具体参数与是否启动须另行确认。冻结目标 b075 保持不变。

用户随后确认稳定化 run03：学习率 `1e-4`、`target_kl=0.03`，其余设置不变。run03 完成 100 updates，best update 70 的平均 J=0.765279、平均 gap=0.541%、P90 gap=1.114%；latest update 100 平均 gap=0.542%，未发生实质性后程退化。工程与 provenance 审计全部通过。run03 满足进入偏好响应和搜索预算诊断的稳定性条件，但仍不冻结 PPO 参数。
