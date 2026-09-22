# 四拓扑新版研究后续任务清单

更新日期：2026-09-12

本清单依据 `docs/research_requirements_and_plan_20260912.md`、四拓扑目标校准结果及用户确认形成。后续默认按以下依赖顺序执行；任何会改变研究口径、冻结参数、数据独立性或正式验收标准的调整，均须另行确认。

## 已完成的前置工作

- 统一 ObjectiveSpec、完整/增量评分、求解器、环境、checkpoint 与恢复校验链。
- 建立四拓扑探索性数据协议及访问登记。
- 完成候选扫描、受控画像复核、完整 calibration 复核和 Pareto 报告。
- 用户确认冻结候选 b075：`main_no_reference__h0.65__c0.35__be1.00__bc1.00`。

## 后续执行顺序

### 1. 持久化冻结目标（已完成）

- 写入独立的四拓扑 v2 冻结目标配置。
- 保存用户确认、目标哈希、校准配置、数据审计和决策报告哈希。
- 增加冻结配置加载及哈希一致性测试。
- 不修改 legacy 默认配置，不把候选扫描 catalogue 改写为冻结配置。
- 后续新版 pilot、训练和评估必须显式传入冻结配置。

验收点：冻结配置可由 `load_objective_spec` 直接读取，语义哈希严格等于 `b075006e79dc73a06ce76049ac0e3350c2d6b9b3ceb79db6fec57294e5b51606`。

### 2. 制定 seed0 pilot 方案（已完成）

- 明确训练画像采样、validation 配对、更新预算、停止条件和资源预算。
- 旧 PPO 超参数仅作为起点，不视为新版已冻结参数。
- 明确一个共享模型覆盖四拓扑、九规模组和多偏好画像。
- 复用已有 b075 精确结果作为固定 validation 参考。
- 明确 pilot 成功、调整或停止的判据。

验收点：先提交 pilot 计划并获得确认，再启动任何新版 PPO 训练。

执行记录：方案已固化于 `docs/seed0_pilot_plan_20260912.md` 和 `pa_moap_rl/configs/seed0_pilot_four_topology_v2.yaml`。其中 PPO 参数明确标记为 pilot 起点而非冻结参数。

### 3. 运行新版 seed0 短程 pilot（已完成）

- 使用冻结目标、`dev_train` 和 Scenario v2。
- 验证目标、方法矩阵、数据和场景哈希，以及 checkpoint 恢复链。
- 监控 J、F_E、F_S、F_T、两类软惩罚、策略熵、方法分布熵、KL、clip fraction、可行性、非法动作、吞吐和显存。
- 在固定精确参考子集上报告相对 gap。

验收点：证明训练与恢复链稳定，并产出足够信息决定是否调整 PPO 或搜索设置；本阶段结果不是正式多种子结论。

执行记录：有效运行目录为 `results/four_topology_b075_seed0_pilot_v1_run02/`，已完成 100 个 update。工程、数据、目标与 checkpoint 恢复链通过；best 为 update 50，latest 为 update 100。update 70 后固定 validation 明显退化，因此旧 PPO 起点参数不得冻结，进入方案1前应先决定是否执行稳定化调整 pilot。`results/four_topology_b075_seed0_pilot_v1/` 是因规模组字段读取错误而中止的无效 run01，只保留 `ABORTED.md`，不得恢复或用于结论。

验收报告：`results/four_topology_b075_seed0_pilot_v1_run02/pilot_decision_report.md`；机器可读审计为同目录 `pilot_audit_summary.json` 和 `checkpoint_audit.json`。72 对固定 validation 的画像与拓扑完全混杂，仅用于工程和质量监控；不得从中独立解释画像响应，方案1必须使用 72×4 全交叉设计。

稳定化执行：用户已确认独立 run03。从头使用 seed0 和同一固定输入，仅将学习率由 `3e-4` 降至 `1e-4`、启用 `target_kl=0.03`，其余设置不变；配置为 `pa_moap_rl/configs/seed0_pilot_four_topology_v2_stabilization_run03.yaml`，输出目录为 `results/four_topology_b075_seed0_pilot_stabilization_run03/`。run03 仍是探索性 pilot，不冻结 PPO 参数。

稳定化结果：run03 已完成 100 updates，工程与 checkpoint 审计通过。best 为 update 70，平均 J=0.765279、平均 gap=0.541%、P90 gap=1.114%；latest update 100 平均 gap=0.542%，未重现 run02 的后程崩塌。run02/run03 对照为 `results/four_topology_b075_seed0_pilot_stabilization_run03/run02_run03_comparison.md`。下一步可进入方案1和方案2，但 PPO 参数仍未冻结。

### 4. 实施方案1：双主体偏好响应（已完成）

- 比较一致、仅学生变化、仅教师变化和双方冲突画像。
- 比较 Gurobi、标量 greedy、单点 best-improvement local search 和 PPO。
- 所有结果按真实输入画像评分。
- 报告 F_E、F_S、F_T、F_P、软惩罚、旧解复用变化和重新求解增益。
- 不以 assignment 是否变化单独判断偏好响应，不把最优解非唯一误判为模型失效。

验收点：确认 PPO 是否实质利用偏好输入，并区分精确模型响应与 PPO 近似误差。

执行记录：用户已确认启动。实验复用 b075 的 72×4 已证最优 `controlled_cross` 结果，不重新求解 Gurobi；PPO 使用稳定化 run03 的 best checkpoint（update 70）。S6–T6 作为每个内容实例的旧解基线，在其余画像下按真实输入画像重算旧解并比较重求解增益。配置为 `pa_moap_rl/configs/preference_response_four_topology_v2.yaml`，独立输出至 `results/four_topology_b075_preference_response_v1/`；不训练 PPO、不冻结 PPO 参数。

完成记录：288 个实例—画像对 × 4 方法共 1,152 条结果，目标、精确结果、checkpoint、可行性和输出审计全部通过。PPO 在三个非基线画像合并后的平均重求解增益为 0.000241，源家族聚类 bootstrap 95% CI 为 [0.000167, 0.000310]，但响应不完整：student-only 仅 44.4% 实例有正增益，teacher-only 在四拓扑全部零 assignment 变化、零重求解增益，conflict 为 79.2% 正增益。精确模型在 teacher-only 的正增益率为 95.8%，所以不能以“目标本身没有响应机会”解释 PPO 零响应。策略输入链已只读核验，student/teacher preference 均作为独立特征进入 actor，因此当前结论是 seed0 策略仅表现出部分偏好响应，尚未证明完整双主体响应。报告为 `results/four_topology_b075_preference_response_v1/preference_response_report.md`，机器摘要为同目录 `preference_response_summary.json`。

后续约束：PPO 与推理参数仍不得冻结。第5项应特别检验提高搜索步数或 patience 能否解除 teacher-only 与 group9 的零响应；若不同预算下仍为零，则应回到训练画像覆盖、损失权衡或网络学习诊断，而不是把 assignment 变化本身当作成功标准。

### 5. 实施方案2：搜索预算与质量诊断（已完成）

- 比较 128、256、512 步及随 N 缩放的推理预算。
- 单独控制 patience，记录 accepted moves、重复动作、覆盖节点数和 best-so-far。
- 输出 anytime 曲线及到精确解/上界的有界 gap。
- 按拓扑、规模和画像分层判断 128 步是否构成实际瓶颈。

验收点：形成搜索步数和 patience 的候选设置，不预设增加预算必然改善。

执行记录：用户确认使用当前锚点 128/32、固定预算 128/256/512、128×ceil(N/128) 的 N 缩放预算，以及固定 512 步下 32/64/128/256/512 的 patience 扫描。固定预算和 N 缩放扫描令 patience 等于预算，以隔离 max_steps；所有条件均由每个实例—画像的一条最长确定性母轨迹按原环境停止规则精确截断复算。实验固定 b075 目标和 run03 best checkpoint（update 70），共完成 288 条母轨迹和 2,880 个条件结果，不训练 PPO。配置为 pa_moap_rl/configs/search_budget_diagnostic_four_topology_v2.yaml，结果为 results/four_topology_b075_search_budget_diagnostic_v1/。

完成记录：所有结果硬可行，mask violation 为 0，最大评分重算误差为 0。当前 128/32 的平均 gap/P90 gap 为 0.563%/1.187%，1% 内比例 85.42%；512/32 为 0.443%/0.766% 和 93.75%，平均推理时间由 0.690s 增至 0.874s。512/32 与 512/512 在全部 288 对上质量相同，而后者平均耗时 4.215s；N 缩放到最高 768 步也未超过固定 512。平均节点覆盖率仅由 22.27% 增至 22.86%，512/512 平均 512 个动作中有 397.8 个重复动作，因此提高 patience 或继续按 N 增长主要产生循环。

分层结论：128 步是 group8/group9 的实际瓶颈，group9 平均 gap 由 1.097% 降至 0.319%；group1–group6 不受益。延长预算使 student-only 正增益率由 44.4% 升至 63.9%、conflict 由 79.2% 升至 98.6%，并解除 group9 的 student-only/conflict 零响应；teacher-only 在所有预算和 patience 下仍为零 assignment 变化和零重求解增益，必须回到训练画像覆盖、损失权衡或策略学习诊断。

候选记录：第6项的主候选为 max_steps=512, patience=32，低时延参照为 128/32；不推荐提高 patience 或默认使用 N 缩放。两者均为候选，尚未冻结 PPO 或推理参数。报告为 results/four_topology_b075_search_budget_diagnostic_v1/search_budget_diagnostic_report.md，机器摘要为同目录 search_budget_summary.json。

### 6. 冻结 PPO 与推理设置（主线已冻结）

- 根据 seed0、偏好响应和搜索预算诊断，确定学习率、rollout、epochs、minibatch、更新数、搜索步数、patience 和 checkpoint 选择规则。
- 制定 1%/2%/3%/5% gap 计算评价档位。
- 明确这些档位不是未经验证的教学质量阈值。
- 不直接移植旧版 1500 updates、target-KL 或旧绝对分数门禁。

验收点：生成独立的新版 PPO/推理冻结记录和完整哈希。

执行状态（2026-09-18）：用户确认暂不冻结。现有候选仍保存在 `pa_moap_rl/configs/ppo_inference_freeze_candidate_four_topology_v2.yaml`，其状态必须保持 `awaiting_user_decision_not_frozen`；第7项和第8项不得据此启动。

冻结前复核将 run03 的 checkpoint 0–100 全部改用候选推理预算 512/32 重评，按“最大 mean total score、再最小 P90 exact gap、再选更早 update”选择 update60。update60 在固定72对上的 mean/P90 gap 为 0.423%/0.818%，在72×4受控画像上的 mean/P90 gap 为 0.442%/0.766%；但 teacher-only 仍为 0/72 正重求解增益和零 assignment 变化。因此 update60 只是诊断 checkpoint，不是通用冻结 checkpoint。

主线冻结决策（2026-09-21）：用户明确决定以第6项初始结果为标准冻结 PPO，把后续 6A 修复及 A/B/C/D/E 消融视为独立分支探索；该分支探索已经失败并关闭，不选用其中任何模型、损失或 checkpoint，也不改写其产物。此前“update60 只是诊断 checkpoint”和“第7/8项不得启动”的状态由本次用户决策显式取代。主线现冻结为 run03 原始 `legacy_separate_v1` 模型与训练协议，seed0 参考 checkpoint 为 update60（SHA256 `f3bc64e0a33d990c6612fbc9a3f33b48e29654dbf284ddb0e4b3a6640fed13df`），推理预算为确定性 512/32；每个后续 seed 仍按固定72对上的 mean J、P90 exact gap、较早 update 规则独立选择 checkpoint。权威配置为 `pa_moap_rl/configs/ppo_inference_frozen_four_topology_v2.yaml`，旧 candidate 文件保持历史只读状态，不再作为当前决策入口。

冻结解释边界：这是为了返回研究主线而作出的工程与实验协议冻结，不表示 teacher-only 零响应已经解决。现有证据仍是 0/72 正重求解增益和零 assignment 变化；后续报告必须将其列为已知局限。第7项批量服务效率实验由此解除冻结前置阻塞；第8项按原研究顺序在第7项之后进行。如果未来重新采用 6A 的 symmetric-v2 网络或其他结构，必须建立新版本并重新做效率实验，不能沿用本次冻结结论。

### 6A. 教师单独偏好零响应诊断与修复（分支探索已关闭，未采用）

用户指令：先根据当前模型和求解机制逐项排查，制定后续解决方案并记录全过程；不冻结 PPO/推理参数，不启动正式三种子训练。

排查链依次覆盖：Scenario v2 画像生成与应用、学生/教师偏好矩阵、冻结目标评分、环境 observation、编码器与 actor 特征、run03 训练画像覆盖、确定性 argmax、动作 mask、best-so-far、patience、搜索预算、真实单点收益排序及确定性局部改进。

已排除原因：

- 教师画像或 `teacher_pref` 没有传入模型。教师矩阵同时进入 NodeEncoder、候选动作绝对特征和 `delta_teacher` 特征。
- 目标对教师降权或置零。b075 中学生和教师系数均严格为 0.325。
- 训练完全没有覆盖 T4 或 S6–T4。run03 的400个训练画像中覆盖全部30个允许模板组合，S6–T4 有12条、S6–T6 有14条、S4–T6 有17条。
- teacher-only 没有可优化机会。精确模型在 95.83% 实例上有正重求解增益，平均增益 0.000223，高于 student-only 的 0.000136。
- 搜索步数或 patience 不足。第5项已验证提高到512步并扫描 patience 后 teacher-only 仍为零。
- 硬约束或动作 mask 阻断。PPO 平衡解附近每个实例都有教师目标正收益合法单点动作，正收益动作平均约占合法动作的 10.02%。
- 教师通道权重整体学成零。教师输入会改变 logits，且 actor 的 `teacher`/`delta_teacher` 直接权重范数非零。

已确认机制：

- update60 在效应贪心初始状态下，student-only 和 teacher-only 均未改变72个实例的首个 argmax；教师扰动虽使 logits 变化更大，但动作排序基本不变。
- 在平衡画像的 PPO best assignment 上按 teacher-only 重新评分，策略首选动作在72/72实例上均为负收益，平均真实收益为 -0.000152；同一状态平均仍有约208个正收益合法动作。
- 教师目标下真实最佳单点动作在策略排序中的平均名次约为1057，中位数631。零响应的直接原因是正收益教师动作被策略严重错排，而不是没有动作或预算不足。
- 训练每次更新使用不同内容的独立随机画像，没有同一内容上的平衡/仅学生变化/仅教师变化配对；已核实同一 update、同一内容的反事实画像对数量为0。当前证据支持“条件识别与排序监督不足”，但单次 seed0 不能把它提升为唯一因果解释。
- 当前 actor 接收未按 N 归一化的分项 delta 与软惩罚 delta，没有显式的冻结目标加权单步 `delta_J`；同时目标在学生/教师交换下等权对称，而网络使用未绑定的独立通道。这两项是需要消融验证的结构风险，不记为已经证明的唯一根因。
- 求解器只屏蔽不可行动作，不屏蔽负收益动作；best-so-far 只能保存旧解，不能纠正持续不变的错误 argmax。提高预算因而主要增加循环，无法产生教师响应。

无训练对照：从各画像的 PPO 输出继续执行确定性 one-point best-improvement。teacher-only 在 95.83% 实例上产生正重求解增益，平均增益 0.000223，平均 exact gap 为 0.000029%，结果与精确/原局部搜索证据一致。该结果证明当前邻域和评分机制能够恢复教师响应，但这是求解器级 hybrid workaround，不能当作“PPO 本身已学习教师条件”的证据。

诊断配置为 `pa_moap_rl/configs/teacher_response_root_cause_diagnostic_v2.yaml`；脚本为 `pa_moap_rl/experiments/diagnose_teacher_response_v2.py`；逐实例结果、checkpoint 敏感性、机器摘要、报告和哈希位于 `results/four_topology_b075_teacher_response_diagnostic_v1/`。诊断没有训练、没有改写 checkpoint、没有冻结参数。

后续解决顺序：

1. 将“PPO + 确定性单点 best-improvement”登记为诊断 hybrid 基线，单独报告成本和增益，不替代纯 PPO 双主体响应验收。
2. 训练采样改为同一内容的反事实配对：至少同时包含 balanced、student-only、teacher-only，并保留随机插值画像；冻结目标和现有数据角色不变。
3. 构造目标一致的动作特征：显式加入按冻结系数、N 和软惩罚计算的单步 `delta_J`，并通过学生/教师偏好和 delta 的对称聚合或共享权重落实等权交换不变性；保留分主体分项供审计。
4. 只做有界 seed0 消融：A=仅配对采样，B=仅目标一致/对称特征，C=A+B；其余 PPO 参数先保持 run03 候选值，不同时改学习率、网络宽度或搜索预算。
5. 在启动消融前另行冻结消融预算、选择规则和教师响应验收门禁。消融后必须重跑72×4纯 PPO 响应、固定72对质量、512/32搜索诊断和学生/冲突画像回归；若纯 PPO 仍无教师响应，不回到第6项冻结。

当前停止点：解决方案已经形成，但尚未创建或启动修复训练。下一步必须先确认短程消融协议及验收门禁。

#### 6A-1. 纯 PPO 修复实现与 A/B/C 消融（协议已确认，未训练）

用户指令：进入纯 PPO 修复实现，先在新脚本中实现配对采样、目标一致特征和对称结构；所有涉及问题模型、特征、网络结构和恢复链的变化必须有独立版本，便于回溯、辨认和异常回退；提交 A/B/C 消融运行协议供确认，在确认前不得启动训练。

实现边界与版本：

- 旧网络和旧采样链未被原位覆盖，继续保留 `legacy_separate_v1` / `independent_random_v1` 回退路径。
- 问题/MDP 独立标记为 `single_replacement_masked_mdp_v1`；冻结目标继续使用 b075 及其既有哈希，不建立新目标版本。
- 配对采样为 `paired_counterfactual_v1`：每 4 个环境使用两个内容实例；同一 anchor 内容生成 S6–T6、S4–T6、S6–T4 三元组，未变化主体的完整 profile 逐值共享，第四个环境保留另一内容的独立随机画像。
- 旧/新 observation 特征分别为 `separate_student_teacher_features_v1` 和 `objective_consistent_symmetric_features_v1`。新特征显式加入按冻结目标、有效节点数 N、方法分布和软惩罚计算的单步 `delta_J`。
- 旧/新网络分别为 `masked_actor_critic_separate_channels_v1` 和 `masked_actor_critic_exchange_invariant_v1`。新网络只消费学生/教师 mean 与 absolute gap，使等权目标下交换学生和教师输入不改变 actor logits 或 critic value。
- 新模型包标记为 `objective_delta_symmetric_v2`；训练接入标记为 `formal_training_versioned_injection_v1`；带实验规格的 checkpoint payload 升为 v3。
- 完整 `experiment_spec` 写入 checkpoint 和运行元数据；恢复时 arm、协议文件哈希、目标、采样、特征、网络、模型包或训练接入任一不匹配即失败。无该规格的旧 checkpoint 不能静默进入新消融路径。

消融协议：A=仅配对采样；B=仅目标一致对称特征/网络；C=A+B。三者均从头训练 seed0，除消融因素外严格沿用 run03 的 100 updates、rollout 128、minibatch 64、4 epochs、学习率 `1e-4`、entropy `0.01`、target-KL `0.03`、hidden 64 和训练 128/32；validation 与后续响应复核统一使用 512/32。纯 PPO 结果不得追加 local search 或 hybrid 后处理。

实现文件：

- `pa_moap_rl/experiments/paired_preference_sampling_v2.py`
- `pa_moap_rl/models/preference_repair_v2.py`
- `pa_moap_rl/experiments/run_preference_repair_ablation_v2.py`
- `pa_moap_rl/configs/preference_repair_ablation_protocol_v2.yaml`
- `docs/preference_repair_ablation_protocol_20260918.md`
- `tests/test_preference_repair_v2.py`

已确认验收门槛：工程硬门槛为可行率 100%、非法动作 0、重算误差不超过 `1e-9` 和严格版本恢复；固定 72 对 mean/P90 exact gap 上限为 0.525%/1.020%；teacher-only 正增益率至少 50%、平均重求解增益至少 `0.000055` 且 source-family 聚类 bootstrap 95% CI 下界大于 0；student-only/conflict 正增益率回归保护为 54%/89%。这些只是当前探索数据的开发门槛，不是教学质量或独立确认阈值。

确认记录（2026-09-18）：用户已接受 A/B/C 参数、验收门槛与选择顺序，并允许执行非训练验证。YAML 状态已更新为 `user_confirmed`，但本轮没有启动研究训练。尚未创建 A/B/C 结果目录、尚未执行任何 A/B/C 研究 update；启动训练仍需用户单独下达指令。

非训练验证记录（2026-09-18）：两组相关回归测试共 47 项全部通过（22+25）；覆盖配对画像共享与复现、学生/教师交换不变性、显式 `delta_J` 与完整评分差、A/B/C 版本映射、未确认协议训练门禁、checkpoint payload v3 与实验规格错配拒绝，以及既有 PPO、checkpoint、评估和诊断链。`test_formal_foundation.py` 仅在 pytest 临时目录执行了极小型 PPO smoke，不属于研究训练且未生成研究 checkpoint。A/B/C 三条 `--validate-only` 均返回 `validated_no_training`，最终协议 SHA256 为 `5f5da34a335fa1ffc8eea642804eba3d5bcd1947cefa1547a9d7e466adf035f6`；差异格式检查通过，`results/four_topology_b075_pure_ppo_ablation_v1/` 不存在。

训练启动记录（2026-09-18 18:47 CST）：用户明确指令按已确认协议依次启动 A/B/C 消融训练。新增只负责顺序编排和失败即停的协调入口 `pa_moap_rl/experiments/run_preference_repair_ablation_sequence_v2.py`，版本为 `preference_repair_ablation_sequence_v1`；协调状态写入 `results/four_topology_b075_pure_ppo_ablation_v1/sequence_state.json`，stdout/stderr 和 launcher PID 写入同目录 `_sequence_logs/`。后台协调进程已经启动，状态文件记录实际 Python PID 33304、协议哈希 `5f5da34a335fa1ffc8eea642804eba3d5bcd1947cefa1547a9d7e466adf035f6`、当前 arm=A、顺序 A→B→C。A 已使用 CUDA 和确认参数进入研究训练，首次复核时已完成 update 4，非法动作数为 0；B/C 尚未启动。任一 arm 抛出异常时协调器会写入 failed 状态并停止，不继续后续 arm；正常完成 A 后才进入 B，完成 B 后才进入 C。

训练完成记录（2026-09-18 19:18 CST）：协调状态为 `completed`，A/B/C 均完成 100 updates，各有 400 条训练实例记录和 792 条固定 validation 记录，stderr 为空，协调进程已正常退出。A/B/C 训练耗时分别为 507.16s、569.18s、702.50s；按已确认的 mean J→P90 gap→较早 update 规则保存的 best checkpoint 分别来自 update 50、30、90。三个 best 的固定 72 对 mean J 均约为 0.766174，P90 exact relative gap 均约为 0.8183%；这只完成 checkpoint 选择，不能据此判断 teacher-only 修复是否成功。下一步必须按协议对三个 best checkpoint 执行 72×4 controlled-cross 纯 PPO 响应复核和验收门槛判断，不得直接冻结 B、C 或 PPO 设置。

Controlled-cross 复核记录（2026-09-18 20:14 CST）：已新增并通过版本感知评估链。`evaluate_freeze_candidate_response_v2.py` 现在兼容旧诊断路径和带 `experiment_spec` 的 checkpoint v3，分别校验每个 arm 的训练输入数据哈希、checkpoint 哈希、arm id、模型/特征/网络版本和 state_dict；三臂评估配置为 `preference_repair_response_arm_{A,B,C}_v2.yaml`，门槛汇总器为 `summarize_preference_repair_ablation_v2.py`，顺序协调器版本为 `preference_repair_controlled_cross_sequence_v1`。三臂 `--validate-only` 全部通过，相关回归测试 14 项通过；随后以固定 512/32 完成 A/B/C 各 288 对、合计 864 次纯 PPO 推理，没有训练、local search 或 hybrid 后处理，stderr 为空，所有结果硬可行、mask violation 为 0、最大分数重算误差为 0。

门槛结果：A/B/C 的固定 72 对 mean gap 均约 0.425%、P90 gap 均约 0.818%，student-only 正增益率均为 63.89%，conflict 均为 98.61%，这些门槛全部通过；但 teacher-only 正增益率均为 0%，平均增益分别为 0、约 `-1.54e-18`、0，source-family 聚类 bootstrap 下界不大于 0，教师正增益率、平均增益和 bootstrap 三项门槛全部失败。因此 `eligible_arms=[]`、`recommended_arm=null`，不得选择或冻结 A/B/C。assignment 审计显示 A 与 C 在 288/288 对上完全相同；B 与 A/C 在 206/288 对上相同，B 虽在 teacher-only 的少量实例产生 assignment 变化，但收益仅为数值零，属于等分替换而非教师偏好响应。结果位于 `results/four_topology_b075_pure_ppo_ablation_v1/controlled_cross_evaluation_v1/`，汇总 SHA256 为 `0ac69528cb286d63a272ac191c393130cdf46cc699d9a44d48d97f7fe72ff609`，报告 SHA256 为 `89c8c71d696c30d2191f4fa70c7b129fedd5e78884991b11dbd2bc9e8d30c396`。

当前停止点：第一轮纯 PPO 修复消融没有解决 teacher-only 零响应。质量和学生/冲突响应不是阻塞项，阻塞集中在教师条件动作排序。根据已确认门槛，不能把“结构已实现”误写成“修复成功”，也不能回到第6项冻结。继续工作需要另行确认第二轮修复方向和预算；不得在未确认时自行改变损失、采样比例、网络结构或训练时长。

全 checkpoint 教师响应轨迹审计（2026-09-18 20:51 CST）：用户确认先做不训练的轨迹审计，以排查教师信号是否只在中途 checkpoint 短暂出现。新增独立配置 `pa_moap_rl/configs/preference_repair_checkpoint_trajectory_audit_v2.yaml` 和版本化脚本 `pa_moap_rl/experiments/audit_preference_repair_checkpoint_trajectory_v2.py`，审计版本为 `teacher_checkpoint_trajectory_audit_v1`。脚本只读取 A/B/C 训练结果，逐一校验协议、冻结目标、训练输入、checkpoint、arm 和完整 `experiment_spec`，不修改 checkpoint、不反向传播、不训练；相关单元测试与版本恢复回归共 7 项通过，`--validate-only` 确认 3 个 arm、33 个 checkpoint 和 72 个受控内容实例全部可用。

筛选覆盖 A/B/C 的 update 0、10、…、100，共 33 个 checkpoint × 72 个实例。每个 checkpoint 都在同一 effect-greedy 初始 assignment 上比较 balanced 与 teacher-only 的 logits，并用冻结 b075 目标精确重算所有合法单点动作的真实收益、策略首选动作收益、真实最佳动作策略名次、首个正收益动作名次、top-k 命中、logit/收益相关性和画像扰动导致的排序变化。初始状态并非没有教师可用信号：除 B10/C10 各 70/72 外，其余 checkpoint 的 teacher-only 策略首选动作在 72/72 实例上均为真实正收益；各臂中途还出现明显画像条件排序变化，A30、B10、C20 的初始首选动作变化率分别达到 100%、94.44%、62.50%。但是，真实最佳教师动作仍未被置于最前，三个臂的最佳中位名次只达到 A70=56.5、B50=60.0、C30=61.5。这把故障位置进一步缩小为“初始正向动作无法在多步轨迹中形成画像特异的 best-so-far 解”，而不是“教师输入完全不影响起始策略”。

按预注册的 `selected_plus_max_top_positive_plus_min_best_rank_v1` 规则去重后，完整复核 A50/A70、B30/B50、C30/C90 共 6 个 checkpoint；每个 checkpoint 对 72 个实例分别运行 balanced 与 teacher-only 的确定性 512/32 纯 PPO 轨迹，合计 864 次推理。六个 checkpoint 的 teacher-only 正重求解增益率全部为 0%，均值和 P90 增益均为 0；A50/A70/B50/C30/C90 的最终 assignment 在 72/72 上完全不变，B30 仅 4/72 出现 assignment 变化，但收益最小为 `-1.11e-16`、最大为 0，属于数值等分替换而非教师偏好响应。全部 432 个 checkpoint–实例复核记录的评分重算误差为 0。由此排除“仅因最终 best checkpoint 选择错误而丢失教师响应”，也没有发现可直接回退采用的中途 checkpoint。

审计结果位于 `results/four_topology_b075_pure_ppo_ablation_v1/checkpoint_teacher_trajectory_audit_v1/`；机器摘要 `audit_summary.json` 的 SHA256 为 `d200dc36c82cad4bd57021d623dcc6abaeca1212d813a9920a9b68d503e279e9`，报告 `checkpoint_teacher_trajectory_report.md` 的 SHA256 为 `aa9afa80bb0173a55f56be3bc85bd74befadf48a914c49c39eeda89326eb25dc`。运行状态明确为 `completed_no_training_not_frozen`，`transient_teacher_signal_found=false`；PPO 与推理设置仍不得冻结。

新的停止点：第二轮修复不应继续尝试挑选已有 checkpoint，也不应仅延长训练或推理预算。当前证据优先支持在保持 b075、数据角色和 512/32 评估不变的前提下，引入“同状态反事实排序监督”，直接约束 teacher-only 相对 balanced 的动作 logits 朝真实 `delta_J` 改善方向变化；另一个需单独版本化的备选是修改问题/动作机制（例如显式停止动作或只允许正收益推进）。两者都会改变训练损失或问题模型，属于关键且不可混用的方向，必须先提交独立版本、A/B 协议、预算与验收门槛并取得用户确认，不得直接启动训练。

#### 6A-2. D/E 同状态反事实排序监督（受控复核完成，仍不合格）

用户边界：完全独立版本、旧产物只读；先实现并提交 D/E 协议，不启动训练。问题模型、动作空间、冻结目标、数据角色、配对采样、目标一致特征和交换对称网络均保持 A/B/C 的既有版本；本轮只新增训练损失及其默认关闭的接入层。若未来修改问题模型或网络结构，必须另建版本，不能混入 D/E。

独立实现与版本：

- 精确动作排序损失为 `exact_delta_listwise_rank_v1`：对每个状态的全部合法单点替换动作计算冻结 b075 下的精确 `delta_J`，以目标温度形成 listwise 分布并对策略排序施加 KL 监督。
- 同状态反事实对齐为 `same_state_counterfactual_logit_alignment_v1`：只在 balanced 状态上替换 student-only 和 teacher-only 完整画像，约束中心化 logit 变化对齐真实 `delta_J` 变化；使用按真实变化绝对值加权的 smooth-L1，并同时监督学生、教师两个方向。
- PPO 辅助损失回调为 `ppo_auxiliary_loss_callback_v1`，训练接入为 `formal_training_counterfactual_ranking_v2`。共享 PPO 与 formal-training 入口只增加默认关闭的 callback；未提供 callback 时旧公式和更新路径不变。
- D=`C + exact-delta listwise ranking`；E=`D + same-state counterfactual logit alignment`。D/E 除对比项外保持相同，均继续使用 `single_replacement_masked_mdp_v1`、`paired_counterfactual_v1`、`objective_delta_symmetric_v2`、`objective_consistent_symmetric_features_v1` 和 `masked_actor_critic_exchange_invariant_v1`。
- 新文件为 `pa_moap_rl/experiments/preference_ranking_supervision_v3.py`、`pa_moap_rl/experiments/run_preference_ranking_ablation_v3.py`、`pa_moap_rl/configs/preference_ranking_ablation_protocol_v3.yaml`、`docs/preference_ranking_ablation_protocol_20260918.md` 和 `tests/test_preference_ranking_supervision_v3.py`。

候选协议：两臂均从头使用 seed0、100 updates 及 run03/C 的其余训练参数；后续评估仍为 512/32。当前未冻结候选为排序权重 0.30、目标温度 0.0025、E 对比权重 3.0。固定 seed0、四环境、4 步真实清单无更新预检中，D/E 排序梯度相对 PPO 基础梯度均为 7.117%；E 对比梯度为 0.981%，E 总辅助梯度为 7.182%，全部通过预注册的 3%--15%、0.5%--5% 和不超过 20% 三项尺度门槛。预检只执行 forward/autograd，不构造 optimizer、不执行 step、不保存文件。

非覆盖与验证记录（2026-09-18）：D/E runner 启动前逐项校验旧 A/B/C 配置、采样器、模型、runner、三个 best checkpoint、controlled-cross 汇总和轨迹审计 SHA256。旧配置、采样器、模型、runner 及 A/B/C best 哈希仍分别为 `5f5da34a...35f6`、`faae1182...7329`、`69183e47...d37d`、`7d559a88...642c0`、`61fc0542...afa6`、`357837db...4425`、`5a4cfeb3...2641`，与协议登记一致。D/E checkpoint 的完整 loss spec 写入 experiment spec schema 2，D checkpoint 恢复为 E 会立即拒绝；非空新目录也不能静默覆盖。

完整本地测试共 104 项通过；其中覆盖 float32 动作标签与 float64 完整评分误差不超过 `5e-7`、最终评分门槛仍为 `1e-9`、D/E 双方向有限梯度、旧 PPO 默认关闭路径逐参数一致、checkpoint 规格错配拒绝及全部既有工程回归。D/E `--validate-only` 均返回 `validated_no_training`，旧产物哈希核验通过。协议、损失模块和 runner 的 SHA256 分别为 `699e87249d6b75e127b8e66ed90f87dc737fba59f34a999acb659f539a527a9d`、`63113c1730e25c24dd77f36cd77d8fee1483cf24114780ed0ab13143012d5249`、`96abe61fa1ed380ebe4783a48dd098bbd1a6315e9f21de0122d19695e7147a0e`。

确认与启动记录（2026-09-21）：用户明确接受排序权重 0.30、目标温度 0.0025、E 对比权重 3.0、梯度预检门槛、验收门槛和 D→E 顺序，并授权训练。协议状态改为 `user_confirmed`，但明确标记为只确认本轮 D/E 消融，不构成全局 PPO/推理设置冻结。确认版协议 SHA256 为 `0497bb868622819d156e24004535b3448426c2adff63e9f5b4909aabb0e53e62`。新增独立顺序协调入口 `run_preference_ranking_ablation_sequence_v3.py`，版本为 `preference_ranking_ablation_sequence_v1`；启动前 26 项相关回归测试、D/E `--validate-only`、两臂无优化器梯度预检和旧 A/B/C 哈希复核全部通过，新结果目录此前不存在。

训练完成记录（2026-09-21 19:30 CST）：协调状态为 `completed`，D/E 均从头完成 100 updates，各有 400 条训练实例记录和 792 条固定 validation 记录；D/E 训练耗时分别为 562.70s 和 534.05s，协调 stderr 为空。D 的 best checkpoint 为 update 90，固定 72 对 mean J=`0.7659875638`、mean exact gap=`0.4497%`、P90 gap=`0.7224%`；E 的 best checkpoint 为 update 80，mean J=`0.7661743297`、mean exact gap=`0.4249%`、P90 gap=`0.8183%`。两臂全部 validation 硬可行、mask violation 为 0，训练非法动作数为 0；排序、对比、总辅助损失和 PPO 关键数值均无 NaN/Inf。D/E best checkpoint SHA256 分别为 `cbde752d982ccc1250752ee35d007c026ce430f526f48150e802b1deae8e5cce` 和 `3d901e3fe2a5e11bd59129491584afc62748cfd80781fa0d6e94bec618fe89e6`，均携带确认版协议哈希和严格区分 D/E 的 experiment spec。旧 A/B/C 代码、best checkpoint、controlled-cross 汇总和轨迹审计哈希再次复核，全部未变化。

当前停止点：D/E 训练完成只证明工程和固定 72 对质量正常，尚不能判断 teacher-only 零响应是否修复。下一步应使用 D90/E80 在固定 72×4 受控画像上执行纯 PPO 512/32 响应复核，并应用已确认的 teacher-only、student-only、conflict、质量和机制门槛；在该复核完成前不得选择 D/E、冻结 PPO/推理设置或进入正式多种子训练。

受控响应复核记录（2026-09-21 20:40 CST）：用户明确授权使用 D90/E80 执行固定 72×4、纯 PPO、512/32 复核。新增独立配置 `preference_ranking_response_arm_{D,E}_v3.yaml`、汇总协议 `preference_ranking_ablation_evaluation_v3.yaml`、endpoint 机制汇总器 `summarize_preference_ranking_ablation_v3.py` 和失败即停协调器 `run_preference_ranking_evaluation_sequence_v3.py`。启动前 D/E `--validate-only`、无训练汇总协议校验、22 项相关回归测试、checkpoint 完整 experiment spec 和旧 A/B/C 哈希全部通过；评估目录此前不存在。

两臂各完成 288 个实例—画像对，共 576 次确定性纯 PPO 推理；没有训练、local search 或 hybrid 后处理，stderr 为空。每臂均生成 288 个逐对结果和 216 条非基线响应证据，全部硬可行、mask violation 为 0、最大评分重算误差为 0。完整测试套件随后 105 项全部通过。

门槛结果：D/E 的固定 72 对 mean/P90 gap 分别为 `0.450%/0.722%` 和 `0.425%/0.818%`，质量与工程门槛均通过。teacher-only 正增益率均为 0%；D 的平均重求解增益为 `-0.00051621`、聚类 bootstrap 下界为 `-0.00074536`，E 分别为约 `-0.00000127` 和 `-0.00000211`。D 虽有平均 `10.28%` 的 teacher-only assignment change，但全部未产生正收益；E 仅有平均 `0.16%` 的 assignment change，同样没有正收益分化。D 的 student-only/conflict 正增益率为 `0%/73.61%`，两个回归门槛均失败；E 为 `63.89%/98.61%`，两项通过。

endpoint 机制证据：在 balanced PPO 最终 assignment 上切换到 teacher-only 后，每个实例仍存在正收益合法单点动作；D/E 的正收益合法动作平均占比为 `8.54%/10.06%`，每实例最少为 `2/11` 个，真实最佳单点收益均值约为 `0.000711/0.000681`。但两臂在 72/72 实例上的策略首选动作均为负收益，endpoint 正收益首选率均为 0%；真实最佳动作的策略中位名次为 D=`46.0`、E=`158.5`，均未达到不高于 10 的门槛。D 的教师画像使同状态首选动作变化率达到 75%，但变化方向错误；E 的教师画像概率 TV 更大，首选动作变化率却只有 9.72%，也未形成正收益轨迹。

最终决策：D/E 所有 teacher-only 响应门槛、两项 endpoint 排序门槛和正收益轨迹分化门槛均失败；D 还破坏学生与冲突响应。因此 `eligible_arms=[]`、`recommended_arm=null`，不得选择 D 或 E，也不得冻结 PPO/推理设置。结果位于 `results/four_topology_b075_pure_ppo_ranking_ablation_v1/controlled_cross_evaluation_v1/`；机器摘要 SHA256 为 `accafb2822fa7f0aeb060cb45ca8debd67ddefeee76be22326a587fcc0d75425`，报告 SHA256 为 `472f859751b5adda534ee7b66036ae5842cfb738966789ecd2502af2faf89f76`。

新的停止点：同状态精确排序监督并未解决 teacher-only 零响应。证据不支持继续放大相同损失权重、延长训练或继续挑 checkpoint；D 表明单独排序监督可能牺牲其他画像响应，E 表明当前反事实 logit 对齐也未把教师信号转化为 endpoint 的正收益动作。下一步若继续纯 PPO，应先形成新的、独立版本的问题机制或训练目标方案并重新预注册；在用户确认前不修改问题模型、动作空间、网络结构或训练预算。

### 7. 实施方案3：批量服务效率实验

- 测试 batch=1/8/32/128。
- 覆盖“同内容多偏好”和“新内容加新偏好”两类负载。
- 按 N 分桶或 padding，记录实际 batch、并发和资源配置。
- 与合理线程/并发设置下的 Gurobi、greedy 和局部搜索比较。
- 报告质量匹配下的吞吐、平均/P95 时延、冷启动、稳态、资源占用和训练摊销。

验收点：只在统一端到端计时和质量匹配条件下讨论效率优势，不预设 PPO 必然更快或更优。

开闸与预检记录（2026-09-22）：用户明确授权启动。实验只使用第6项已冻结主线 `legacy_separate_v1`、run03 update60（SHA256 `f3bc64e0a33d990c6612fbc9a3f33b48e29654dbf284ddb0e4b3a6640fed13df`）、b075 目标和确定性 512/32；6A 的 A/B/C/D/E 分支继续只读且不进入本实验。新增真批推理引擎 `batched_policy_service_v1`、runner `service_efficiency_runner_v1`、配置 `pa_moap_rl/configs/service_efficiency_benchmark_four_topology_v1.yaml` 和协议 `docs/service_efficiency_protocol_20260922.md`。引擎为每个请求独立维护终止、patience 与 best-so-far，完成请求从活动 batch 压缩移除，不复用训练期结束后 reset 的采样逻辑，也不把 batch=128 静默拆成微批。

启动门禁已通过：Stage7 与 Stage6 冻结相关测试共 5 项通过；`validate-only` 已核验冻结配置、checkpoint、目标、方法矩阵和训练数据哈希；CUDA smoke 的串行/批量 assignment、评分、步数和停止原因完全一致，覆盖 batch=1/8、三类服务负载、greedy 和局部搜索，输出位于 `results/four_topology_b075_service_efficiency_v1_smoke/`。正式结果将独立写入 `results/four_topology_b075_service_efficiency_v1/`，支持按 cell 恢复；不训练 PPO，不修改冻结配置、checkpoint 或6A产物。

正式执行与协议纠偏记录（2026-09-22）：第一次正式执行目录 results/four_topology_b075_service_efficiency_v1/ 已机械完成，但复核发现 interpolation_scenarios_v2.json 实际只有 30 个画像，导致同内容 batch=128 在批内循环复用画像，且同内容基线只生成 30/36 个请求，违反“批内偏好互异”和固定请求数协议。该 run01 未删除、未覆盖，已写入 INVALIDATED_PROTOCOL_DEVIATION.md 并明确禁止作为正式证据。纠偏没有生成新底层内容，只新增 128 个确定性、目标无关且画像唯一的服务场景库 data_processed/four_topology_exploratory_v2/service_efficiency_scenarios_128_v1.json（SHA256 06cd120de7743ae04f4a7463871a4d138b83ceb4b21c9c599ee01db38afda9db），并在 runner 中增加场景库哈希、数量、全库画像唯一性、批内画像唯一性和精确请求数硬校验。

纠偏后的独立配置为 pa_moap_rl/configs/service_efficiency_benchmark_four_topology_v1_run02.yaml（SHA256 1dd3cc0406bbca0df33310b6dcea4b6ec5a13bfdddeeb628fa3a7b8e0da161f6），独立输出为 results/four_topology_b075_service_efficiency_v1_run02/；run01、冻结 checkpoint、旧配置和 6A 产物均未改写。新增 2 项回归后，Stage7/冻结主线定向测试共 7 项通过；纠偏后的 validate-only、CUDA smoke、串行/批量等价预检均通过。正式完成后完整本地测试套件 112 项全部通过。三次冷启动使用彼此独立的 Python 进程执行，完整子进程均值 3.948 秒、进程内加载至首请求完成均值 1.531 秒、首次推理均值 0.330 秒。

run02 于 2026-09-22 01:43--01:55 CST 正式完成，stderr 为空；共 10 个质量配置、24 个 PPO 服务单元、12 个基线服务单元，审计覆盖 360 条质量结果和 6,516 条服务请求，全部硬可行且 mask violation 为 0。冻结 PPO 512/32 的 mean/P90 exact gap 为 0.4580%/0.7781%，质量样本平均延迟 0.7520s；低预算 PPO 128/32 为 0.5755%/0.9089%。scalarized greedy 为 0.1167%/0.2353%、平均延迟 0.000637s，在本开发集质量与延迟上均优于冻结 PPO；local search 和全部三个 Gurobi 档也满足预登记的 PPO 质量匹配条件。

批量化本身有效：冻结 PPO 的同内容多偏好吞吐从 batch1 的 4.104 提升到 batch128 的 52.675 req/s（约 12.83 倍），N 分桶的新内容新偏好从 1.999 提升到 13.698 req/s（约 6.85 倍）；混合 padding 负载仅从 6.538 提升到 13.354 req/s（约 2.04 倍），显示 padding 明显削弱批量收益。128/32 的最高吞吐为同内容 batch128 的 53.639 req/s。但 scalarized greedy 的端到端吞吐在同内容/新内容单线程下分别为 367.57/246.09 req/s，同时质量更高；因此本数据不支持“冻结 PPO 在质量匹配的总体服务效率上优于合理基线”的主张。PPO 只证明了内部真批量扩展能力；对 Gurobi 的相对吞吐随负载和并发变化，不能外推为普遍优势。局部搜索使用 Python 线程并发时受 GIL 影响，4 线程反而低于单线程，须按实现限制解释。

资源与摊销：PPO 服务单元最大 CUDA 已分配显存约 2220.7 MiB。run03 seed0 训练耗时 577.330s 单独报告，不计入在线延迟；按 1千/1万/10万/100万请求摊销分别为 0.5773/0.05773/0.005773/0.000577s 每请求。最终机器审计为 final_audit_summary.json（SHA256 8b020d6123714ed25ec4ae1e2601cf5c77ffb9a837c5ef4f473370b44e64575b），UTF-8 报告为 service_efficiency_report_utf8.md（SHA256 4d2b2288b6d841b0e0fb4533447e1196d72b3fd21e8125c48676236c143fc00a），并生成完整 code/input/output 哈希清单。第7项状态为已完成；仍只属于开发数据效率证据，teacher-only 零响应局限保持不变，实验未训练 PPO。

### 8. 三种子正式训练与开发集确认

- 使用冻结目标和冻结 PPO 设置运行 seed0/1/2。
- 保存 best/latest checkpoint、TensorBoard、逐 update 指标、固定 validation 分项和恢复审计。
- 在 `dev_calibration` 与 `dev_seen_diagnostic` 上完成分层分析。
- `dev_seen_diagnostic` 必须继续标记为旧结果已查看，不得称为独立测试。

验收点：完成多种子稳定性、质量和运行成本汇总，但不把已查看数据包装为独立确认。

### 9. 获取真正的新内容数据并重新求解

- 在模型、超参数和分析口径冻结后，再获取新的底层内容数据。
- 按源家族隔离，生成四拓扑新实例和预先登记的新偏好组合。
- 使用冻结 b075 目标重新求解 Gurobi 精确/有界参考。
- 在正式封存前不查看 PPO 对新数据的结果。

验收点：形成未用于开发、未提前查看的新内容确认集。只有完成此项，才能进入真正的独立确认。

### 10. 正式独立评估

- 比较三种子 PPO、Gurobi、greedy、局部搜索及已确认的混合方案。
- 报告均值、中位数、P90、最差值以及 1%/2%/3%/5% gap 达标率。
- 按拓扑、规模和画像冲突程度分层。
- 区分统计差异、计算质量和教学意义。
- 主张范围限定为四类已知拓扑中的新内容与新偏好，不宣称未见拓扑泛化。

验收点：完成独立确认报告，且所有结论可追溯到冻结配置、数据和运行哈希。

### 11. 可选方案4：动态偏好调整

- 区分小幅偏好漂移与突变。
- 比较旧解不动、PPO 重启/旧解启动、局部搜索旧解启动和 Gurobi warm start。
- 记录质量、时延和 assignment 变化，但不把变化数量直接解释为偏好处理质量。
- 只有另行确认后，才将切换成本加入目标函数。

验收点：作为后置补充，不阻塞核心主线。

Phase A 确认与协议（2026-09-22）：用户确认先于第8项执行冻结 PPO 的无训练动态偏好诊断。独立版本为 dynamic_preference_adjustment_v1，配置为 pa_moap_rl/configs/dynamic_preference_adjustment_phase_a_v1.yaml，协议为 docs/dynamic_preference_adjustment_phase_a_protocol_20260922.md，输出为 results/four_topology_b075_dynamic_preference_phase_a_v1/。共同旧解定义为每个实例在无扰动 S6–T6 下由冻结 run03 update60、确定性 512/32 PPO 产生的 assignment；所有旧解启动方法使用同一旧解。36 个拓扑—规模分层实例分别执行 student-only、teacher-only、conflict 的 25% 线性漂移和 100% 突变，共 216 个动态条件。未加入切换成本，未训练 PPO，未修改冻结 checkpoint、Stage6A 或 Stage7 产物。

正式执行结果：216/216 个动态条件、1,512 条方法结果全部完成，stderr 为空，全部硬可行且 mask violation 为 0；5 秒 Gurobi 参考最优证明率为 99.54%，唯一未证最优条件按 certified bound 计算有界 gap。完整本地测试套件 115 项通过。正式配置 SHA256 为 b00a74dcf58d0c274ccb4e59424dbeb4ba178de80f355a99ffde07f65c7fe4b1，冻结 checkpoint SHA256 仍为 f3bc64e0a33d990c6612fbc9a3f33b48e29654dbf284ddb0e4b3a6640fed13df。

总体结果：PPO 旧解启动的 mean bounded gap 为 0.5747%，正重求解增益率 10.65%，平均增益仅 0.00000434，平均 assignment change rate 为 0.107%，中位端到端时延 0.1059s。PPO 重启分别为 0.5529%、33.80%、0.00017262、2.897% 和 0.3008s。旧解启动相对重启中位时延下降 64.78%，且 gap 退化 0.0219 个百分点，因而通过“相对 PPO 重启收益”门槛；但仅在 1/6 个方向—幅度单元进入 Pareto，低于 4/6 门槛，并在总体 gap—时延上同时被 scalarized greedy 和 Gurobi 旧解 warm start 支配。

基线证据：scalarized greedy 的 mean bounded gap 为 0.1616%、正增益率 83.80%、中位端到端时延 0.00288s；Gurobi 重建模型+旧 PPO MIP start 为 0.0198%、95.37% 和 0.0805s；16 次旧解局部搜索为 0.1685%、100% 和 2.1899s。Gurobi 重建模型+默认标量初解为 0.0090% 和 0.0651s，表明本数据中旧 PPO warm start 甚至没有改善 Gurobi 的总体质量—时延。

teacher-only 结果继续为零响应：25% 漂移和 100% 突变下，PPO 旧解启动与 PPO 重启的正增益率、平均增益和 assignment change rate 均为 0；对应 mean bounded gap 分别为 0.8065% 和 0.8473%。相比之下 scalarized greedy 两档正增益率均为 100%，mean bounded gap 约 0.0014%，局部搜索两档正增益率也均为 100%。因此动态旧解启动没有解决教师偏好零响应。

预注册结论：decision_status=stop_after_phase_a_no_signal，eligible_for_phase_b=false。按协议不扩展到 72 实例，不实现持久化 Gurobi 模型复用，也不把 conflict 100% 突变中唯一一次 PPO 旧解启动进入 Pareto 的局部现象解释为总体优势。机器摘要 dynamic_preference_summary.json 的 SHA256 为 dad2479e8e36e484c72d1dbddfeb82cce9a72090bb869285efb7706b1066347c，报告 dynamic_preference_report.md 的 SHA256 为 9e955f7547498b7ef7211003e17a9b8a1c13a3213b368b8c48c40fa6f34e248c，完整输入/输出哈希清单为 evidence_hashes.json。

### 12. 正式消融与成果整理

- 目标项、偏好输入和搜索预算等正式 PPO 消融必须从头训练；启发式目标项脚本不能替代。
- 整理方法定义、数据范围、staircase 排除理由、运行环境、线程、设备、配置、哈希、图表和复现命令。
- 如实披露探索性数据已查看、合成偏好未经真实教学验证，以及 F_G/rho 仅作为敏感性参照。
- 同步更新研究报告、图表源数据和最终交接文档。

验收点：形成可复核的最终研究包，不夸大泛化、因果性或效率结论。

## 数据独立性门禁

- 第2至第8项可以使用当前四拓扑探索性数据推进模型开发。
- 第9项是必须引入新底层内容并重新求解的节点。
- 未完成第9项前，不得将任何现有 validation/test 派生集合称为全新的独立确认集。
