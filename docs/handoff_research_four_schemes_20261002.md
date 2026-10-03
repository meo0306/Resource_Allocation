# 四拓扑教学方法分配研究交接：项目基础、研究动机与四方案验证

编写日期：2026-10-02（北京时间）  
项目目录：`D:\project\Resource_Allocation`  
用途：后续研究路线讨论、人员或新会话交接。  
证据范围：本次直接核对项目文档、配置、数据清单及实验报告；最近一轮实验证据截至 2026-09-22。2026-10-02 是交接文档编写与文件复核日期，不是新增实验日期。

## 1. 接手时首先需要知道的结论

项目已完成四拓扑目标校准、探索性数据协议、seed0 稳定化训练、四个研究方案的开发性验证，以及教师偏好零响应的两轮纯 PPO 修复探索。目前目标和 PPO 主线均已冻结，四个方案均已有结果；不能退回“目标尚未冻结”“尚待制定 seed0 pilot”或“方案4尚未实施”的历史状态。

当前证据支持两项 PPO 内部能力：真批推理能够提高其自身吞吐，旧解启动能够缩短其相对重启的推理时间。但是，在已经验证的目标、数据、预算和实现下，尚未证明当前 PPO 相对合理基线具有总体质量—效率优势。教师单独偏好变化的有效响应也仍未解决。

后续讨论的核心是：现有问题结构是否需要学习型求解器，以及继续研究时应接受并确认负结果、转向强简单基线，还是提出有真实应用依据的新问题机制。继续执行哪条路线尚待讨论；本交接不构成启动训练或改变模型的决策。

| 对象 | 当前有效状态 | 接手时避免混淆 |
|---|---|---|
| 四拓扑目标 b075 | 2026-09-12 用户确认后冻结 | 校准报告仍保留当时“待冻结”的文字，不代表现在未冻结 |
| PPO 主线 | 2026-09-21 冻结 `legacy_separate_v1`、run03 update60、512/32 | 这是用户选择的工程基准，教师响应局限继续保留 |
| Stage6A 的 A/B/C/D/E | 训练、复核完成，全部未通过教师响应门槛，分支关闭且未采用 | 不把其中对称网络或辅助损失写成当前主线 |
| 方案1/2 | 已完成，使用 run03 update70 | 与后续 update60 的结果不可视为完全相同实验 |
| 方案3 | 已完成，有效证据为 run02 | 原 run01 有画像复用协议偏差，已标记无效并保留 |
| 方案4 | Phase A 已完成，按门槛停止 | Phase B、72实例扩展、持久化 Gurobi 模型复用均未实施 |
| 新版三种子、新内容独立确认 | 尚未完成；三种子正式训练未启动 | 旧版五拓扑三种子已完成，不能冒充新版重复实验 |

`mission_list.md` 的第4/5/7/11项分别对应研究方案1/2/3/4；第6项是冻结决策，6A 是修复分支，任务编号与方案编号不同。

## 2. 项目的基础问题与研究边界

### 2.1 项目解决什么问题

PA-MOAP 工程处理教学资源配置的教学方法分配阶段：上游已经选定知识点，本阶段为每个选中知识点从11种教学方法中选择一种，兼顾教学效果、学生偏好、教师偏好以及方法组合的多样性。

输入包括节点类别、认知负荷、方法属性、固定的方法—类别效果矩阵及学生/教师画像。知识点类别数为6，教学方法数为11。学生与教师局部偏好由画像与节点—方法交互特征计算；现有画像属于合成建模设定，尚无真实教学实证证明其效用函数或参数最优。

上游的先修关系、教学时间和外部资源容量等内容选择约束，没有在本分配阶段重新作为完整约束系统求解。当前分配模型也没有直接使用拓扑边；四种拓扑主要通过上游选择出的内容组成影响任务。因此不能将现有 PPO 描述为已经实现知识图谱拓扑推理，也不能将本阶段等同于完整的课程排程或资源调度。

### 2.2 当前数学目标

令 `a_i` 为节点 i 采用的方法，`π_m` 为方法 m 在全部 N 个节点中的占比。局部项均为节点平均值：

```text
F_E = mean_i E[i, a_i]
F_S = mean_i P_student[i, a_i]
F_T = mean_i P_teacher[i, a_i]
F_P = (F_S + F_T) / 2

H(π) = -sum_m π_m log(π_m + ε) / log(11)
V_H = max(0, 0.65 - H(π))²
V_C = sum_m max(0, π_m - 0.35)²

J_b075 = 0.35 F_E + 0.65 F_P - V_H - V_C
       = 0.35 F_E + 0.325 F_S + 0.325 F_T - V_H - V_C
ε = 1e-8
```

学生与教师严格等权；允许教学效果和偏好之间补偿性权衡，没有另设 F_E 下限。熵0.65与最大占比0.35是软惩罚阈值，不是必须逐条满足的硬约束。当前 `αG=0`，不启用参考分布项 `F_G`，`target_distribution=null`；不能据此解释为整个目标不含全局耦合，因为熵和占比惩罚仍然作用于整体分配。

目标权威文件为 [objective_frozen_four_topology_v2.yaml](../pa_moap_rl/configs/objective_frozen_four_topology_v2.yaml)。[model_definition.md](model_definition.md) 可用于理解输入、偏好函数和动作，但其旧默认权重不是新版参数依据。

### 2.3 当前求解方式

工程已实现随机可行、effect-only greedy、scalarized greedy、单点 best-improvement 局部搜索、Gurobi 精确/有界求解、混合求解及 PPO。

PPO 是从完整初解出发的单点替换策略：每一步选择节点—方法 `(i,m)`，奖励为真实目标增量 `J(a_next)-J(a_current)`。推理从 effect-only greedy 初解开始，使用确定性 masked argmax，并返回全过程 best feasible assignment。mask 排除不可行方法、当前方法与 padding 节点，但不排除负收益动作。持续选择错误动作时，best-so-far 可以保留旧解，却不能自动修正动作排序。

部署设想是一个共享策略覆盖四拓扑、九规模组和多种画像，无需每个实例或每次画像变化重新训练。能否获得这种共享策略的实际收益，需要与可直接更新偏好系数的启发式和精确求解器比较。

## 3. 为什么重新提出四个方案：motivation 与分析过程

### 3.1 旧版成果留下了什么问题

旧版五拓扑、旧目标下已经完成参数搜索及 seed0/1/2 各1500 updates，旧部署模型为 seed1 update1450。两类已查看测试中，PPO 的总分高于 scalarized greedy，但低于完整局部搜索。例如 interpolation 的旧 J 分别约为 greedy 0.655647、PPO 0.671166、局部搜索0.687444。

这些结果提供了进一步研究的起点，但有四个未回答的问题：

1. **总分提高来自什么？** 既有分项分析发现 PPO 相对标量 greedy 的提升主要来自参考方法分布项，不能由总分提高直接推出学生与教师偏好均改善。
2. **质量损失是否只是推理预算不足？** PPO 从初解做单点替换，128步可能限制大实例的节点覆盖，但重复动作、提前停止和错误排序也可能才是瓶颈。
3. **学习成本是否能被反复服务摊薄？** 单实例求解比较不能回答共享策略面对大量偏好请求是否有吞吐收益；旧计时边界也不足以支持新的公平加速结论。
4. **连续偏好变化是否改变方法排序？** 若保留旧解，PPO 是否能比重启及其他热启动方法更高效地调整，尚未直接验证。

旧版 Gurobi/混合 pilot 也显示精确求解并非必然很慢，短程 PPO+局部搜索未通过当时的质量—时间门槛。因此不能预设“PPO天然比优化器快”，也不能把昂贵的完整局部搜索作为唯一竞争对象。

来源：[旧版最终报告](ppo_parameter_search_and_final_evaluation_20260908.md)、[9月12日交接](handoff_research_restart_20260912.md)。旧版 J 与新版 J 的权重、范围不同，数值不能横向解释为算法进步；旧局部搜索是基线，不是全局最优上界。

### 3.2 新版研究需求如何收敛

用户希望继续研究 PPO，但研究主张转向“双主体偏好权衡 + 多偏好共享求解能力”，并接受算法可能没有优势的结果。对应调整为：学生教师等权；效果可以为偏好让步；避免方法单一与匹配某个人为参考比例分开处理；软惩罚独立校准；目标权重先由精确解的分项权衡确定，不根据 PPO 成绩反向挑选。

数据范围调整为 uniform、random、hourglass、bottleneck 四拓扑。排除 staircase 是探索后对课程适用范围与内容组成作出的决定，需要如实披露；不能写成 staircase 普遍不合法，也不能以算法成绩低作为唯一理由。原始数据及五拓扑结果继续保留。

这产生了两层需要分别回答的研究问题：

- **模型层面：** 冻结目标能否表达效果与双主体偏好的合理、可解释权衡？
- **算法层面：** 共享 PPO 是否在该目标上提供足以抵偿训练和工程成本的质量、时延、吞吐或动态调整收益？

四个方案围绕第二层展开，不能用模型存在偏好响应替代 PPO 已学会偏好响应的证明。

### 3.3 四个方案的提出逻辑

| 方案 | 当时的假设 | 需要什么证据 | 为什么有必要 |
|---|---|---|---|
| 1 双主体偏好响应 | 共享策略会针对不同主体的偏好改变给出有效改进 | 同一内容、真实新画像下，重求解优于旧解复用；学生/教师/冲突分别报告 | 排除总分提升只来自其他目标项或画像不敏感 |
| 2 搜索预算与质量 | 大规模误差有一部分来自128步限制 | 固定 checkpoint，隔离 max_steps 与 patience，比较 anytime、覆盖率与 gap | 区分容量/学习问题与纯推理预算不足 |
| 3 批量服务效率 | GPU 真 batch 能摊薄共享网络成本 | 质量匹配下的端到端吞吐、请求时延、冷启动与摊销 | 验证共享 PPO 最有希望的计算价值，而非只比单次前向 |
| 4 动态偏好调整 | 旧 assignment 可使 PPO 在偏好变化后低成本恢复高质量 | 同一个旧解出发，比较 PPO、局部搜索、Gurobi 与重新贪心 | 检验跨时刻状态复用是否形成静态实验未覆盖的收益 |

原计划将方案4后置；前三项未建立优势后，用户确认先于正式三种子执行方案4 Phase A。它是有明确停止门槛的补充假设检验，不是已决定无限延续的调参路线。

## 4. 四拓扑研究已完成的基础建设

### 4.1 目标校准及统一工程链

新增显式 `ObjectiveSpec`，统一完整/增量评分、结果重算、greedy、局部搜索、Gurobi、PPO环境与 checkpoint 恢复。新版调用使用显式目标及哈希；旧目标保留 legacy 路径。Scenario v2 保存画像信息，目标规格单独传入。

目标校准完成112个候选筛选、12个候选的受控交叉复核、6个候选的完整540对复核。最终复核的证优率为100%，最大评分重算误差 `7.922e-11`，低于 `1e-9` 门槛。校准没有训练 PPO，也没有按 PPO 分数选择目标。用户随后选择并冻结 b075。

精确校准的作用是建立分项权衡及计算参考，并不能证明某个系数具有教学因果最优性。校准报告中的历史等价锚点未通过对应求解门槛，不能把所有阶段、所有候选统称为已证最优。

证据：[Pareto 决策报告](../results/objective_calibration_four_topology_v2/pareto_decision_report.md)。

### 4.2 数据协议与实际规模

数据目录：[four_topology_exploratory_v2](../data_processed/four_topology_exploratory_v2/)。保持原 lineage，不重新随机划分；`source_family_id=group/instance_stem`，同一家族的四拓扑不能跨角色。

| 角色 | 来源 | 源家族 | 实例数 | 本次清单复核的 N 范围 | 访问语义 |
|---|---|---:|---:|---:|---|
| dev_train | 原 train 去 staircase | 630 | 2,520 | 37–738 | 开发训练 |
| dev_calibration | 原 validation 去 staircase | 135 | 540 | 38–743 | 目标校准、选参与开发评估 |
| dev_seen_diagnostic | 原 test 去 staircase | 135 | 540 | 38–738 | 旧结果已查看，不是独立测试 |

九个原始规模组 V 为60/80/100/150/200/300/500/800/1000；V 是上游原始节点规模，N 才是选中后实际待分配节点数。全体开发数据 N 范围为37–743，不能写成当前分配器已测试 N=1000。

核心72实例来自每规模组选两个源家族，按四拓扑平均 N 接近组内中位数和75%分位数确定，保留每家族四拓扑。四个无扰动受控画像为 S6–T6、S4–T6、S6–T4、S1–T4，形成72×4=288对。固定训练监控的72对不是这一全交叉集，画像与拓扑混杂，不能单独用于判断主体响应。

原 interpolation scenario bank 为30个画像；方案3另建128个互异服务画像。增加画像没有生成新的底层内容，也不产生独立内容测试集。角色计数、哈希、家族隔离与访问标签见 [data_audit.json](../data_processed/four_topology_exploratory_v2/data_audit.json) 和 [access_registry.csv](../data_processed/four_topology_exploratory_v2/access_registry.csv)。

### 4.3 seed0、update70 与 update60 的关系

新版 pilot 的有效 run02 完成100 updates，但后期 validation 退化。稳定化 run03 从头训练，将学习率从 `3e-4` 改为 `1e-4`、启用 `target_kl=0.03`，同样完成100 updates，未重现后程崩塌。

run03 在原128/32评估下的 best 是 update70，方案1/2使用它。随后将所有编号 checkpoint 按候选512/32重评，按 mean J 最大、P90 gap 最小、较早 update 的顺序选中 update60。用户在6A分支失败后决定冻结这一原始模型为主线，故不能用 `best.pt` 文件名代替显式 update60 路径。

当前训练协议：joint B=4、rollout128、100 updates、4 epochs、minibatch64、hidden64、lr=`1e-4`、target KL=`0.03`；训练环境128/32，评估与冻结推理512/32。以后若执行其他 seed，应各自按已冻结选择规则选 checkpoint，不应机械固定 update60。

权威记录：[PPO 冻结配置](../pa_moap_rl/configs/ppo_inference_frozen_four_topology_v2.yaml)、[冻结决策报告](../results/four_topology_b075_ppo_inference_freeze_v1/freeze_decision_report.md)。

## 5. 方案1：双主体偏好响应——完成，发现教师响应缺失

实验固定 b075、run03 update70、128/32，72实例×4画像×4方法，共1,152条结果。Gurobi 使用已有 b075 受控交叉精确结果。每种方法以其在 S6–T6 下的解为旧解，在新画像下重算旧解，再与该方法针对新画像重求解的结果比较。

有效响应量定义为 `J_new_profile(a_resolved) - J_new_profile(a_old)`。assignment变化仅描述敏感性；跨画像原始得分水平差不能直接解释为收益。

| 方法 | 288对 mean gap | P90 gap | student-only 正增益率 | teacher-only 正增益率 | conflict 正增益率 |
|---|---:|---:|---:|---:|---:|
| Gurobi exact | 0% | 0% | 100% | 95.83% | 100% |
| Scalarized greedy | 0.197% | 0.460% | 1.39% | 95.83% | 91.67% |
| 单点 best-improvement | 0.001% | 0.003% | 95.83% | 95.83% | 100% |
| PPO update70 | 0.563% | 1.187% | 44.44% | 0% | 79.17% |

PPO 在三个变化画像合并后的平均增益为0.000241，源家族聚类 bootstrap 95% CI为[0.000167, 0.000310]。这个正均值不能掩盖 teacher-only 在全部四拓扑零 assignment 变化、零增益。精确模型 teacher-only 平均增益0.000223，表明目标确有响应机会。

本方案也说明 scalarized greedy 不是在任何响应指标上都最好：学生变化下其重新逐点选择可能加重全局惩罚，因而低于自己的旧解。后续比较应保留这种分层事实，不能只挑总体均值。

结论：已证明部分响应，未证明完整双主体响应。证据限于已查看的开发数据及这一 seed/checkpoint。

报告：[preference_response_report.md](../results/four_topology_b075_preference_response_v1/preference_response_report.md)。

## 6. 方案2：搜索预算与质量——完成，解释部分大实例误差

固定同一 update70，对288个实例—画像生成最长确定性母轨迹，再按环境规则截断复算，得到2,880个条件结果。比较128/256/512步、随N缩放，以及固定512步的多档patience；未重新训练。

| 设置 | Mean gap | P90 gap | 平均推理时间 | 解释 |
|---|---:|---:|---:|---|
| 128/32 | 0.563% | 1.187% | 0.690s | 原锚点 |
| 512/32 | 0.443% | 0.766% | 0.874s | 获得主要质量收益 |
| 512/512 | 与512/32相同 | 与512/32相同 | 4.215s | 延长patience没有收益 |
| N缩放、最高768步 | 与固定512相同 | 与固定512相同 | 2.541s | 不支持默认按N持续加预算 |

group9 mean gap 从1.097%降至0.319%，group1–6没有改善。student-only 正增益率升至63.9%，conflict升至98.6%；teacher-only在全部预算和patience下仍为零。

长尾存在明显重复：512/512平均512个动作中397.8个为重复动作；平均节点覆盖率仅从22.27%升至22.86%。因此128步确是部分大实例瓶颈，但教师缺失不是单靠延长搜索能解决的。

本方案形成512/32主候选与128/32低时延参照，后经第6项决策冻结。此表的时间属于该次预算诊断，不应直接与方案3/4的不同请求集和计时口径相除。

报告：[search_budget_diagnostic_report.md](../results/four_topology_b075_search_budget_diagnostic_v1/search_budget_diagnostic_report.md)。

## 7. 插入的 Stage6A：教师零响应排查与 A—E 修复探索

### 7.1 已排除的解释与可定位的失败机制

直接核查和无训练诊断排除了：教师字段未传入、教师权重置零、完全未覆盖 T4/S6–T4、没有真实优化机会、动作mask阻断，以及单纯推理预算不足。教师矩阵进入编码器和actor，系数为0.325；run03的400个训练画像覆盖30个允许组合，其中 S6–T4 有12条。

在 update60 的平衡画像 PPO 最终分配上切换教师画像后，72/72实例策略首选动作均为负收益；同状态平均仍约有208个正收益合法动作，正收益动作占比约10.02%。真实最佳教师动作的策略平均名次约1057，中位数631。从 PPO 解继续做确定性 best-improvement 则可恢复95.83%的教师正增益响应。

因此，可以定位的直接机制是：**策略在关键终态附近严重错排可用的教师正收益动作，best-so-far保留旧解，重复轨迹最终停止。** 这支持策略学习或排序机制存在问题，而非目标或合法邻域没有机会。

需要继续保持因果边界：独立画像采样缺少同内容反事实、未显式给出目标一致delta、学生教师通道未绑定、标量奖励中的条件信号难以识别，是据此提出的候选解释；单seed、有限预算的消融未证明其中任一项是唯一根因。

诊断报告：[teacher_response_diagnostic_report.md](../results/four_topology_b075_teacher_response_diagnostic_v1/teacher_response_diagnostic_report.md)。

### 7.2 A/B/C：配对采样、目标一致特征与对称结构

| Arm | 同内容配对采样 | 目标一致特征及对称网络 | Best update | Teacher-only 正增益率 |
|---|---|---|---:|---:|
| A | 是 | 否 | 50 | 0% |
| B | 否 | 是 | 30 | 0% |
| C | 是 | 是 | 90 | 0% |

配对采样使用同一内容的 balanced/student-only/teacher-only 三元组，加另一内容的随机画像。新特征显式提供按N归一化且包含软惩罚的 `delta_J`；新结构使用均值与绝对差组合，保证学生教师交换不变性。问题/动作仍为 `single_replacement_masked_mdp_v1`，新模型包为 `objective_delta_symmetric_v2`。

三臂均从头seed0训练100 updates，并以512/32完成各288对受控复核。质量和学生/冲突保护门槛通过，但教师响应门槛失败；A/C在288/288对的最终assignment相同，B的少量教师变化只是数值等分替换。

进一步审计33个编号checkpoint，并对6个预注册选出的checkpoint完整复核。初始状态能够出现教师条件排序变化，但最终教师有效增益全部为零；没有发现可以直接替换采用的中途checkpoint。因而不能将失败归咎于仅仅选错了最终best。

报告：[A/B/C复核](../results/four_topology_b075_pure_ppo_ablation_v1/controlled_cross_evaluation_v1/ablation_evaluation_report.md)、[checkpoint轨迹审计](../results/four_topology_b075_pure_ppo_ablation_v1/checkpoint_teacher_trajectory_audit_v1/checkpoint_teacher_trajectory_report.md)。

### 7.3 D/E：精确动作排序与反事实对齐监督

D在C基础上加入精确 `delta_J` listwise 排序监督；E进一步加入同状态 balanced→student-only/teacher-only 的logit对齐。预检后确认排序权重0.30、温度0.0025、E对比权重3.0，各自从头训练100 updates。

| 指标 | D / update90 | E / update80 |
|---|---:|---:|
| 固定72对 mean/P90 gap | 0.450% / 0.722% | 0.425% / 0.818% |
| Teacher-only 正增益率 | 0% | 0% |
| Teacher-only 平均增益 | -0.00051621 | -0.00000127 |
| Endpoint 首选正收益率 | 0% | 0% |
| 真实最佳动作中位排名 | 46.0 | 158.5 |
| Student-only / conflict 正增益率 | 0% / 73.61% | 63.89% / 98.61% |

D对教师画像确有动作变化，但方向错误且损害学生/冲突响应；E保留其他响应，教师改进仍失败。两臂均 `eligible_arms=[]`、无推荐模型。

A—E 的教师开发门槛包括正增益率至少50%、平均增益至少0.000055、源家族bootstrap下界大于0；D/E还要求endpoint首选和真实最佳动作排序改善。它们属于已查看开发集上的工程研究门槛，不是教学意义阈值。

这些结果只能说明已测试的实现与协议未成功修复，不能推出“配对采样无效”“对称网络必然无效”或“任何PPO都无法使用教师偏好”。把确定性局部改进接到PPO后面能修复输出，是混合求解器证据，也不等于纯PPO已学会。

用户最终决定以第6项初始结果返回主线，关闭整个6A分支。全部实现与结果保留，主线未采用对称网络或D/E损失。

报告：[D/E复核](../results/four_topology_b075_pure_ppo_ranking_ablation_v1/controlled_cross_evaluation_v1/ranking_evaluation_report.md)。

## 8. 方案3：批量服务效率——完成，内部加速成立但外部优势未成立

### 8.1 实际实现与有效运行

使用冻结 update60、`legacy_separate_v1`，独立实现 `batched_policy_service_v1`。各请求独立维护环境、patience和best-so-far，完成后从活动batch中移除。通过串行/批量assignment、步数、停止原因一致性门禁后，运行batch=1/8/32/128，比较512/32与128/32。

负载覆盖同内容多偏好、新内容新偏好的N分桶，以及混合N padding压力。质量前沿使用36个拓扑×规模分层请求；基线包含标量greedy、4/16/64轮局部搜索及0.25/1/5秒Gurobi。

正式有效目录是 `four_topology_b075_service_efficiency_v1_run02`。此前run01批内重复使用30画像，且部分基线请求数不符，已保留并标记无效。run02使用128个互异画像，完成10个质量配置、24个PPO服务单元、12个基线单元；共360条质量结果和6,516条服务请求，全部硬可行且零mask违规。

实际run02配置为每单元1个预热batch、6个测量batch、3次独立进程冷启动，基线并发1/4使用线程实现。此前讨论中的更大重复次数或独立进程并发建议不应写成已执行事实。

### 8.2 质量与吞吐证据

| 方法（质量前沿36请求） | Mean gap | P90 gap |
|---|---:|---:|
| PPO 128/32 | 0.5755% | 0.9089% |
| PPO冻结512/32 | 0.4580% | 0.7781% |
| Effect-only greedy | 8.2960% | 11.1796% |
| Scalarized greedy | 0.1167% | 0.2353% |
| 局部搜索16轮 | 0.0018% | 0.0042% |
| Gurobi 1秒档 | 0.0171% | 0.0000% |
| Gurobi 5秒档 | 0.0000% | 0.0000% |

PPO冻结档最差gap约4.9245%，不应只用均值和P90暗示所有实例都已达到1%以内。

| PPO冻结512/32负载 | Batch1 吞吐 | Batch128 吞吐 | 自身吞吐提升 | Batch128 P95请求时延 |
|---|---:|---:|---:|---:|
| 同内容多偏好 | 4.104 req/s | 52.675 req/s | 12.83倍 | 4.669s |
| 新内容新偏好、N分桶 | 1.999 req/s | 13.698 req/s | 6.85倍 | 21.502s |
| 混合N padding | 6.538 req/s | 13.354 req/s | 2.04倍 | 9.126s |

Scalarized greedy 在两类基线服务负载下的单线程端到端吞吐分别为367.57与246.09 req/s，同时质量前沿更优。因此批量化带来了PPO自身加速，未建立总体质量—效率优势；吞吐提高也不等于单请求时延降低。

测量边界需要随结果一起交接：质量前沿与服务负载使用不同请求组织，不能直接将某个质量表单请求时间取倒数当服务吞吐；PPO与基线的服务请求数和调度方式也不完全相同。现有证据可支持当前协议下未形成优势，但不宜外推为所有硬件、全部请求分布上的固定速度比。局部搜索线程并发还受Python/GIL限制，不能等同于最优并行实现。

冷启动：子进程总墙钟均值3.948s，进程内加载至首请求完成1.531s，首次推理0.330s；峰值CUDA已分配显存约2220.7MiB。run03训练耗时577.330s单列摊销，未计入在线时延。

报告：[正式报告](../results/four_topology_b075_service_efficiency_v1_run02/service_efficiency_report_utf8.md)、[审计摘要](../results/four_topology_b075_service_efficiency_v1_run02/final_audit_summary.json)、[实际run02配置](../pa_moap_rl/configs/service_efficiency_benchmark_four_topology_v1_run02.yaml)。

## 9. 方案4：动态偏好调整——Phase A完成，按预注册门槛停止

### 9.1 公平起点与测试条件

使用冻结update60、512/32、b075。选36个拓扑—规模分层实例，每个实例先在S6–T6下生成同一个PPO旧解；所有warm-start方法均从该assignment出发。student-only、teacher-only、conflict各设25%线性漂移与100%突变，共216个动态条件、7方法、1,512条结果。

PPO不训练，不加入assignment切换成本。局部搜索最多16次best-improvement；Gurobi在线档1秒、单线程、固定seed，参考档最多5秒。两种Gurobi均重新建模：分别加默认标量初解或旧PPO解MIP start，未实现持久化模型复用。

### 9.2 结果与停止判定

| 方法 | Mean bounded gap | 正重求解增益率 | 中位端到端时延 |
|---|---:|---:|---:|
| 旧解不动 | 0.5753% | 0% | 0.002295s |
| PPO旧解启动 | 0.5747% | 10.65% | 0.105931s |
| PPO从effect-greedy重启 | 0.5529% | 33.80% | 0.300806s |
| Scalarized greedy | 0.1616% | 83.80% | 0.002878s |
| 局部搜索旧解启动 | 0.1685% | 100% | 2.189853s |
| Gurobi重建+旧PPO初解 | 0.0198% | 95.37% | 0.080496s |
| Gurobi重建+默认标量初解 | 0.0090% | 98.15% | 0.065122s |

PPO旧解启动相对重启快64.78%，gap退化约0.0219个百分点，通过相对自身收益门槛。但只在六个方向—幅度单元中的1个进入Pareto，低于4/6要求，并被标量greedy和Gurobi旧解warm start在总体质量、时延上同时支配。

Teacher-only两档中，PPO重启和旧解启动均为零正增益、零assignment变化；greedy和局部搜索两档正增益率均为100%。Gurobi使用旧PPO解也没有优于默认标量初解的总体表现，旧解复用不是天然收益。

5秒参考215/216证得最优，剩余1条使用certified bound。因此表中用bounded gap，不能将全部216条统称精确最优参考。所有结果硬可行、零mask违规。

最终状态 `stop_after_phase_a_no_signal`、`eligible_for_phase_b=false`。依协议不扩展72实例，不实施持久化Gurobi模型复用。唯一Pareto单元为conflict的100%突变，不足以替代总体门槛。

证据：[动态协议](dynamic_preference_adjustment_phase_a_protocol_20260922.md)、[报告](../results/four_topology_b075_dynamic_preference_phase_a_v1/dynamic_preference_report.md)、[机器摘要](../results/four_topology_b075_dynamic_preference_phase_a_v1/dynamic_preference_summary.json)。

## 10. 当前结果如何解释，以及还不能解释什么

### 10.1 跨方案能够支持的判断

当前目标能够表达教师变化带来的优化机会，精确与局部方法能够利用这些机会；已测试PPO则未形成完整双主体响应。加推理步数、改变checkpoint、同内容配对、目标一致特征、对称结构、排序与反事实监督、旧解启动均已在登记范围内检验，未找到可采用的纯PPO修复版本。

方案1/2主要定位响应与预算问题，方案3/4主要检验部署价值。它们共同支持“当前冻结PPO及已测试修复路线未实现预期额外收益”，不支持“所有PPO或所有强化学习都不适合此类问题”的普遍结论。

### 10.2 简单方法较强的结构性解释（分析判断，非因果证明）

b075的正向部分是节点—方法的可加平均收益。标量greedy能以很低成本逐点选取高分方法；跨节点耦合主要来自方法计数分布的熵与占比惩罚。当前没有直接拓扑边约束、跨时间切换成本或大量额外组合资源约束。在这一结构和现有数据上，greedy已捕获多数收益，局部搜索可进一步修正分布惩罚，Gurobi也较快。

PPO却需承担逐步网络推理和环境更新，并从effect-only而非完整标量目标初解出发。这为“问题给学习型策略留下的收益空间较小”提供合理解释，但尚未通过单独隔离耦合强度、初解选择或数据难度的实验建立唯一因果关系。

即使以后修复教师排序，也需单独证明相对强基线的额外价值；教师响应合格与质量—效率竞争力是两个不同验收目标。

### 10.3 必须保留的证据边界

- 所有新版结果来自反复查看的开发数据；尚无真正新内容的独立确认。
- 新版主线及修复主要是seed0，不能据此报告新版多种子稳定性或普遍失败概率。
- 画像、效果矩阵及软阈值为建模设定，计算得分不能直接解释为真实学习效果提升。
- 方案1/2使用update70，方案3/4使用update60；固定72对、288对、36请求和216动态条件的均值不能直接互作增量。
- 硬可行和评分一致只能证明已检查的工程性质，不等于策略机制被完全解释。
- “115 passed”是9月22日保存的完整测试结果；本次交接只做文档、清单和关键哈希核验，没有重新执行测试或实验。
- 尚未进行新的运行进程盘点；历史对话中的“当前无进程运行”不能直接当作10月2日的实时监控结果。

## 11. 后续方案讨论的可用起点（建议，尚未批准实施）

建议先确定预期研究贡献，再决定是否投入下一轮训练。下列路线可讨论组合，但每条都需要写清可推翻的假设。

| 路线 | 要回答的问题 | 最小下一步 | 主要取舍 |
|---|---|---|---|
| A 确认与整理当前负结果 | 当前PPO弱点是否跨seed稳定，比较是否可复现？ | 预注册多种子和必要计时复核，随后新内容确认 | 能提高结论可信度，但不应期待重复种子自动产生优势 |
| B 模型为主、强基线求解 | 双主体权衡模型是否有独立价值？ | 围绕精确/greedy/local结果整理权衡与解释，PPO保留为比较项 | 更贴合现有证据，但需重新明确方法贡献 |
| C 有应用依据的新问题机制 | 真实场景是否存在当前遗漏的跨节点、跨时刻或不确定性约束？ | 先说明业务来源和数学定义，用小型精确/基线实验判断难度，再决定学习方法 | 问题与数据会发生实质变化，必须独立版本并重新校准 |
| D 新学习求解机制 | 能否在强初解附近可靠识别改进并降低总成本？ | 先诊断状态/动作/监督与初解机制，设计可解释小实验 | 不能把简单局部改进包装为PPO学习贡献；强基线仍必须保留 |

路线C中可能讨论真实的课程序列联动、教师资源竞争、在线反馈或分配切换代价，但这些目前都不是已验证需求。不能为了使greedy变差而凭空加约束，也不能未经确认并入b075。

如果继续纯PPO修复，下一版应解释为何A—E已失败的机制会在新设计下改变，并设置早停门槛；只加训练时长、放大同一损失或继续挑checkpoint，现有证据不足以支持优先投入。若使用greedy初始化、动作收益筛选或局部搜索后处理，应明确成本与贡献来源，并保留同样配置的非学习基线。

建议下一轮讨论形成一页决策：目标应用与贡献、保留/变更的数学对象、强基线、数据来源、最小可行实验、通过及停止门槛、版本与预算。正式训练、新数据生成和独立确认安排应在该决策明确后推进。

## 12. 交接操作、关键证据与核验记录

### 12.1 阅读优先级与历史状态处理

接手顺序：本文 → 两份冻结配置 → 对应正式结果/机器摘要 → `mission_list.md` 的时间顺序记录 → 9月12日需求和旧版文档。针对每个具体结论以当次配置、结果与后续明确决策共同判定，不能只读取旧协议标题。

特别注意四处历史状态：

1. 9月12日需求和交接中的“未冻结”已由b075和9月21日PPO冻结取代。
2. A/B/C协议标题中的“未启动”是编写时状态；实际训练与复核已完成。
3. `ppo_inference_freeze_candidate_four_topology_v2.yaml` 保留历史待决状态；当前权威入口是 `ppo_inference_frozen_four_topology_v2.yaml`。
4. 冻结配置解除的是第8项前置条件，不能解读为用户后来已经下达新版三种子启动命令。方案4停止后，当前请求是整理交接以讨论下一路线。

### 12.2 关键标识及本次实际哈希复核

```text
代码基准（2026-10-02读取HEAD，不代表全部历史实验均在此提交运行）：
4e15a8164ec90bed7d608026465667215a9ccbdb
feat: checkpoint four-topology research baseline

冻结ObjectiveSpec语义哈希（配置登记）：
b075006e79dc73a06ce76049ac0e3350c2d6b9b3ceb79db6fec57294e5b51606

objective_frozen_four_topology_v2.yaml 文件SHA256（本次复算一致）：
71c48aa3eb3c48934aa86eba5761e4b403950e8737a6716453c2be449c7670e7

ppo_inference_frozen_four_topology_v2.yaml 文件SHA256（本次复算一致）：
eca9a2aa6c221acdd04a54fb391f2304c35ab54cff57975165b54a2a43928cd6

run03/checkpoints/update_000060.pt 文件SHA256（本次复算一致）：
f3bc64e0a33d990c6612fbc9a3f33b48e29654dbf284ddb0e4b3a6640fed13df
```

语义哈希与YAML文件字节哈希不是同一个量。checkpoint需同时匹配目标、方法矩阵、数据、结构和实验规格；只看到相同文件名不构成恢复依据。

run03保存环境包括Python3.11.9、PyTorch2.12.0+cu126、RTX4060 Laptop GPU。历史效率实验硬件还记录i7-13700H、8GB GPU；未来运行应重新核对实际设备、软件和Gurobi许可证，不能把保存元数据当作当前环境自动合格证明。

### 12.3 文件与代码入口

| 用途 | 入口 |
|---|---|
| 初始新版需求和四方案定义 | [research_requirements_and_plan_20260912.md](research_requirements_and_plan_20260912.md) |
| 执行与授权时间线 | [mission_list.md](mission_list.md) |
| 简明仓库快照 | [project_status_20260922.md](project_status_20260922.md) |
| 目标规格与统一评分 | [objective.py](../pa_moap_rl/objective.py)、[scoring.py](../pa_moap_rl/utils/scoring.py) |
| 环境与精确求解 | [method_assignment_env.py](../pa_moap_rl/envs/method_assignment_env.py)、[gurobi_exact_solver.py](../pa_moap_rl/solvers/gurobi_exact_solver.py) |
| 稳定化训练配置 | [run03配置](../pa_moap_rl/configs/seed0_pilot_four_topology_v2_stabilization_run03.yaml) |
| 方案1入口 | [run_preference_response_v2.py](../pa_moap_rl/experiments/run_preference_response_v2.py) |
| 方案2入口 | [run_search_budget_diagnostic_v2.py](../pa_moap_rl/experiments/run_search_budget_diagnostic_v2.py) |
| A/B/C协议及D/E协议 | [A/B/C](preference_repair_ablation_protocol_20260918.md)、[D/E](preference_ranking_ablation_protocol_20260918.md) |
| 方案3入口与证据哈希 | [run_service_efficiency_v1.py](../pa_moap_rl/experiments/run_service_efficiency_v1.py)、[evidence_hashes.json](../results/four_topology_b075_service_efficiency_v1_run02/evidence_hashes.json) |
| 方案4入口与证据哈希 | [run_dynamic_preference_adjustment_v1.py](../pa_moap_rl/experiments/run_dynamic_preference_adjustment_v1.py)、[evidence_hashes.json](../results/four_topology_b075_dynamic_preference_phase_a_v1/evidence_hashes.json) |

以上是定位入口，不是要求重新执行已完成任务。新版训练、求解或恢复前应先读取所用runner和配置的实际参数与哈希门禁，不能直接套用README中的通用训练示例。

### 12.4 非覆盖与交付边界

用户明确要求独立版本和可回退：问题模型、动作空间、网络结构、损失、采样、配置和结果若有实质变化，应新增可识别版本；旧配置、checkpoint和历史结果只读保留。无效运行保留原因，不删除、不混入有效统计。

源码、轻量配置和文档纳入Git；`pre_data/`、`data_processed/`、`results/`、checkpoint与TensorBoard等被`.gitignore`排除。仅交付Git仓库不足以复现实验，换机器或交接人员时必须另行交付必要数据、运行产物及哈希清单。本次没有复制或移动这些大体积产物。

本次写入前工作区存在未跟踪的 `docs/stage_report_20260908_2245/`，保持原样。本次仅新增本文，不修改mission历史、冻结配置、代码、数据或实验结果，不启动训练或求解。关键冻结文件哈希和三个角色清单计数已直接复核；没有声称完成全量产物重新审计。

### 12.5 可直接用于后续讨论的启动说明

> 请接续 Resource_Allocation 项目，先读 docs/handoff_research_four_schemes_20261002.md，再按需核对冻结配置、mission_list和相应正式结果。b075目标和legacy_separate_v1/run03 update60/512-32主线已经冻结；方案1至4均已在开发数据上验证，未建立当前PPO的总体优势；A—E修复分支已失败关闭，方案4停止在Phase A。当前任务是讨论下一条研究路线，先区分已证实机制和未证实假设，提出有应用依据、可证伪且版本独立的方案。保留所有旧产物。不要把旧协议状态当成当前状态，不要自动重跑旧实验、启动三种子或更改问题模型；下一阶段的实质选择需按已有用户确认边界明确决定。
