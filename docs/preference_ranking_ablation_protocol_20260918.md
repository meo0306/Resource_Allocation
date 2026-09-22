# D/E 同状态反事实排序监督协议

状态：用户已于 2026-09-21 确认候选协议并授权按 D→E 顺序训练；该确认只适用于本轮消融，不构成全局 PPO/推理参数冻结。

## 目标与边界

本轮只检验一个新假设：A/B/C 已提供教师输入、配对画像、精确 delta_J
特征和交换对称网络，但 PPO 标量回报没有直接约束同一状态下的动作排序。

冻结目标仍为 b075，问题模型仍为
single_replacement_masked_mdp_v1，模型仍为
objective_delta_symmetric_v2，采样仍为
paired_counterfactual_v1。本轮不修改数据角色、网络结构、动作空间、
训练预算、推理预算或 checkpoint 选择规则。

## 非覆盖约束

- 原 A/B/C 配置、代码模块、checkpoint、训练日志、受控交叉结果和轨迹审计均只读。
- 新配置为 preference_ranking_ablation_protocol_v3.yaml。
- 新 runner 为 run_preference_ranking_ablation_v3.py。
- 新损失模块为 preference_ranking_supervision_v3.py。
- D/E 输出只能进入 results/four_topology_b075_pure_ppo_ranking_ablation_v1/。
- runner 启动前逐项验证旧产物 SHA256；任一漂移立即失败。
- 非空 D/E 输出目录不得被静默覆盖；恢复必须显式提供完全匹配的 checkpoint。

共享 PPO 更新入口只新增默认关闭的 auxiliary callback。旧调用不提供该
callback 时继续执行原 PPO 损失。扩展前的共享文件哈希保存在协议中，
旧路径行为由回归测试验证。

runner 另提供 loss-preflight 模式，只在真实训练清单抽取一个确定性四环境
batch，执行 4 步 rollout、前向和反向梯度检查；它不构造 optimizer、
不执行参数更新、不保存文件，也不创建 D/E 结果目录。

## 独立版本

- 排序标签/损失：exact_delta_listwise_rank_v1
- 同状态对比约束：same_state_counterfactual_logit_alignment_v1
- 辅助损失接口：ppo_auxiliary_loss_callback_v1
- 新训练接入：formal_training_counterfactual_ranking_v2
- experiment spec schema：2
- checkpoint payload：仍为 3，因为磁盘字段结构未改变；新增的完整损失规格
  写入 schema 2 experiment spec，恢复时严格比较，不能与 A/B/C checkpoint
  互换。

## D/E 定义

- D：C + 全合法动作精确 delta_J listwise KL 排序监督。
- E：D + 同一 balanced assignment 上的 student-only 和 teacher-only
  中心化 logit 变化，与按目标温度缩放的真实 delta_J 变化做 smooth-L1
  对齐，并按真实变化绝对值加权，使监督集中在画像敏感动作上。该形式在预测
  差接近零时仍保持有界梯度。

E 必须同时监督学生和教师方向，禁止只修教师通道。D/E 除对比项外完全相同。

训练 observation 使用 float32，因此逐动作 delta_J 标签与 float64 完整评分的
一致性门槛设为 5e-7；最终 assignment 的评分重算门槛仍保持 1e-9。

当前待确认的候选值为：排序权重 0.30、目标温度 0.0025、E 对比权重
3.0。这些是运行候选，不是冻结参数。权重尺度不能直接跨损失比较：在固定的
seed0 四环境、4 步无更新预检上，原候选 0.03/0.03 的排序梯度仅为 PPO
基础梯度的 0.71%，对比梯度仅为 0.0098%，因此按梯度而非损失数值校准。
候选门槛要求排序梯度比为 3%--15%，E 对比梯度比为 0.5%--5%，总辅助
梯度比不超过 20%。该预检只用于排除明显失衡的尺度，不替代训练消融。

## 训练与验收候选

如后续获准训练，D、E 均从头运行 seed0、100 updates，保留 run03/C 的
其余设置；validation 和受控响应复核继续使用 512/32。

除原有质量、teacher-only、student-only 和 conflict 门槛外，增加机制门槛：

- teacher endpoint 的策略正收益动作率至少 50%；
- 教师真实最佳动作的中位策略名次不高于 10；
- balanced 与 teacher-only 最终轨迹必须产生正收益分化。

若 D/E 失败，不通过继续增加训练或推理预算规避；下一方向必须作为新的问题
模型版本单独提出。当前协议状态不会启动任何训练。
