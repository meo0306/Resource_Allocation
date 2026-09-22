# 纯 PPO 双主体偏好修复 A/B/C 消融协议（已确认，未启动训练）

状态：用户已于 2026-09-18 明确确认协议与验收门槛，YAML 状态为 `user_confirmed`。当前只完成实现与非训练验证，尚未收到启动 A/B/C 训练的指令。

## 1. 不变量与独立版本

- 冻结问题目标：b075，哈希 `b075006e79dc73a06ce76049ac0e3350c2d6b9b3ceb79db6fec57294e5b51606`。
- 问题/MDP：`single_replacement_masked_mdp_v1`。
- 采样：`independent_random_v1` 或 `paired_counterfactual_v1`。
- 旧特征：`separate_student_teacher_features_v1`；新特征：`objective_consistent_symmetric_features_v1`。
- 旧网络：`masked_actor_critic_separate_channels_v1`；新网络：`masked_actor_critic_exchange_invariant_v1`。
- 模型包：`legacy_separate_v1` 或 `objective_delta_symmetric_v2`。
- 训练接入：`formal_training_versioned_injection_v1`；versioned checkpoint payload 为 v3。

以上标签分别进入 checkpoint 和运行元数据。恢复时完整 `experiment_spec` 必须逐字段相等；任何采样、特征、网络、目标、协议或 arm 不一致均立即失败。旧实现未被覆盖，仍可走 `legacy_separate_v1` 回退路径。

## 2. 三个 arm

| Arm | 配对采样 | 目标一致对称特征/网络 | 解释 |
|---|---|---|---|
| A | 是 | 否 | 只检验同内容反事实监督 |
| B | 否 | 是 | 只检验显式 `delta_J` 与交换不变结构 |
| C | 是 | 是 | 检验两者组合与交互 |

配对采样每 4 个环境使用两个内容实例：同一 anchor 内容生成 S6–T6、S4–T6、S6–T4 三元组，第四个环境保留另一内容的独立随机画像。未变化主体的完整 profile 逐值共享，不只共享模板名。

新结构把学生/教师通道变换为 mean 和 absolute gap，并显式加入按 b075、N 和软惩罚计算的单步 `delta_J`。因此交换学生/教师矩阵时 logits 和 critic value 应保持不变；分主体原始得分仍由环境日志保存用于审计。

## 3. 公平运行协议

三个 arm 均从头训练 seed0，沿用 run03：100 updates、每次 4 个环境、rollout 128、minibatch 64、4 epochs、学习率 `1e-4`、entropy `0.01`、target-KL `0.03`、hidden 64、训练预算 128/32。固定 72 对 validation 使用 512/32，在 update 0、10、…、100 评估。

每个 arm 都必须完整运行，不能因前一个 arm 科学门槛失败而停止后续 arm。checkpoint 仍按 mean J、P90 exact gap、较早 update 的顺序选择。训练与推理均为纯 PPO，不追加 local search 或 hybrid 修补。

## 4. 训练后复核和拟议门槛

每个 arm 的入选 checkpoint 还需在 72×4 controlled cross 上以 512/32 复核。教师响应使用 S6–T6 旧解在 S6–T4 下重算与纯 PPO 重求解之差，并按 source family 聚类 bootstrap。

以下门槛已经用户确认：

- 工程硬门槛：可行率 100%，非法动作 0，分数重算误差不超过 `1e-9`，checkpoint 版本严格匹配。
- 固定 72 对：mean exact gap 不超过 0.525%，P90 不超过 1.020%。这是相对 run03 update60 的开发容差，不是教学质量阈值。
- teacher-only：正增益率至少 50%，平均增益至少 `0.000055`，source-family 聚类 bootstrap 95% CI 下界大于 0。
- 回归保护：student-only 正增益率至少 54%，conflict 至少 89%。
- 多个 arm 均合格时：先最大化 teacher-only 平均增益，再最大化固定 72 对 mean J，再最小化 P90 gap，最后按 A、B、C 偏好更低复杂度。

这些都是当前探索性数据上的开发门槛，不能替代第9项新数据和第10项独立确认，也不能解释为教学有效性阈值。

## 5. 当前停止点

协议已经确认，但本轮仍只执行 `--validate-only`、语法检查和测试。不得创建 A/B/C 训练结果目录，不得启动任何 update；训练须等待单独的启动指令。
