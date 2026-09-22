# 方案4 Phase A：动态偏好调整协议

状态：用户已确认。版本：`dynamic_preference_adjustment_v1`。

## 目标与边界

检验冻结 PPO 能否从共同旧分配启动，在偏好小幅漂移或突变后形成时间—质量非支配优势。实验仅使用现有四拓扑开发数据，不训练 PPO，不修改冻结目标、checkpoint、Stage6A 或 Stage7 产物，不加入 assignment 切换成本。

共同旧解定义为每个内容实例在无扰动 `S6–T6` 画像下，由冻结主线 `legacy_separate_v1`、run03 update60、确定性 512/32 生成的 PPO 解。所有旧解启动方法使用同一 assignment。

## 数据与动态条件

从 calibration core 中按 `size_group × topology` 每层选择 `original_id` 字典序第一条，共 36 个实例。初始画像为 `S6–T6`，变化方向为：

- student-only：`S6–T6 → S4–T6`；
- teacher-only：`S6–T6 → S6–T4`；
- conflict：`S6–T6 → S1–T4`。

每个方向分别使用 25% 线性漂移和 100% 突变，共 `36×3×2=216` 个实例—动态条件。画像不加随机扰动，所有向量显式保存在冻结配置和结果快照中。

## 方法与计时

比较旧解不动、scalarized greedy、PPO 从 effect-only 初解重启、PPO 从共同旧解启动、16 次 best-improvement local search 从共同旧解启动、Gurobi 重建模型加默认标量初解、Gurobi 重建模型加共同旧 PPO 解 MIP start。

在线 Gurobi 档为固定 seed、单线程、1 秒。质量参考为固定 seed、单线程、最多 5 秒；已证最优时使用最优值，否则使用 certified bound 计算有界 gap。所有方法记录偏好条件化时间、求解时间和两者之和。Gurobi 当前两档都会重建模型；不得将 MIP start 描述为持久化模型复用。

## Phase B 门槛

PPO 旧解启动必须同时满足：

1. 在六个方向—幅度单元中的至少四个进入 gap—时延 Pareto 前沿；
2. 相比 PPO 重启，中位时延至少下降 20% 且 mean gap 退化不超过 0.1 个百分点，或者 mean gap 至少改善 0.1 个百分点且时延不增加；
3. 总体上不被 scalarized greedy、旧解局部搜索或旧解 Gurobi warm start 在 mean bounded gap 和中位端到端时延上同时支配。

任一条件失败即停止在 Phase A，不实现持久化 Gurobi 模型复用，不扩展至 72 个实例。teacher-only 必须单列，不能由 student-only 或 conflict 汇总掩盖。
