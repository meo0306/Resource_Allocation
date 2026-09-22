# 精确求解与混合精炼实施状态（2026-09-08）

## 环境状态

- 项目解释器：`.venv\Scripts\python.exe`；
- Gurobi Python API：`.venv\Lib\site-packages\gurobipy`；
- Gurobi 版本：12.0.3；
- 本机许可证已成功创建模型；
- 不再需要临时启动器或外部 `PYTHONPATH`。

## 已完成

- Gurobi 精确/有界模型；
- tiny instance 暴力枚举一致性校验；
- 当前局部搜索的规范名称与兼容别名；
- PPO+4/8/16/32 accepted moves 短程精炼；
- 45 算例 validation 分层选择、120/600 秒分级求解；
- 逐实例保存、逐拓扑汇总和混合预算门禁报告。

## 验证结果

- 全项目回归：`68 passed`；
- Gurobi 专项：`3 passed`；
- 直接 `.venv` 真实 validation smoke：bottleneck/group1，N=38，scenario=`interp-0020`；
- PPO / PPO+4 / 单点局部搜索 / Gurobi：`0.703834 / 0.705769 / 0.705835 / 0.706896`；
- 直接环境中的对应时间：`0.079 / 0.143 / 0.418 / 0.032` 秒；
- Gurobi 状态 `optimal`，MIP gap `0`，目标重算误差 `1.11e-16`；
- 四种方法全部硬可行。

该单实例仅证明流程和目标模型一致。大规模精确可解性、PPO 的真实 optimality gap 和混合预算仍需由 45 算例 validation pilot 决定。
