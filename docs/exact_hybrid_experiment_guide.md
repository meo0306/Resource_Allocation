# Gurobi 精确界与 PPO 混合精炼实验手册

## 环境与验证

本项目使用同一个虚拟环境运行 PPO、PyTorch 和 Gurobi。当前许可证对应 Gurobi 12，因此依赖限制为 `gurobipy>=12,<13`。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-exact.txt
.\.venv\Scripts\python.exe -m pytest tests\test_exact_and_hybrid.py -q
```

不需要切换 Python 环境，也不需要设置额外的 `PYTHONPATH`。

## 算法名称

历史代码中的 `local_search_best_improvement` 与论文中的“单点替换最优改进局部搜索”是同一个算法。规范别名入口为：

```python
from pa_moap_rl.solvers.one_point_local_search import (
    one_point_best_improvement_local_search,
)
```

旧入口仅为兼容历史脚本和 CSV 保留。不得把该算法称为“全局局部搜索”或“全局最优搜索”。

## 45 算例 validation pilot

默认固定抽取 `5 类拓扑 × 9 个规模组 × 每格 1 个中位 N 算例 = 45`。实例仍按原 manifest index 与固定 validation scenario bank 配对，不读取测试集。

```powershell
.\.venv\Scripts\python.exe -m pa_moap_rl.experiments.compare_exact_hybrid `
  --checkpoint results\formal_joint_seed1_run01\checkpoints\best.pt `
  --output-dir results\exact_hybrid_validation_pilot `
  --initial-time-limit 120 `
  --retry-time-limit 600 `
  --acceptable-gap 0.01 `
  --move-budgets 4 8 16 32
```

分级规则：首轮最多 120 秒；已证最优直接结束；未证最优但 MIP gap 不超过 1% 时保留有效上下界；gap 大于 1% 才以首轮 incumbent 热启动继续 600 秒。“120 秒未证最优”不能表述为“Gurobi 无法精确求解”。

输出包括：

- `paired_results.csv`：统计和绘图用逐实例表；
- `paired_results.jsonl`：包含完整 assignment；
- `gurobi_logs/`：逐算例日志；
- `run_metadata.json`：checkpoint、数据、预算、种子和设备。

## 汇总与混合预算选择

```powershell
.\.venv\Scripts\python.exe -m pa_moap_rl.experiments.summarize_exact_hybrid `
  --input-csv results\exact_hybrid_validation_pilot\paired_results.csv `
  --output-dir results\exact_hybrid_validation_pilot\summary
```

只使用 validation 选择 4/8/16/32 中的预算：所有解必须硬可行；选择恢复至少 50% PPO—完整局部搜索差距、且总时间不超过完整局部搜索 20% 的最小预算。若没有候选同时满足，则报告质量—时间 Pareto 前沿，不使用测试集挑选。

Gurobi 结果中，`objective_incumbent` 是最优值下界，`objective_bound` 是全局上界。若未证最优，应报告 `incumbent <= J* <= bound`。所有 Gurobi assignment 都由 `scoring.py` 重算并记录 `objective_recompute_error`。
