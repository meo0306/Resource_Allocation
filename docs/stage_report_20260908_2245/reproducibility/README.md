# 复现和统计口径

汇总脚本只读历史数据，输出到指定新目录，不调用项目训练和求解器。

```powershell
# 项目根目录下运行；将路径替换为你希望保存的新输出目录
.\.venv\Scripts\python.exe docs\stage_report_20260908_2245\reproducibility\build_evidence.py --project-root D:\project\Resource_Allocation --output-dir D:\project\Resource_Allocation\results\stage_report_rebuild
.\.venv\Scripts\python.exe docs\stage_report_20260908_2245\reproducibility\make_figures.py --output-dir D:\project\Resource_Allocation\results\stage_report_rebuild --qa-path D:\project\Resource_Allocation\results\stage_report_rebuild\figure_contact_sheet.png
```

这两条命令重建数据表、绘图输入和8组图，不重建人工撰写的汇报正文与审批稿。历史来源必须保留在项目内；当前包并未复制4500个大型JSON或训练checkpoint。

汇总使用pandas/numpy；补绘使用项目Python3.11.9、pandas3.0.3、matplotlib3.10.9及Windows微软雅黑字体。报告制作没有成功安装新包，也没有改变项目虚拟环境。原始实验参数以保存的run_metadata为准。

## 统计定义

- J越高越好。0.30F_E+0.25F_S+0.25F_T+0.10F_G−0.10V_soft。
- absolute gap = J*−J；relative gap (%) = 100×(J*−J)/J*，先逐实例计算后平均。
- runtime单位秒。675例测试保持原计时，45例Gurobi采用构建加求解调用复测值；不跨批次拼接时间。
- 三种子SD采用ddof=0。新表中的单实例分数SD采用ddof=1；历史表中对应SD为ddof=0，因此末位或小幅差异是定义差异，不是实验更新。
- 原有95% bootstrap CI使用10000次实例重采样，没有按底层身份/重复场景聚类。报告明确了这一局限，不新增独立性或因果性结论。
- 训练曲线横轴为update；50-update平滑只用于阅读。历史主图保留原绘图口径；补充图同时展示原始点与移动平均。
- 类别组成：先算每个实例的类别比例，再在拓扑内平均。规模图的误差线为min–max，不是置信区间。
- 代表算例按uniform pilot样本N的中位规模选择，不根据算法优劣选例。

## 历史重建边界

Git没有9月8日独立代码提交快照；当前代码包含后续版本修改。证据以9月8日及以前保存的日志、CSV、JSON、checkpoint元数据和会话为主。新图与新统计均标记为“本次重新整理”，不能写成当时已制作的图。

当前方案原文的SHA-256保存在review/original_sha256.txt。批准后修改前需要再次核对该哈希。原文备份为字节级复制，不应将修改提案误写为已应用版本。
