# 问题建模与 Gurobi 精确求解说明

状态截止：2026 年 9 月 8 日 22:45（北京时间）。文档整理与独立核验日期：2026 年 10 月 2 日。

主文档：`问题建模与Gurobi精确求解说明.tex`，独立中文 LaTeX 文件，不依赖外部图片或 BibTeX。包含完整符号表、输入构造、硬约束与软惩罚、完整目标、Gurobi 各项表达及等价证明、模型规模、上下界和历史证据。可使用支持 `ctex`、Fandol 中文字体的 XeLaTeX 环境编译两遍以更新目录及交叉引用。

## 核心核对结论

- 固定参数中的乘积、绝对值和截断在求解前计算，不属于决策非线性。
- Gurobi 使用二元分配、整数计数、线性辅助约束、11 个 PWL 熵约束和 12 个二次目标项，不是纯 MILP。
- PWL 覆盖每个可行整数计数，结合历史参数条件，在所有可行分配上精确表达实际平滑截断熵。
- 熵下限和占比上限是软惩罚，不是硬可行性约束。
- 当前代码存在后续新增接口和容差设置，文档没有将其倒推为历史事实。

## 随附文件

- `mathematical_audit.json`：315 个历史分配的独立重算和 45 份日志结构核对结果。
- `audit_formulation.py`：独立核验脚本，依赖 Python 与 NumPy；不训练、不调用 Gurobi。默认从 `D:\project\Resource_Allocation` 读取数据，并把核验结果写到脚本所在目录。历史结果内的 `instance_path` 仍为绝对路径，异机运行需要恢复这些输入路径。
- `source_hashes.json`：本次读取的源码和部分历史输入文件内容指纹，不是历史完整代码快照。
- `document_check.json`：环境嵌套、花括号、引用等静态检查及编译状态。

复核命令示例：

```powershell
python audit_formulation.py --project D:\project\Resource_Allocation
```

## 编译状态

已请求在 Codex 内置 LaTeX 编辑器中打开源文件。内置编译器连续两次返回 `Unable to find standard directories for platform`，未进入可用的源文件诊断阶段。因此本次未交付或声称验证过 PDF；静态检查不代替编译和视觉排版检查。源文件保留可编辑，可在可用的 XeLaTeX 环境继续编译。

`docs\scheme_final.md` 没有修改。本文是独立建模说明，不构成对该原方案文档修改的自动批准。
