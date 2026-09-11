# A题：药材烘干问题

本仓库用于复现和继续开发 2026 年高教社杯全国大学生数学建模竞赛 A 题。当前代码已经完成问题一、问题二的温度与水分浓度求解、验证、结果导出和论文图表生成；问题三、问题四的建模路线与验收标准记录在报告中，暂不把未运行的结果当作已完成结果。

## 快速开始

建议使用 Python 3.12。进入仓库根目录后运行：

```text
python -m pip install -r requirements.txt
python main.py
```

`main.py` 是唯一入口。程序读取 `附件/附件1.xlsx`，使用固定的 320 个径向网格和确定性求解设置，在 `outputs/` 中重建表格、验证数据和图表。

## 仓库结构

```text
附件/                         题目附件与原始模板（只读）
data_processing.py            数据读取、插值、采样和结果导出
model.py                      圆柱热湿传递模型与 FVM--BDF/Radau 求解
validation.py                 网格、算法、守恒、值域和单调性验证
plot_results.py               从统一结果对象生成图表和图源数据
main.py                       唯一运行入口
fill_tables_pdf.py            将表1--表4结果排入题目 PDF
docs/                         团队可直接阅读的模型方案文档
outputs/tables/               表1--表4的 CSV 结果
outputs/validation/           验证结果 CSV
outputs/figure_data/          图源数据 CSV
outputs/figures/              论文 PNG/PDF 图
outputs/result1.xlsx          问题一完整结果
outputs/result2.xlsx          问题二完整结果
output/pdf/                   已填入表1--表4的题目 PDF
```

## 已有结果

- 问题一：固定热物性、非线性水分扩散，有限体积法 + 方法线 + BDF；用 CN--Picard、解析基准、网格收敛和水分守恒复核。
- 问题二：附录 3 变物性热湿耦合，重新从题给初始状态求解 0--3 h；用 Radau 和网格收敛复核。
- `outputs/tables/table_01_q1_temperature.csv` 至 `table_04_q2_moisture.csv` 分别对应题目表1至表4。
- `output/pdf/A题_表1至表4已填.pdf` 是保持原题版式、已填入表1至表4四位小数结果的成品。

## 文档与结果口径

- `docs/A题_综合优化最终建模报告.md`：模型方案冻结稿，说明四问关系、方程、算法、验证标准和后续实施边界。
- `outputs/A题_综合优化最终建模报告.md`：结合当前代码结果的综合报告，包含问题一、问题二的数值结果以及问题三、问题四的实施路线。

## 队友接手流程

1. 克隆仓库并安装 `requirements.txt` 中的依赖。
2. 先运行 `python main.py`，确认 `outputs/` 能完整重建。
3. 修改前从 `main` 新建个人分支，例如 `feature/q3-drying-time` 或 `feature/q4-shrinkage`。
4. 新模型应放在现有责任文件中，保持“读取数据 → 求解 → 验证 → 导出”的顺序；不要改写 `附件/` 原始文件。
5. 每次提交同时保留结果 CSV、图源数据和验证输出；论文中的数字只引用这些统一输出。
6. 合并前至少检查：代码可运行、结果可复现、约束满足、图表未裁切、单位和四位小数口径一致。

## 当前 Git 基线

`main` 分支保留了已有的模型与图表提交历史。后续工作建议以小而清晰的提交推进，例如：

```text
feat: 完成问题三全域终点判定
feat: 加入问题四收缩域模型
fix: 修正结果导出或图表标注
docs: 更新模型报告与复现说明
```

不要对共享分支强制推送；需要回退时使用新的修复提交，并在提交说明中写清影响的题目小问和输出文件。

