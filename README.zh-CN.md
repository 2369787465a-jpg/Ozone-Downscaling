# 粤港澳大湾区臭氧降尺度研究

[English](README.md) · [项目报告 PDF](report.pdf) · [作者与分工](AUTHORS.md)

本项目结合 GEOS-Chem 模拟、ERA5 气象数据和地面站观测，研究站点臭氧浓度的机器学习偏差订正。代码包含日均场预处理、时空匹配、XGBoost 训练、多模型对比、模型置信集（MCS）分析及 SHAP 解释。

报告题目为 **A Physics-Consistent Machine Learning Framework for High-Resolution Ozone Downscaling in the Greater Bay Area**，作者为 **Wentao Wang、Lifeng Han、Chenxi Pan，三人同等贡献**。Lifeng Han 负责代码实现与图表制作。

<p align="center">
  <img src="assets/model-comparison.png" width="700" alt="原项目图：随机森林、XGBoost、MLP 和 DNN 在香港及大湾区两种场景下的对比">
</p>

*图片直接来自原项目输出。报告中大湾区 XGBoost 的测试集 R2 约为 0.86，采用随机站点日记录划分；这是原报告记载的结果，本次整理未重新运行完整实验。*

## 代码结构

| 阶段 | 文件 | 输出 |
| --- | --- | --- |
| 日均场预处理 | `src/preprocessing.py` | `cache/` 内的 GEOS-Chem 与 ERA5 NetCDF |
| 站点时空匹配 | `src/data_matcher.py` | `cache/final_dataset_full.csv` |
| 特征工程 | `src/features.py` | 污染物、气象、模拟臭氧及日历变量 |
| 偏差订正 | `main.py`、`src/model.py`、`src/analysis.py` | XGBoost 模型、相关系数/RMSE 和诊断图 |
| 多模型对比与解释 | `generate.py` | RF/XGBoost/MLP/DNN 对比、MCS 表及 SHAP 图 |

## 环境与数据

建议在独立的 Python 3.11 或 3.12 环境运行。部分 Windows netCDF4 构建无法打开中文路径，遇到此情况请将项目及数据放在纯英文路径下。原项目未保存完整依赖锁定文件，`requirements.txt` 列出所需依赖，已补充 MCS 使用的 `arch`。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux 可使用 `source .venv/bin/activate` 激活环境。

原始数据压缩包、NetCDF、站点表、训练模型和缓存未纳入仓库。请按 [data/README.md](data/README.md) 准备兼容且有权使用的输入，放入 `data/`，或指向已有数据目录：

```powershell
$env:OZONE_DATA_DIR = 'F:\AI_math\data'
python scripts/check_data.py
```

macOS/Linux 可用 `export OZONE_DATA_DIR=/path/to/prepared-data`。此检查只检查文件存在性，不进行训练，也不代替变量、坐标和单位核对。

## 运行

```bash
# 预处理、观测匹配、XGBoost 训练与诊断
python main.py

# main.py 生成匹配表后，再运行多模型实验与图表生成
python generate.py
```

`main.py` 将模型保存为 `models/bias_correction_model_full.pkl`，并交互显示 Matplotlib 诊断图；无图形界面的环境可设置 `MPLBACKEND=Agg`。交互图关闭后脚本继续执行。`generate.py` 的图片保存在仓库根目录的 `Paper_Ultimate_Final_Tuned/` 中。完整调参、并行训练与 SHAP 计算可能耗时较长。

<details>
<summary>查看原项目 SHAP 汇总图</summary>

<img src="assets/shap-summary.png" width="650" alt="大湾区 XGBoost 臭氧模型的原项目 SHAP 汇总图">

</details>

## 实验范围与复现说明

本次公开整理保留原始建模流程，只补充 `arch`、输入文件检查和可配置数据目录，并使图表输出路径不依赖当前工作目录；没有重新调参或改写报告实验。

代码接收已经准备好的网格数据，不包含原始下载及最初的 0.01 度高分辨率重网格流程。`generate.py` 使用随机 80/20 站点日记录划分和三折模型选择；`main.py` 同样先打乱记录，因此其中的 `TimeSeriesSplit` 实际运行于打乱后的顺序，**不能视为按时间先后进行的预报验证**。目标异常值过滤发生在划分之前，也未实现独立站点留出验证。报告指标应按这一具体实验设置理解。臭氧换算保留原代码的质量混合比假设，使用前须核实单位；完整 SHAP 图组还需湿度、辐射辅助输入，详见数据说明。

公开 PDF 保留全部作者、同等贡献声明和研究内容，已移除学号。原数据及派生站点表的再分发授权未确认，因此未上传。完整复现需要自行取得兼容数据。

## 许可

Python 代码及仓库说明文档使用 [MIT 许可](LICENSE)。研究报告和原始科研图片保留作者权利，不随代码许可重新授权；第三方数据和依赖遵循各自的使用条款。

