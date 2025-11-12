# SWaiprediction_demo
# LSTM 空间天气预测模型

## 概述

本项目是一个基于 PyTorch 的简单演示模型，用于太阳耀斑（Solar Flare）预测，参考 2019 年发表在《The Astrophysical Journal》上的论文《Predicting Solar Flares Using a Long Short-term Memory Network》（作者：Hao Liu 等）。该模型使用长短期记忆网络（LSTM）结合注意力机制（Attention），处理太阳活动区（Active Region, AR）的时序数据，预测未来 24 小时内是否会发生特定类别的耀斑（例如 ≥M 类）

## 环境要求

- Python 版本：3.8 或更高（推荐 3.10，使用 Anaconda 管理虚拟环境）。
- 操作系统：Windows、macOS 或 Linux。
- 依赖库：PyTorch（~2.1.0）、NumPy、Matplotlib（用于可视化）。

## 安装依赖

1. 克隆或下载项目文件（`ai_lstm_prediction.py`）。
2. 创建虚拟环境（推荐）：
   ```
   conda create -n lstm_demo python=3.10
   conda activate lstm_demo
   ```
3. 安装依赖：
   ```
   pip install torch numpy matplotlib
   ```
   - 如果使用 Conda：`conda install pytorch numpy matplotlib -c pytorch`。

## 使用方法

1. **准备数据**：代码直接包含 OMNI 历史样本（1998 年 1 月 1 日 5 小时数据）。若扩展，下载 OMNI CSV 文件（从 NASA OMNIWeb），替换 `data` 字典中的数组。
   
2. **运行代码**：
   - **终端方式**（推荐）：
     ```
     python ai_lstm_prediction.py
     ```
     - 输出：控制台打印训练损失、预测结果；弹出 Matplotlib 图窗；生成 `lstm_space_weather_viz.png`（PNG 文件）。
   
   - **VSCode 方式**：
     - 打开 VSCode > File > Open Folder（新建 demo 文件夹） > 粘贴代码到 `ai_lstm_prediction.py` > 右上角点击 “Run Python File” 按钮（需安装 Python 扩展）。
     - 或按 F5 进入调试模式。

3. **预期输出**：
   - **控制台输出**
   ```
   Using device: cpu
   Training for 5 epochs...
   Epoch 1/5, Average Loss: 0.7123
   ...
   Epoch 5/5, Average Loss: 0.6389
   Visualization saved to lstm_flare_prediction_viz.png
   Flare probability: 0.4877

   Note: This is a demo with synthetic data. For real use, load SHARP data via SunPy and GOES labels.
   ```

   - **可视化图表**
   - **子图 1**：输入时序（示例：USFLUX 特征随 10 小时变化）。
   - **子图 2**：注意力权重条形图（突出模型关注的时序步）。
   - **子图 3**：预测概率条形图（No Flare vs. Flare）。


## 参考文献

   - Liu et al. (2019). *Predicting Solar Flares Using a Long Short-term Memory Network*. The Astrophysical Journal, 877:121. [DOI: 10.3847/1538-4357/ab1b3c](https://doi.org/10.3847/1538-4357/ab1b3c)
   - 数据来源：SDO/HMI SHARP (JSOC)，GOES X-ray 目录 (NCEI)。

## 联系和碎碎念

   基于Grok完成的演示代码。请自由使用。\\
   其实这些只是一个选修课演示，老实说我都不知道这些东西会不会最后用到我的pre里。我真的很想一出是一出！\\
   我下次再也不熬夜了，我要咖啡因中毒了。或者说咖啡因免疫。\\
   好险我没有找其他人合作，我的拖延症会害死所有人的。但是我总能在ddl前准备完何尝不是一种天赋，嘿嘿。\\
   联系方式[fangew410@outlook.com]

---

*最后更新：2025 年 11 月 12 日*