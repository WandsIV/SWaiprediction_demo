# SWaiprediction_demo
# LSTM 空间天气预测模型

## 概述

这个代码实现了一个简单的LSTM的空间天气预测模型，使用 PyTorch 框架处理太阳风历史数据（Bz 磁场分量、n 密度、V 速度），预测地磁指数 Dst（负值表示地磁扰动增强）。模型参考 2025 年文献《AI-Driven Space Weather Forecasting: A Comprehensive Review》，参数包括隐藏层大小 50，序列长度 3（1 小时提前预测），适用于演示太阳风暴预报的准确性。

## 环境要求

- Python 版本：3.8 或更高（推荐 3.10，使用 Anaconda 管理虚拟环境）。
- 操作系统：Windows、macOS 或 Linux。
- 依赖库：PyTorch（~2.1.0）、NumPy、Matplotlib（用于可视化）。

## 安装依赖

1. 克隆或下载项目文件（`ai_lstm_viz.py`）。
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
     python ai_lstm_viz.py
     ```
     - 输出：控制台打印训练损失、预测结果；弹出 Matplotlib 图窗；生成 `lstm_space_weather_viz.png`（PNG 文件）。
   
   - **VSCode 方式**：
     - 打开 VSCode > File > Open Folder（新建 demo 文件夹） > 粘贴代码到 `ai_lstm_viz.py` > 右上角点击 “Run Python File” 按钮（需安装 Python 扩展）。
     - 或按 F5 进入调试模式。

3. **预期输出**：
   - **训练损失**：约 0.0010（MSE，收敛快）。
   - **预测 Dst 序列**：例如 [-9.2, -12.8] nT（接近实际 [-9, -13] nT）。
   - **下一个小时预测**：约 -12.5 nT（负值表示风暴迹象，误差 <1 nT）。
   - **可视化**：双子图 PNG 文件。
     - 左侧：训练损失曲线（Epoch vs MSE，下降趋势展示收敛）。
     - 右侧：实际 Dst（蓝实线） vs 预测 Dst（红虚线） + 下一个预测点（绿星）。

## 参考文献

- 《AI-Driven Space Weather Forecasting: A Comprehensive Review》（arXiv, 2025）
- OMNI 数据集：NASA/GSFC，Wind 航天器原位太阳风测量（2005-2023）

## 联系和碎碎念

其实这些只是一个选修课演示，老实说我都不知道这些东西会不会最后用到我的pre里。我真的很想一出是一出！
我下次再也不熬夜了，我要咖啡因中毒了。或者说咖啡因免疫。
好险我没有找其他人合作，我的拖延症会害死所有人的。但是我总能在ddl前准备完何尝不是一种天赋，嘿嘿。
联系方式[fangew410@outlook.com]

---

*最后更新：2025 年 11 月 12 日*