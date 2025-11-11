import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 历史数据加载（OMNI样本，小时级）
data = {
    'bz': np.array([2.2, 3.3, -0.9, -1.1, -1.2], dtype=np.float32),  # Bz (nT)
    'n': np.array([7.7, 8.3, 8.2, 8.8, 8.5], dtype=np.float32),      # 密度 (cm⁻³)
    'v': np.array([366.0, 367.0, 359.0, 364.0, 362.0], dtype=np.float32),  # 速度 (km/s)
    'dst': np.array([-9, -8, -9, -9, -13], dtype=np.float32)         # Dst (nT, 目标)
}
features = np.stack([data['bz'], data['n'], data['v']], axis=1)  # (5, 3)
target = data['dst']

# Min-Max归一化
min_f, max_f = np.min(features, axis=0), np.max(features, axis=0)
features_scaled = (features - min_f) / (max_f - min_f + 1e-8)
min_t, max_t = np.min(target), np.max(target)
target_scaled = (target - min_t) / (max_t - min_t + 1e-8)

# 序列准备 (seq_len=3, 预测下一个Dst)
seq_len = 3
X, y = [], []
for i in range(len(features_scaled) - seq_len):
    X.append(features_scaled[i:i+seq_len])
    y.append(target_scaled[i+seq_len])
X = torch.tensor(np.array(X))  # (2, 3, 3)
y = torch.tensor(np.array(y))  # (2,)

# LSTM模型 (hidden_size=50, 文献参数)
class LSTMDst(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = LSTMDst()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练 (100 epochs, 记录损失)
epochs = 100
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output.squeeze(), y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# 预测所有序列 (反归一化)
with torch.no_grad():
    preds_scaled = model(X).squeeze().numpy()
    preds = preds_scaled * (max_t - min_t) + min_t
    actual = y.numpy() * (max_t - min_t) + min_t

# 预测下一个Dst (最后序列)
last_seq = torch.tensor(features_scaled[-seq_len:].reshape(1, seq_len, 3))
pred_next_scaled = model(last_seq).item()
pred_next_dst = pred_next_scaled * (max_t - min_t) + min_t

# 可视化展示 (Matplotlib: 损失曲线 + 预测对比)
plt.figure(figsize=(10, 4))
# 子图1: 训练损失
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)

# 子图2: Dst预测 vs 实际
plt.subplot(1, 2, 2)
time_steps = np.arange(len(actual))
plt.plot(time_steps, actual, 'b-o', label='Actual Dst (nT)')
plt.plot(time_steps, preds, 'r--s', label='Predicted Dst (nT)')
plt.scatter(len(actual), pred_next_dst, c='g', s=100, marker='*', label=f'Next Pred: {pred_next_dst:.1f} nT')
plt.title('Dst Index Prediction')
plt.xlabel('Time Step (Hour)')
plt.ylabel('Dst (nT)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('lstm_space_weather_viz.png', dpi=300, bbox_inches='tight')  # 保存高清PNG，便于PPT嵌入
plt.show()  # 显示图窗 (可选)

# 输出结果
print(f"最终训练损失: {losses[-1]:.4f}")
print(f"预测Dst序列: {preds}")
print(f"实际Dst序列: {actual}")
print(f"下一个小时预测Dst: {pred_next_dst:.1f} nT (负值表示地磁扰动增强)")