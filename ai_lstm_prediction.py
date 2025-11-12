import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  # Matlab-style plotting
from torch.utils.data import Dataset, DataLoader

# 1. 数据集类：合成演示数据（随机序列，模拟40维特征）
# 注意：真实数据需从SDO/HMI SHARP + GOES目录加载，并归一化（z-score for 磁场，min-max for 历史）
class FlareDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, num_features=40):
        # 合成数据：(num_samples, seq_len, num_features)，标签：0(无耀斑)/1(耀斑)，随机生成
        # 真实中：正样本为耀斑前24h数据，负为其他；使用零填充补全
        self.data = torch.randn(num_samples, seq_len, num_features)  # 模拟归一化特征
        self.labels = torch.randint(0, 2, (num_samples,))  # 随机二元标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx].long()  # Long for CrossEntropyLoss

# 2. 模型类：简化LSTM + Attention（参考论文Section 3.2 & 图3-4）
class SimpleLSTMSolarFlare(nn.Module):
    def __init__(self, input_size=40, hidden_size=10, num_layers=1, num_classes=2):
        super(SimpleLSTMSolarFlare, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 简单注意力：为每个时间步计算权重（参考公式15-18的变体）
        self.attention = nn.Linear(hidden_size, 1)
        # 全连接层：参考论文 (hidden -> 200 -> 500 -> 2)
        self.fc1 = nn.Linear(hidden_size, 200)
        self.fc2 = nn.Linear(200, 500)
        self.fc3 = nn.Linear(500, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # LSTM前向：输出 (batch, seq_len, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)
        # 注意力权重：(batch, seq_len, 1)，softmax over seq_len
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # 加权上下文：(batch, hidden_size)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        # FC层
        out = self.relu(self.fc1(context))
        out = self.relu(self.fc2(out))
        out = self.softmax(self.fc3(out))  # (batch, 2) 概率
        return out, attn_weights, lstm_out  # 返回额外输出用于可视化

# 3. 新增：可视化函数（Matlab-style plots）
def visualize_predictions(model, test_sample, device, save_path='lstm_flare_prediction_viz.png'):
    model.eval()
    with torch.no_grad():
        pred_probs, attn_weights, lstm_out = model(test_sample)
        predicted_class = torch.argmax(pred_probs, dim=1).item()
        flare_prob = pred_probs[0][1].item()
        
        # 提取数据（CPU for plotting）
        test_sample_cpu = test_sample.cpu().numpy()[0]  # (seq_len, features)
        attn_weights_cpu = attn_weights.cpu().numpy()[0].flatten()  # (seq_len,)
        lstm_out_cpu = lstm_out.cpu().numpy()[0]  # (seq_len, hidden_size)
        pred_probs_cpu = pred_probs.cpu().numpy()[0]  # [no_flare, flare]
        
        # 创建Matlab-style figure (3 subplots)
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), facecolor='white')
        time_steps = np.arange(1, 11)  # 10 time steps (hours)
        
        # 子图1: 输入序列示例特征 (e.g., feature 0 as USFLUX)
        axs[0].plot(time_steps, test_sample_cpu[:, 0], 'b-o', linewidth=2, markersize=6, label='Normalized USFLUX (Feature 0)')
        axs[0].set_title('Input Time Series: Example Magnetic Feature', fontsize=14, fontweight='bold')
        axs[0].set_xlabel('Time Steps (Hours)', fontsize=12)
        axs[0].set_ylabel('Normalized Value', fontsize=12)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # 子图2: Attention Weights (bar plot, like Matlab bar)
        axs[1].bar(time_steps, attn_weights_cpu, color='orange', alpha=0.7, edgecolor='black')
        axs[1].set_title('LSTM Attention Weights per Time Step', fontsize=14, fontweight='bold')
        axs[1].set_xlabel('Time Steps (Hours)', fontsize=12)
        axs[1].set_ylabel('Attention Weight', fontsize=12)
        axs[1].grid(True, alpha=0.3)
        
        # 子图3: Prediction Probabilities (bar plot)
        classes = ['No Flare (0)', 'Flare (1)']
        axs[2].bar(classes, pred_probs_cpu, color=['green', 'red'], alpha=0.7, edgecolor='black')
        axs[2].set_title(f'Prediction Output (Class: {predicted_class})', fontsize=14, fontweight='bold')
        axs[2].set_ylabel('Probability', fontsize=12)
        for i, v in enumerate(pred_probs_cpu):
            axs[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高分辨率保存
        plt.show()
        print(f"Visualization saved to {save_path}")
        print(f"Flare probability: {flare_prob:.4f}")

# 4. 演示训练与预测
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 实例化模型
    model = SimpleLSTMSolarFlare().to(device)
    criterion = nn.CrossEntropyLoss()  # 真实中：加权以处理不平衡 (参考公式19)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 数据加载器（batch_size=32，模拟论文中N序列训练）
    train_dataset = FlareDataset(num_samples=1000)  # 1000个序列样本
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 训练5个epoch（演示，真实中用2010-2013训练集）
    print("Training for 5 epochs...")
    for epoch in range(5):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(batch_x)  # 只用outputs训练
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/5, Average Loss: {avg_loss:.4f}')

    # 测试预测：单个序列样本
    test_sample = torch.randn(1, 10, 40).to(device)  # 模拟新AR 10h序列
    visualize_predictions(model, test_sample, device)  # 新增：生成可视化

    print("\nNote: This is a demo with synthetic data. For real use, load SHARP data via SunPy and GOES labels.")