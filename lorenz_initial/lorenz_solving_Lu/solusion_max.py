import torch
import torch.nn as nn
from torch.autograd import gradcheck
from data_pre import data, sign
import numpy as np

torch.autograd.set_detect_anomaly(True)


# 定义洛伦兹方程模型类
class LorenzModel(nn.Module):
    def __init__(self):
        super(LorenzModel, self).__init__()
        # 初始化参数为可学习的张量
        self.sigma = nn.Parameter(torch.tensor([1.]))
        self.rho = nn.Parameter(torch.tensor([1.]))
        self.beta = nn.Parameter(torch.tensor([1.]))
        self.stats = nn.Parameter(torch.tensor([1., 1., 1., 0]))

    def lorenz(self, x, y, z):
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return dx, dy, dz

    # 由参数获取预测值
    def forward(self, t, dt=0.01):
        data_lorenz = torch.empty((len(t), 4))
        data_lorenz[0] = self.stats
        for i in range(len(t) - 1):
            dx, dy, dz = self.lorenz(data_lorenz[i][0], data_lorenz[i][1], data_lorenz[i][2])
            data_lorenz[i + 1][0] = data_lorenz[i][0] + dx * dt
            data_lorenz[i + 1][1] = data_lorenz[i][1] + dy * dt
            data_lorenz[i + 1][2] = data_lorenz[i][2] + dz * dt
            data_lorenz[i + 1][3] = dt * (i + 1)
        return data_lorenz


# 创建模型实例
model1 = LorenzModel()
# 定义损失函数
criterion = nn.MSELoss()
# 设置时间范围

t = np.arange(0, 40, 0.01)
ray = len(t)
# 模拟数据


# 定义优化器
optimizer = torch.optim.SGD(model1.parameters(), lr=0.1)

loss_history = []

for epoch in range(10):
    print(f"第{epoch + 1}轮训练")
    loss_sum = 0
    data_lorenz = model1(t)
    for i in range(len(data) - 1):
        for j in range(len(data_lorenz) - 1):
            if data[i][3] == data_lorenz[j][3]:
                # 使用MSE损失函数
                loss = criterion(data[i][:3], data_lorenz[j][:3])
                print(f"匹配到的t为{j * 0.01}")
                print(f"当前匹配loss为{loss.item()}")
                # 计算梯度
                loss.backward()
                # 更新模型参数
                optimizer.step()
                print(f"更新后的参数为:"
                      f"sigma:{model1.sigma.item()}, "
                      f"rho:{model1.rho.item()}, "
                      f"beta:{model1.beta.item()}, "
                      f"stats:{model1.stats.item()}")
                loss_sum += loss.item()
    # 清除梯度信息
    optimizer.zero_grad()

    if epoch % 1 == 0:
        print(f"第{epoch + 1}轮训练总loss为：{loss_sum}")

        # 记录损失函数值历史信息
    loss_history.append(loss_sum / len(t))
    # 训练完成，显示结果
print(f"sigma:{model1.sigma.item()}, rho:{model1.rho.item()}, beta:{model1.beta.item()}, stats:{model1.stats}")
