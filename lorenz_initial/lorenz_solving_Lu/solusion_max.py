import torch
import torch.nn as nn
from torch.autograd import gradcheck
from data_pre import data, sign
import numpy as np
import time
torch.autograd.set_detect_anomaly(True)
device = torch.device("cpu")


# 定义洛伦兹方程模型类
class LorenzModel(nn.Module):
    def __init__(self):
        super(LorenzModel, self).__init__()
        # 初始化参数为可学习的张量
        self.sigma = nn.Parameter(torch.tensor([1.]))
        self.rho = nn.Parameter(torch.tensor([1.]))
        self.beta = nn.Parameter(torch.tensor([1.]))
        self.stats = nn.Parameter(torch.tensor([1., 1., 1.]))

    def lorenz(self, x, y, z):
        dx = self.sigma * (y - x)

        dy = x * (self.rho - z) - y

        dz = x * y - self.beta * z

        return dx, dy, dz

    # 由参数获取预测值
    def forward(self, t, dt=0.01):
        print("神经网络开始工作")
        data_lorenz = {}
        data_lorenz[0.] = self.stats.clone()
        for i in range(len(t) - 1):
            if i % 100 == 0:
                print(i)
            dx, dy, dz = self.lorenz(data_lorenz[i / 100][0], data_lorenz[i / 100][1], data_lorenz[i / 100][2])
            xi1 = data_lorenz[i / 100][0] + dx * dt

            yi1 = data_lorenz[i / 100][1] + dy * dt
            zi1 = data_lorenz[i / 100][2] + dz * dt
            stats_i1 = torch.stack((xi1, yi1, zi1), dim=0)

            data_lorenz[(i + 1) / 100] = stats_i1
        print("神经网络工作完成")
        return data_lorenz


# 创建模型实例
model1 = LorenzModel()
model1 = model1.to(device)
# 设置时间范围

t = np.arange(0, 40, 0.01)
ray = len(t)


# 自定义loss函数
def loss(data_train, lorenz_data):
    return (data_train[0] - lorenz_data[0]) ** 2 + (data_train[1] - lorenz_data[1]) ** 2 + (
            data_train[2] - lorenz_data[2]) ** 2


# 定义优化器
optimizer = torch.optim.SGD(model1.parameters(), lr=0.0001)

loss_history = []
start_time = time.time()
for epoch in range(100):
    print(f"第{epoch + 1}轮训练")
    # 清除梯度信息
    optimizer.zero_grad()
    loss_sum = 0
    data_lorenz = model1(t)
    for i in range(len(data) - 1):
        tn = data[i][3].item()
        print("目前匹配t为:", tn)
        if tn in data_lorenz.keys():
            print("匹配成功")
            temp = data_lorenz[tn]
            # 使用自定义损失函数
            loss_t = loss(data[i], temp)
            loss_t = loss_t.to(device)
            print(f"匹配到的t为{tn}")
            print(f"当前匹配loss为{loss_t.item()}")
            # 计算梯度
            loss_t.backward(retain_graph=True)
            print("梯度计算正确")

            loss_sum += loss_t.item()

    # 更新模型参数
    optimizer.step()
    # 清除梯度信息
    optimizer.zero_grad()

    if epoch % 1 == 0:
        print(f"第{epoch + 1}轮训练总loss为：{loss_sum}")
        print(
            f"sigma:{model1.sigma.item()}, rho:{model1.rho.item()}, beta:{model1.beta.item()}, stats:{model1.stats[1].item()}")
        # 记录损失函数值历史信息
    loss_history.append(loss_sum / len(t))
    # 训练完成，显示结果
print(
    f"sigma:{model1.sigma.item()}, rho:{model1.rho.item()}, beta:{model1.beta.item()}, "
    f"stats_x:{model1.stats[0].item()}, stats_y:{model1.stats[1].item()},stats_z:{model1.stats[2].item()}")
