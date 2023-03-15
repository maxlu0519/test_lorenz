import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

content = np.loadtxt('data.txt')
data = content.reshape(100, 4)
data = torch.from_numpy(data)
target = data[:100, :3].clone()
start = torch.ones(1, 3)
Input = torch.cat([start, data[0:99, :3]], dim=0)
Input = Input.to(torch.float64)
# 定义训练设备
device = torch.device("cuda:0")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 3)
        self.sigma = nn.Parameter(torch.tensor([1.]))
        self.rho = nn.Parameter(torch.tensor([1.]))
        self.beta = nn.Parameter(torch.tensor([1.]))

    def lorenz(self, r):
        k = torch.zeros(3, 1).to(device)
        k[0, 0] = self.sigma * (r[1, 0] - r[0, 0])
        k[1, 0] = self.rho * r[0, 0] - r[1, 0] - r[0, 0] * r[2, 0]
        k[2, 0] = r[0, 0] * r[1, 0] - self.beta * r[2, 0]
        return k

    def rk4(self, r, dt=0.1):
        dt = torch.tensor(dt).to(device)
        k1 = self.lorenz(r.T)
        k2 = self.lorenz(r.T + dt * k1 / 2)
        k3 = self.lorenz(r.T + dt * k2 / 2)
        k4 = self.lorenz(r.T + dt * k3)
        return r.T + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.rk4(x)


# 添加tensorboard
write = SummaryWriter("logs_train")
net = Net()
net = net.to(device, dtype=torch.float64)
train_loader = DataLoader(Input, batch_size=100, shuffle=False)
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
criterion = nn.MSELoss()
criterion = criterion.to(device)
# 设置时间
target = target.to(device)
start1_time = time.time()
for epoch in range(1000):
    start2_time = time.time()
    for i, (inputs) in enumerate(train_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = outputs.to(device)
        loss = criterion(outputs, target.T) / 100
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(loss.item())
        end_time = time.time()
        print(f"训练时长总为：{end_time - start1_time}s")
        print(f"训练次数为：{epoch}")
        print(f"该次训练时长总为：{end_time - start2_time}s")
        write.add_scalar("train_loss", loss.item(), epoch)
print(outputs)
print(target.T)
print(net.sigma)
print(net.rho)
print(net.beta)
write.close()