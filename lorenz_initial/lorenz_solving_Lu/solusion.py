import torch
from data_pre import data, sign
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


# 定义洛伦兹方程
def lorenz(state):
    # 初始化参数
    params = torch.tensor([1., 1., 1.], requires_grad=True)

    beta = params[0]
    rho = params[1]
    sigma = params[2]
    dx = sigma * (state[1] - state[0])
    dy = state[0] * (rho - state[2]) - state[1]
    dz = state[0] * state[1] - beta * state[2]
    return dx, dy, dz


# 设置学习率

learning_rate = 0.001
state0 = torch.tensor([1., 1., 1.], requires_grad=True)
t = np.arange(0, 40, 0.01)
dt = 0.01
# 求解洛伦兹方程
temp = state0
states = torch.tensor([])
for i in range(len(t)):
    dx0, dy0, dz0 = lorenz(temp)
    temp = [state0[0] + dx0 * dt, state0[1] + dy0 * dt, state0[2] + dz0 * dt]
    states = torch.cat((states, )
# 绘制
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[:, 0].detach().numpy(), states[:, 1].detach().numpy(), states[:, 2].detach().numpy())
plt.title('initial chart')
plt.show()

# 参考
# for t in np.arange(0, 40, 0.01):
#     loss = 0.
#
#     for i in range(len(data) - 1):
#         x, y, z, t_sign = data[i]
#         dt = 0.01
#         dx_pred, dy_pred, dz_pred = lorenz(x, y, z, params)
#         dx_true, dy_true, dz_true, _ = data[i + 1] - data[i]
#
#         loss += ((dx_pred * dt) - dx_true) ** 2 + ((dy_pred * dt) - dy_true) ** 2 + ((dz_pred * dt) - dz_true) ** 2
#
#     loss.backward()
#
#     with torch.no_grad():
#         params -= learning_rate * params.grad
#         params.grad.zero_()
#
# print(params)
