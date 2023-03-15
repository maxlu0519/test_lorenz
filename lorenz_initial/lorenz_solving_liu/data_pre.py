import torch


arr_all = []
with  open('../Lorenz_data/dataset.txt') as f:
    temp = f.readlines()
    print(temp)
    f.close()

for i in temp:
    i = i.strip('[]\n ')  # 去掉字符串两端的中括号和换行符
    arr = [float(x) for x in i.split()]  # 用列表推导式，将列表中的每个元素转换成浮点数
    print(arr)  # 输出结果
    arr_all.append(arr)
print(arr_all)

sign = torch.tensor(arr_all[0])
data = torch.tensor(arr_all[1:])
