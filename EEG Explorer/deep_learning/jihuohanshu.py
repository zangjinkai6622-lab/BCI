import torch
import torch.nn as nn

# x=torch.tensor([
#     [1.,2.],
#     [3.,4.],
#     [5.,6.],
#     [7.,8.]]
# )
# model=nn.Sequential(
#     nn.Linear(2,3),
#     nn.ReLU(),
#     nn.Linear(3,1),
# )
# y=model(x)
# print(x.shape)
# print(y.shape)

x = torch.tensor([
    [1.],
    [2.],
    [3.],
    [4.]
])

y = torch.tensor([
    [2.],
    [4.],
    [6.],
    [8.]
])

model=nn.Sequential(
    nn.Linear(1,1)
)
# 均方误差MSE
loss_fn=nn.MSELoss()

optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

for epoth in range(1000):    
    # pytorth会保留上次的梯度，清空上一次留下的Gradient。
    optimizer.zero_grad()
    prediction=model(x)
    loss=loss_fn(prediction,y)
    # 反向传播，根据Loss计算模型参数的Gradient。
    loss.backward()
    # 更新参数
    optimizer.step()
    if epoth%10==0:
        print(epoth,loss.item())
