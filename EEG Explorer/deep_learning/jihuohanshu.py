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
    # 有些网络层在训练和测试时是不一样的，比如DropOut、BatchNorm，所以需要显示告诉是在训练还是测试模式。
    # 训练模式
    model.train()
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

    # model.eval() # 测试模式 
    # # 测试模式下，不需要计算梯度
    # with torch.no_grad():
    #     for x, y in val_loader:
    #         prediction = model(x)


prediction = torch.tensor([
    [2.1, 0.3, 0.5],
    [0.2, 3.5, 0.1],
    [0.1, 0.4, 2.8],
    [1.2, 2.3, 0.5]
])

y = torch.tensor([0, 1, 2, 1])

prediction_result=prediction.argmax(dim=1) #在每一行中，返回每行最大值的索引。
correct=(prediction_result==y)
accuracy=correct.float().mean() # .float()将tensor转换成float类型（true->1,false->0），.mean()求平均值。
print(prediction_result)
print(correct)
print(accuracy)
