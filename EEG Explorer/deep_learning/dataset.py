import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# x=torch.tensor([1,2,3])
# # y=torch.tensor([[1,2,3],[4,5,6]])
# y=torch.tensor([4,5,6])
# shape是属性,dim是方法,size是方法，不传参数时等于shape,torch.Size([3]) 1 torch.Size([3])
# print(x.shape)
# print(x.dim())
# print(x.size())
# print(y.shape)
# print(y.dim())
# print(y.size(0))

# dataset=TensorDataset(x,y)
# print(dataset[1])
# # DataLoader是把两个Tensor拼接到一起，之后也都一直保持Tensor，若要只想得到数字，使用.item()
# dataloader=DataLoader(dataset,batch_size=2)
# for i in dataloader:   # DataLoader里的基本单位是Batch所以直接遍历就是一个Batch，[tensor([1, 2]), tensor([4, 5])]，[tensor([3]), tensor([6])]
#     print(i)            #这里的i也就是一个Batch即多个sample（样本），
# for x,y in dataloader: # 输出：tensor([1, 2]) tensor([4, 5])，tensor([3])，tensor([6])，DataLoader只要取就是按batch_size取，所以第一次取两个，x，y各有两个
#     print(x)
#     print(y)
# for batch_id,(x,y) in enumerate(dataloader): # enumerate()函数同时取出索引和值，0 tensor([1, 2]) tensor([4, 5])，1 tensor([3]) tensor([6])
#     print(batch_id,x,y)
# x = torch.tensor([
#     [1.,2.],
#     [3.,4.],
#     [5.,6.]
# ])

# w = torch.tensor([
#     [10., 20.],
#     [30., 40.]
# ])
x = torch.tensor([
    [1., 2.],
    [3., 4.],
    [5., 6.],
    [7., 8.]
])

y = torch.tensor([
    0,
    1,
    0,
    1
])
# print(x+y)
# print(x-y)
# print(x*y)
# print(x/y)
# print(torch.matmul(x,w))
# print(x@w)
# 之所以要先实例化Linear是因为Linear是一个类，而不是一方法，写成类便于存储训练参数
# Linear的输入必须是Float类型，因为Linear里的weight和bias都是Float类型，weight要与x进行矩阵乘法，而矩阵乘法要求类型相同
# Linear可以改变特征属性，即可以改变特征维度
linear=nn.Linear(in_features=2,out_features=3)
y=linear(x)
print(y)
print("权重:", linear.weight)
print("偏置:", linear.bias)