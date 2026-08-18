import torch
from torch.utils.data import TensorDataset, DataLoader

x=torch.tensor([1,2,3])
y=torch.tensor([[1,2,3],[4,5,6]])
# shape是属性,dim是方法,size是方法，不传参数时等于shape,torch.Size([3]) 1 torch.Size([3])
print(x.shape)
print(x.dim())
print(x.size())
print(y.shape)
print(y.dim())
print(y.size(0))

dataset=TensorDataset(x,y)
print(dataset[1])
# DataLoader是把两个Tensor拼接到一起，之后也都一直保持Tensor，若要只想得到数字，使用.item()
dataloader=DataLoader(dataset,batch_size=2)
for i in dataloader:   # DataLoader里的基本单位是Batch所以直接遍历就是一个Batch，[tensor([1, 2]), tensor([4, 5])]，[tensor([3]), tensor([6])]
    print(i)            #这里的i也就是一个Batch即多个sample（样本），
for x,y in dataloader: # 输出：tensor([1, 2]) tensor([4, 5])，tensor([3])，tensor([6])，DataLoader只要取就是按batch_size取，所以第一次取两个，x，y各有两个
    print(x)
    print(y)
for batch_id,(x,y) in enumerate(dataloader): # enumerate()函数同时取出索引和值，0 tensor([1, 2]) tensor([4, 5])，1 tensor([3]) tensor([6])
    print(batch_id,x,y)
