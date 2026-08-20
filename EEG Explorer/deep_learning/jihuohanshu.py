import torch
import torch.nn as nn

x=torch.tensor([
    [1.,2.],
    [3.,4.],
    [5.,6.],
    [7.,8.]]
)
model=nn.Sequential(
    nn.Linear(2,3),
    nn.ReLU(),
    nn.Linear(3,1),
)
y=model(x)
print(x.shape)
print(y.shape)
