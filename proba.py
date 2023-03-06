import torch


t = torch.Tensor([1,2,3,4,6,7,8,9])
k = torch.abs(t).log()

print(k)