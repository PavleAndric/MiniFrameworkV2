import torch
import numpy as np
t = torch.Tensor([[0.6],[0.2]]).float() ; t.requires_grad = True 
k = torch.Tensor([[1.2],[0.3]]).float() ; k.requires_grad = True
o = torch.Tensor([[2.0],[3.0]]).float() ; o.requires_grad = True 
 
first = t + k 
second = o * first
L = second.sum()
print(L)
L.backward()

print(t.grad)
print(k.grad)
print(o.grad)


t = np.array([[0.6],[0.2]])
k = np.array([[1.2],[0.3]])
o = np.array([[2.0],[3.0]])

first  = t+ k
second = o * first
L = np.sum(second)
print(L)

L_grad = 1

second_grad = np.ones_like(first) * L_grad
first_grad = second_grad * o
o_grad=  second_grad * first
t_grad= first_grad * 1
k_grad = first_grad * 1



print(t_grad)
print(k_grad)
print(o_grad)