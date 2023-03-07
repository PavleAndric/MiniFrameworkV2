import numpy as np
import torch
from Tensor import Tensor

np.random.seed(100)


x_test = np.random.randn(5,5).astype(np.float32)
y_test = np.random.randn(5,5).astype(np.float32)


t = Tensor(x_test)
p = Tensor(y_test)
O = t * p
first =  O.Sigmoid() #  good
second  = first * t
L = (second).sum()
L.backward()


t_ = torch.Tensor(x_test)                   ; t_.requires_grad = True 
p_ = torch.Tensor(y_test)                   ; p_.requires_grad = True 
O_ = t_ * p_                                 ; O_.retain_grad()    
first_ = torch.sigmoid(O_)                   ; first_.retain_grad()
second_  = first_ * t_
Loss = torch.sum(second_)
Loss.backward()

#print(first)
#print(first_)
print(t.grad)
print(t_.grad)