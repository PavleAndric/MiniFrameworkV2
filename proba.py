import numpy as np
import torch
from Tensor import Tensor

np.random.seed(100)


x_test = np.random.randn(5,5).astype(np.float32)
y_test = np.random.randn(5,5).astype(np.float32)
z_test = np.random.randn(5,5).astype(np.float32)
i_test = np.random.randn(1,5).astype(np.float32)
j_test = np.random.randn(1,5).astype(np.float32)


def test_mini():
    a = Tensor(x_test)
    b  = Tensor(y_test)
    c = Tensor(i_test)
    rom  = Tensor(j_test) 
    first = a.dot(b) + c.T()
    second = first.abs().log() * rom.T()
    L = second.sum()
    L.backward()
    #print(first.grad)
    #print(first.grad) # good
    #print(np.inner(first.grad, c))
    return a.grad.data, b.grad.data, c.grad.data, rom.grad.data

def test_torch():
    a_ = torch.Tensor(x_test)                               ; a_.requires_grad = True 
    b_ = torch.Tensor(y_test)                               ; b_.requires_grad = True 
    c_ = torch.Tensor(i_test)                               ; c_.requires_grad = True 
    rom_ = torch.Tensor(j_test)                             ; rom_.requires_grad = True 

    first_ = a_.matmul(b_) + torch.transpose(c_, 0 , 1)     ; first_.retain_grad()
    second_ = torch.log(torch.abs(first_))  * torch.transpose(rom_, 0, 1)                ; second_.retain_grad()
    L = torch.sum(second_)
    L.backward()
    #print(first_.grad)
    #print(first_.grad)
    grad1, grad2, grad3 , grad4 = a_.grad.cpu().detach().numpy(), b_.grad.cpu().detach().numpy(), c_.grad.cpu().detach().numpy(), rom_.grad.cpu().detach().numpy()
    return grad1, grad2, grad3, grad4
    



a, b , c, d = test_mini()
a_, b_, c_ ,d_= test_torch()


print(d)
print(d_)
print(c)
print(c_)
print()
print()
print()
print(a)
print(a_)
print()
print()
print(b)
print(b_)
