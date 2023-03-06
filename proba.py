import torch
import numpy as np
from  Tensor import Tensor
#t = torch.Tensor([[1,2,3], [4,5,6]]) ; t.requires_grad = True 
#k = torch.Tensor([[0.1,0.2],[9,8],[1,2]]) ; k .requires_grad = True 
#print(f"shapes  = {t.shape}, {k.shape}")
#
#o = t.matmul(k) ; o.retain_grad() # o grad 2*2 matrxi 1,1,1,1
#Loss = o.sum() ; Loss.retain_grad()
#Loss.backward()
#
#print(o.grad)
#print(t.grad)
#print(k.grad)

#t = np.array([[1,2,3], [4,5,6]])  # W
#k = np.array([[0.1,0.2],[9,8],[1,2]]) # X
#print(f"shapes = {t.shape} , {k.shape}")
#o = t.dot(k)
#L = o.sum()
#L_grad = 1
#o_grad = L_grad * np.ones_like(o) # o_grad = 2,2 of ones
#
#t_grad = o_grad.dot(k.T)
#k_grad = t.T.dot(o_grad)
#
#print(k_grad)
#print(t_grad)

np.random.seed(100)
x_test = np.random.randn(5,5).astype(np.float32)
y_test = np.random.randn(5,5).astype(np.float32)
z_test = np.random.randn(5,5).astype(np.float32)
i_test = np.random.randn(5,1).astype(np.float32)
j_test = np.random.randn(5,1).astype(np.float32)

#a_ = torch.Tensor(x_test) ; a_.requires_grad = True 
#b_ = torch.Tensor(y_test) ; b_.requires_grad = True 
#frist_ = a_ * b_ ; frist_.retain_grad()
#secodn_ =  torch.log(torch.abs(frist_)) ; secodn_.retain_grad()
#L = torch.sum(secodn_)
#L.backward()
#
#print(b_.grad)
#
#a = Tensor(x_test) ; a_.requires_grad = True 
#b = Tensor(y_test) ; b_.requires_grad = True 
#frist = a * b
#secodn =  frist.abs().log()
#L = secodn.sum()
#L.backward()
#print(b.grad)
#
def test_mini():
    a = Tensor(x_test)
    b  = Tensor(y_test)
    c = Tensor(z_test)
    first = a.dot(b) + c
    second = first.abs().log()
    L = second.sum()
    L.backward()
    print(first.grad)
    return a.grad.data, b.grad.data, c.grad.data

def test_torch():
    a_ = torch.Tensor(x_test) ; a_.requires_grad = True 
    b_ = torch.Tensor(y_test) ; b_.requires_grad = True 
    c_ = torch.Tensor(z_test) ; c_.requires_grad = True  
    first_ = a_.matmul(b_) + c_ ; first_.retain_grad()
    second_ = torch.log(torch.abs(first_)) ; second_.retain_grad()
    L = torch.sum(second_) ; L.retain_grad()
    L.backward()
    print(first_.grad)
    grad1, grad2, grad3= a_.cpu().detach().numpy(), b_.cpu().detach().numpy(), c_.cpu().detach().numpy()
    return grad1, grad2, grad3


a, b, c = test_mini()

a_, b_ ,c_ = test_torch()
#print(a) # MINI
print()
print()
print()
#print(a_) # TORCH
print(a == a_)