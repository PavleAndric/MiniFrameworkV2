import torch
import numpy as np
t = torch.Tensor([[0.6],[0.2]]).float() ; t.requires_grad = True 
k = torch.Tensor([[1.2],[0.3]]).float() ; k.requires_grad = True
o = torch.Tensor([[2.0],[3.0]]).float() ; o.requires_grad = True 
q = torch.tensor([[1.4], [0.3]]);  q.requires_grad = True
first = t + k  ; first.retain_grad() 
second = o * first ; second.retain_grad()
third = torch.sqrt(second); third.retain_grad()
forth = third ** q ; forth.retain_grad()
L = forth.mean() ; L.retain_grad()
L.backward()



print(f" ograd {o.grad}")
print(f"t.grad = {t.grad}")
print(f" k.grad {k.grad}")
print(f"qgrad = {q.grad}")


#t = np.array([[0.6],[0.2]])
#k = np.array([[1.2],[0.3]])
#o = np.array([[2.0],[3.0]])
#
#first  = t+ k
#second = o * first
#third  = np.log(second)
#
#L = third.mean()
#print("loss", L)
#
#
#L_grad = 1
#t = np.size(third)
#n = np.sum(third)
#third_grad=  L_grad * (np.ones_like(third)/ np.size(third))
#second_grad = third_grad * (np.ones_like(second) /second)
#first_grad = o * second_grad
#o_grad = first * second_grad
#t_grad = first_grad * 1
#k_grad = first_grad * 1
#
#print(f"o_grad {o_grad}")
#print(f"t_grad = {t_grad}")
#print(f"k_grad = {k_grad}")

