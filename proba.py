from Mini.Tensor import Tensor
import torch
import numpy as np


x_test = np.random.randn(5,5).astype(np.float32) * 0.1
y_test = np.random.randn(5,5).astype(np.float32) * 0.1
z_test = np.random.randn(5,5).astype(np.float32) * 0.1 

MiniReLU = lambda Tensor: Tensor.ReLU()
MiniTanh = lambda Tensor: Tensor.Tanh()
MiniSigmoid =  lambda Tensor:  Tensor.Sigmoid()
TorchReLU = lambda t: torch.relu(t)
TorchTanh = lambda t: torch.tanh(t)
TorchSigmoid = lambda t: torch.sigmoid(t)
list_of_func_mini = [MiniReLU, MiniTanh, MiniSigmoid]
list_of_func_torch = [TorchReLU , TorchTanh, TorchSigmoid]



#t = Tensor(x_test)
#print(MiniReLU(t))

#for func_mini, torch_func in zip(list_of_func_mini, list_of_func_torch):
#     
#     def test_generic(func_mini, func_torch):
#
#            def test_mini():
#                a = Tensor(x_test)
#                b = Tensor(y_test)
#                z = Tensor(z_test)
#                first = (a.dot(b) + z).func_mini()
#                L = first.sum()
#                L.backward()
#                return first.data, a.grad.data, b.grad.data, z.grad.data
#            
#            def test_torch():
#                a_ = torch.Tensor(x_test)                       ; a_.requires_grad =True 
#                b_ = torch.Tensor(y_test)                       ; b_.requires_grad =True
#                z_ = torch.Tensor(z_test)                       ; z_.requires_grad =True   
#                first_ = func_torch((a_.matmul(b_) + z_))
#                L_ = first_.sum()
#                L_.backward()
#                first_ , a_grad, b_grad, c_grad = first_.cpu().detach().numpy(), a_.grad.cpu().detach().numpy(), b_.grad.cpu().detach().numpy(), z_.grad.cpu().detach().numpy()
#                return first_, a_grad, b_grad, c_grad
#            
#            for x, y in zip(test_mini(), test_torch()):
#                np.testing.assert_allclose(actual = x , desired = y, rtol = 1e-5)

#for func_mini, torch_func in zip(list_of_func_mini, list_of_func_torch):
#
#    a = Tensor(x_test)
#    b = Tensor(y_test)
#    z = Tensor(z_test)
#    first = func_mini((a.dot(b) + z))
#    L = first.sum()
#    L.backward()
#    print(first)
#
#    a_ = torch.Tensor(x_test)                       ; a_.requires_grad =True 
#    b_ = torch.Tensor(y_test)                       ; b_.requires_grad =True
#    z_ = torch.Tensor(z_test)                       ; z_.requires_grad =True   
#    first_ = torch_func((a_.matmul(b_) + z_))       ; first_.retain_grad()
#    L_ = first_.sum()
#    L_.backward()
#    print(first_)
#    
class ROM():

    def test_mini():
        mini_test = []    
        for func in list_of_func_mini:
            a = Tensor(x_test)
            b = Tensor(y_test)
            z = Tensor(z_test)
            first = func((a.dot(b) + z))
            mini_test.append(first.data)
        return mini_test

    def test_torch():
        torch_test = []
        for func in list_of_func_torch:
            a_ = torch.Tensor(x_test)                       ; a_.requires_grad =True 
            b_ = torch.Tensor(y_test)                       ; b_.requires_grad =True
            z_ = torch.Tensor(z_test)                       ; z_.requires_grad =True   
            first_ = func((a_.matmul(b_) + z_))
            first_ = first_.cpu().detach().numpy()
            torch_test.append(first_)
        return torch_test

    #for x, y in zip(test_mini(), test_torch()):
    #    np.testing.assert_allclose(actual = x , desired = y, rtol = 1e-5)

    for x, y in zip(test_mini(), test_torch()):
        np.testing.assert_allclose(actual = x , desired = y, rtol = 1e-5)
        print(x, y)
