import sys
sys.path.append(r"C:\Users\pavle\MiniFrameworkV2\Mini")
import numpy as np
from Mini.Tensor import Tensor
import torch
import unittest

x_test = np.random.randn(5,5).astype(np.float32) * 0.1
y_test = np.random.randn(5,1).astype(np.float32) * 0.1
z_test = np.random.randn(5,1).astype(np.float32) * 0.1 

MiniReLU = lambda Tensor: Tensor.ReLU()
MiniTanh = lambda Tensor: Tensor.Tanh()
MiniSigmoid =  lambda Tensor:  Tensor.Sigmoid()

TorchReLU = lambda t: torch.relu(t)
TorchTanh = lambda t: torch.tanh(t)
TorchSigmoid = lambda t: torch.sigmoid(t)

list_of_func_mini = [MiniReLU, MiniTanh, MiniSigmoid]
list_of_func_torch = [TorchReLU , TorchTanh, TorchSigmoid]

class test_Activations(unittest.TestCase):

    def test_activations(self):

        def test_mini():
            mini_test = []    
            for func in list_of_func_mini:
                a = Tensor(x_test)
                b = Tensor(y_test)
                z = Tensor(z_test)
                first = func(a.dot(b) + z)
                L = first.sum()
                L.backward()
                mini_test.append(first.data) ; mini_test.append(first.grad.data)
            return mini_test

        def test_torch():
            torch_test = []
            for func in list_of_func_torch:
                a_ = torch.Tensor(x_test)                                           ; a_.requires_grad =True 
                b_ = torch.Tensor(y_test)                                           ; b_.requires_grad =True
                z_ = torch.Tensor(z_test)                                           ; z_.requires_grad =True   
                first_ = func(a_.matmul(b_) + z_)                                   ; first_.retain_grad()
                L_ = torch.sum(first_)
                L_.backward()
                first_ , first_grad_= first_.cpu().detach().numpy(), first_.grad.cpu().detach().numpy()
                torch_test.append(first_) ; torch_test.append(first_grad_)
            return torch_test

        for x, y in zip(test_mini(), test_torch()):
            np.testing.assert_allclose(actual = x , desired = y, rtol = 1e-5)

    def test_Softmax(self): #

        target = np.array([[1.], [0.], [0.], [0.], [0.]]).astype(np.float32)
        loss_fn = torch.nn.CrossEntropyLoss()

        def test_mini(): 
            target_mini = Tensor(target)
            a = Tensor(x_test)
            b = Tensor(y_test)
            z = Tensor(z_test)
            first = (a.dot(b) + z).Softmax(axis = 0)
            L =  -(target_mini * (first.log())).sum()           # Cross entropy loss
            L.backward()
            return a.grad.data, b.grad.data, z.grad.data, 
    
        def test_torch(): 
            target_torch = torch.Tensor(target)
            a_ = torch.Tensor(x_test)                                               ;a_.requires_grad = True                                      
            b_ = torch.Tensor(y_test)                                               ;b_.requires_grad = True
            z_ = torch.Tensor(z_test)                                               ;z_.requires_grad = True
            first_ = a_.matmul(b_) + z_ 
            L_ = loss_fn(first_.squeeze(), target_torch.squeeze())
            L_.backward()
            a_grad , b_grad , z_grad = a_.grad.detach().numpy(), b_.grad.detach().numpy(), z_.grad.detach().numpy()
            return a_grad, b_grad, z_grad

        for x, y in zip(test_mini(), test_torch()):
            np.testing.assert_allclose(actual = x , desired = y, rtol = 1e-5)


if __name__ == '__main__':
    unittest.main()


# TODO MAKE THIS PRETTY