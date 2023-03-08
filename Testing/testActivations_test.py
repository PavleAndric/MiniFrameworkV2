import sys
sys.path.append(r"C:\Users\pavle\MiniFrameworkV2\Mini")
import numpy as np
from Mini.Tensor import Tensor
import torch
import unittest

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

class test_Activations(unittest.TestCase):

    def test_activations(self):

        def test_mini():
            mini_test = []    
            for func in list_of_func_mini:
                a = Tensor(x_test)
                b = Tensor(y_test)
                z = Tensor(z_test)
                first = (a.dot(b) + z)
                secodn  = func(first)
                L = secodn.sum()
                L.backward()
                mini_test.append(first.data)
                mini_test.append(first.grad.data)
            return mini_test

        def test_torch():
            torch_test = []
            for func in list_of_func_torch:
                a_ = torch.Tensor(x_test)                                           ; a_.requires_grad =True 
                b_ = torch.Tensor(y_test)                                           ; b_.requires_grad =True
                z_ = torch.Tensor(z_test)                                           ; z_.requires_grad =True   
                first_ = (a_.matmul(b_) + z_)                                       ; first_.retain_grad()
                second_ = func(first_)
                L_ = torch.sum(second_)
                L_.backward()
                first_ , first_grad_= first_.cpu().detach().numpy(), first_.grad.cpu().detach().numpy()
                torch_test.append(first_)
                torch_test.append(first_grad_)
            return torch_test

        for x, y in zip(test_mini(), test_torch()):
            np.testing.assert_allclose(actual = x , desired = y, rtol = 1e-5)

if __name__ == '__main__':
    unittest.main()


# TODO MAKE THIS PRETTY