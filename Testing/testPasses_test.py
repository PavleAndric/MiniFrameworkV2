import sys
sys.path.append(r"C:\Users\pavle\MiniFrameworkV2\Mini")
from Mini.Tensor import Tensor
import numpy as np
import torch
import unittest


x_test = np.random.randn(4,4).astype(np.float32)
y_test = np.random.randn(4,4).astype(np.float32)
z_test = np.random.randn(4,4).astype(np.float32)
i_test = np.random.randn(1,4).astype(np.float32)
j_test = np.random.randn(1,4).astype(np.float32)
k_test = np.random.randn(4,1).astype(np.float32)

class Tensor_operations_testing(unittest.TestCase):


    def test_backward(self):
        def test_mini():
            a = Tensor(x_test)
            b = Tensor(y_test)
            c = Tensor(i_test)
            d = Tensor(j_test)
            first = a.dot(b) + c.T()
            second = first.abs().log() * d.T()
            L = second.sum()
            L.backward()
            return a.grad.data, b.grad.data, c.grad.data, d.grad.data
        
        def test_torch():
            a_ = torch.Tensor(x_test)                                           ; a_.requires_grad = True 
            b_ = torch.Tensor(y_test)                                           ; b_.requires_grad = True 
            c_ = torch.Tensor(i_test)                                           ; c_.requires_grad = True
            d_ = torch.Tensor(j_test)                                           ; d_.requires_grad = True 
            first_ = a_.matmul(b_) + torch.transpose(c_, 0 , 1)
            second_ = torch.log(torch.abs(first_)) * torch.transpose(d_, 0 , 1)
            L = torch.sum(second_)
            L.backward()
            
            grad1, grad2, grad3 ,grad4 = a_.grad.cpu().detach().numpy(), b_.grad.cpu().detach().numpy(), c_.grad.cpu().detach().numpy(), d_.grad.cpu().detach().numpy()
            return grad1, grad2, grad3, grad4
            
        for x, y in zip(test_mini(), test_torch()):
            np.testing.assert_allclose(actual = x , desired = y, rtol = 1e-4)

    def test_forward(self):
        def test_mini():
            a = Tensor(x_test)
            b = Tensor(k_test)
            c = Tensor(i_test)
            Z = a.dot(b) + c.T()
            A  = Z.Tanh()
            return A.data
        
        def test_torch():
            a_ = torch.Tensor(x_test)
            b_ = torch.Tensor(k_test)
            c_ = torch.Tensor(i_test)
            Z_ = a_.matmul(b_) + torch.transpose(c_, 0 , 1)
            true  = torch.tanh(Z_)
            true = true.cpu().detach().numpy()
            return true 

        np.testing.assert_allclose(actual = test_mini() , desired = test_torch(), rtol = 1e-5)

if __name__ == '__main__':
    unittest.main()


