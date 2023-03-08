import sys
sys.path.append(r"C:\Users\pavle\MiniFrameworkV2\Mini")
import numpy as np
from Mini.Tensor import Tensor
import torch
import unittest

x_test = np.random.randn(4,4).astype(np.float32)
y_test = np.random.randn(4,4).astype(np.float32)
z_test = np.random.randn(4,4).astype(np.float32)
i_test = np.random.randn(1,4).astype(np.float32)
j_test = np.random.randn(1,4).astype(np.float32)
k_test = np.random.randn(4,1).astype(np.float32)

class Tensor_operations_testing(unittest.TestCase):

    def test_binary_operations(self):
        def test_mini():
            a = Tensor(x_test)
            b = Tensor(y_test)
            first = a + b
            second = 0.23 * first
            third = second / 2
            actual = third - 0.3
            return actual.data
        
        def test_torch():
            a_ = torch.Tensor(x_test)
            b_ = torch.Tensor(y_test)
            first_ = a_ + b_
            second_ = 0.23 * first_
            third_ = second_ / 2
            true_ = third_ - 0.3 ; true  = true_.cpu().detach().numpy()
            return true
        
        np.testing.assert_allclose(actual = test_mini() , desired = test_torch(), rtol = 1e-5)

    def test_unary_operations(self):
        def test_mini():
            a = Tensor(x_test)
            b = Tensor(y_test)
            c = Tensor(z_test)
            first = a * b.abs().log()
            second = first.abs().sqrt() / -c.mean()
            actual = second.T()
            return actual.data

        def test_torch():
            a_ = torch.Tensor(x_test)
            b_  = torch.Tensor(y_test)
            c_ = torch.Tensor(z_test)
            first_ = a_ * torch.abs(b_).log()
            secodn_ = torch.abs(first_).sqrt() / -torch.mean(c_)
            true = secodn_.cpu().detach().numpy()
            return true.T
        
        np.testing.assert_allclose(actual = test_mini() , desired = test_torch(), rtol = 1e-5)
    
    def test_dot(self):
        def test_mini():
            a = Tensor(x_test)
            b = Tensor(k_test)
            c = Tensor(i_test)
            actual = a.dot(b) + c
            return actual.data
        
        def test_torch():
            a_ = torch.Tensor(x_test)
            b_ = torch.Tensor(k_test)
            c_ = torch.Tensor(i_test)
            true = a_.matmul(b_) + c_
            true = true.cpu().detach().numpy()
            return true
        
        np.testing.assert_allclose(actual = test_mini() , desired = test_torch(), rtol = 1e-5)

if __name__ == '__main__':
    unittest.main()


