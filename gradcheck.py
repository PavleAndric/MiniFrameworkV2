from Tensor import Tensor
import numpy as np
import torch
import unittest

x_test = np.random.randn(5,5).astype(np.float32)
y_test = np.random.randn(5,5).astype(np.float32)
z_test = np.random.randn(5,5).astype(np.float32)
i_test = np.random.randn(5,5).astype(np.float32)
j_test = np.random.randn(5,5).astype(np.float32)

class Test(unittest.TestCase):

    def test_binary_operations(self):
        def test_mini():
            a = Tensor(x_test)
            b = Tensor(y_test)
            first = a + b
            second = 0.23 * first
            third = second / 2
            expected = third - 0.3
            return expected.data
        
        def test_torch():
            a_ = torch.Tensor(x_test)
            b_ = torch.Tensor(y_test)
            first_ = a_ + b_
            second_ = 0.23 * first_
            third_ = second_ / 2
            true_ = third_ - 0.3 ; true  = true_.cpu().detach().numpy()
            return true
        
        np.testing.assert_allclose(actual = test_mini() , desired = test_torch(), rtol = 1e-6)

    def test_unary_operations(self):
        def test_mini():
            a = Tensor(x_test)
            b = Tensor(y_test)
            c = Tensor(z_test)
            first = a * b.abs().log()
            second = first.abs().sqrt() / -c.mean()
            expected = second.T()
            return expected.data

        def test_torch():
            a_ = torch.Tensor(x_test)
            b_  = torch.Tensor(y_test)
            c_ = torch.Tensor(z_test)
            first_ = a_ * torch.abs(b_).log()
            secodn_ = torch.abs(first_).sqrt() / -torch.mean(c_)
            true = secodn_.cpu().detach().numpy()
            return true.T
        
        np.testing.assert_allclose(actual = test_mini() , desired = test_torch(), rtol = 1e-6)
    
    def test_backward(self):
        pass

if __name__ == '__main__':
    unittest.main()


