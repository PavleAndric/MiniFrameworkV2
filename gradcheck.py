from Tensor import Tensor
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

if __name__ == '__main__':
    unittest.main()


