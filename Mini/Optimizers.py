from Tensor import Tensor
import numpy as np

class Optim():

    def __init__(self, parameters , learning_rate = 0.1):
        self.params = [x for x in parameters]
        self.lr =  learning_rate

    def zero_grad(self, ALL_PARAMS):
            for x in  ALL_PARAMS: x.grad = 0

class SGD(Optim):

    def __init__(self , parameters, learning_rate , momentum = 0):
        self.Vt = [Tensor([0.]) for x  in parameters]
        self.mm = momentum
        super(SGD, self).__init__(parameters, learning_rate)

    def step(self):
        
        for param, vt in zip(self.params,  self.Vt):
            vt.data =  self.mm * vt.data + self.lr * param.grad.data
            param.data = param.data - vt.data

class RMSProp(Optim):

    def __init__(self, parameters, learning_rate, beta = 0.999):
        self.Beta = [Tensor([0.]) for x in parameters]
        self.mm, self.e = beta, 1e-8
        super(RMSProp, self).__init__(parameters, learning_rate)

    def step(self):
        
        for param, bt in zip(self.params, self.Beta):
            bt.data = self.mm * bt.data + (1 - self.mm) * param.grad.data**2
            param.data = param.data - self.lr * (param.grad.data / np.sqrt(bt.data + self.e))

        
class Adam(Optim):

    def __init__(self, parameters , learning_rate, betas):
        self.Beta1  = [Tensor([0.]) for x in parameters]
        self.Beta2 = [Tensor([0.]) for x in parameters]
        self.mm1 , self.mm2 = betas
        super(Adam, self).__init__(parameters, learning_rate)
        self.t = 0
        
    def step(self):
        self.t += 1
        for param, b1 , b2  in zip(self.params, self.Beta1, self.Beta2):
            b1.data = self.mm1 * b1.data + (1 - self.mm1) * param.grad.data
            b2.data = self.mm2 * b2.data + (1 - self.mm2) * param.grad.data**2
            b1_cor , b2_cor  = b1.data / (1 - self.mm1**self.t),  b2.data / (1 - self.mm2**self.t)
            param.grad.data = -self.lr * (b1_cor / ((b2_cor + 1e-8) ** 0.5))
            param.data = param.data + param.grad.data 
        