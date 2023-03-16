
class Optim():

    def __init__(self , parameters, learning_rate = 0.001):
        self.params = [x for x in parameters]
        self.lr = learning_rate

    def zero_grad(self):
        
        for x in self.params:
            x.grad = 0
    
    def step(self):

        for param in self.params: param.data  = param.data * self.lr
