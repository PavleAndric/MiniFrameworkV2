import numpy as np
from typing import Tuple

np.set_printoptions(precision = 4)  # this is ugly here

class Tensor():

    def __init__(self, data, _children = ()):
        
        self.data = data if isinstance(data, (np.ndarray, np.generic)) else np.array(data, dtype = np.float32)
        self.shape = self.data.shape
        self._backward = lambda : None
        self.prev = set(_children)
        self.grad = 0.0

    def __repr__(self):
        return f"<Tensor data = {self.data}>"
    
    def shape(self)-> Tuple[int]: return self.shape

    def size(self)-> int: return self.data.size
    #-                                            BINARY                                                 -
    def __add__(self, other )-> 'Tensor': 
        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T  = Tensor(self.data + other.data, (self,other))

        def _backward():
            self.grad += output_T.grad 
            other.grad += output_T.grad

        output_T._backward = _backward
        return output_T

    def __mul__(self, other)-> 'Tensor': 
        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T = Tensor(self.data * other.data,(self, other))

        def _backward():
            self.grad += other * output_T.grad 
            other.grad += self * output_T.grad
    
        output_T._backward = _backward
        return output_T
    
    def __pow__(self, other) -> 'Tensor': #https://testbook.com/learn/maths-derivative-of-exponential-function
        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T = Tensor(self.data ** other.data, (self, other))

        def _backward():
            self.grad += other * (self ** (other - 1)) * output_T.grad
            other.grad += output_T * self.log() * output_T.grad

        output_T._backward = _backward
        return output_T

    def __sub__(self, other)-> 'Tensor':
        other  = other if isinstance(other , Tensor) else Tensor(other)
        output_T = Tensor(self.data - other.data, (self, other))

        def _backward():
            self.grad += output_T.grad
            other.grad += -output_T.grad

        output_T._backward = _backward
        return output_T 

    def __radd__(self, other) -> 'Tensor':
        return self + other
    
    def __rmul__(self, other)-> 'Tensor':
        return self * other 
    
    def __rsub__(self, other)-> 'Tensor':
        return other + (self * -1)
    
    def __truediv__(self, other)-> 'Tensor':
        return self * (other **-1)
    
    def __rtruediv__(self, other)-> 'Tensor':
        return other * (self**-1)
    #-                                             UNARY      math                                     -
    def sum(self) -> 'Tensor':
        output_T = Tensor(self.data.sum(), (self, ))
        
        def _backward():
            self.grad += Tensor.ones_like(self) * output_T.grad

        output_T._backward = _backward
        return output_T
    
    def log(self)-> 'Tensor':
        output_T = Tensor(np.log(self.data), (self, ))

        def _backward():
            self.grad += Tensor.ones_like(self) / self * output_T.grad
        
        output_T._backward = _backward
        return output_T
    
    def mean(self)-> 'Tensor':
        output_T = Tensor(np.mean(self.data), (self, ))

        def _backward():
            t = Tensor.ones_like(self)  
            self.grad += t / self.size() * output_T.grad

        output_T._backward = _backward
        return output_T
    
    def sqrt(self)-> 'Tensor':
        output_T = Tensor(np.sqrt(self.data), (self, ))

        def _backward():
            self.grad += 1 / (2 * output_T) * output_T.grad

        output_T._backward = _backward
        return output_T
    #                                            UNARY transformation                                          -
    def __neg__(self)-> 'Tensor':      # TODO this may couse errors 
        return self * -1 

    def abs(self) -> 'Tensor':   
        output_T =  Tensor(np.abs(self.data), (self, ))
        def _backward():
            self.grad += Tensor(np.sign(self.data)) * output_T.grad

        output_T._backward = _backward
        return output_T 
    # TODO write T.grad more efficiently               
    def T(self) -> 'Tensor':
        output_T = Tensor(np.transpose(self.data), (self, ))

        def _backward():
            
            self.grad += Tensor(np.transpose(np.inner(output_T.grad.data, np.ones_like(self.data))))   #TODO find a nicer way to do this

        output_T._backward  = _backward
        return output_T
    
    def unsqueeze(self, axis) -> 'Tensor':
        return Tensor(np.expand_dims(self.data, axis = axis))
    
    #                                                DOT                                                     - 
    def dot(self, other) -> 'Tensor':
        other = other if isinstance(other , Tensor) else Tensor(other)

        output_T  = Tensor(np.dot(self.data, other.data), (self, other))

        def _backward():
            self.grad += Tensor(output_T.grad.data.dot(other.data.T))
            other.grad +=  Tensor(self.data.T.dot(output_T.grad.data))

        output_T._backward = _backward
        return output_T
    #                                               Activation functions                                      - 
    def ReLU(self):
        output_T = Tensor(np.maximum(0, self.data), (self, ))

        def _backward():
            self.grad += Tensor(output_T.data > 0) * output_T.grad

        output_T._backward = _backward
        return output_T
    
    def Sigmoid(self):

        exp = np.exp(-self.data)
        output_T = Tensor((1/(1 + exp)), (self, )) # der_sig (1/(1 + np.exp(-input))* 1- 1/(1 + np.exp(-input)))

        def _backwrad():
            self.grad += Tensor(output_T.data - output_T.data**2) * output_T.grad 

        output_T._backward = _backwrad
        return output_T
    
    def Tanh(self):
        output_T = Tensor(np.tanh(self.data), (self, ))

        def _backward():
            self.grad = Tensor(1- output_T.data**2) * output_T.grad
        
        output_T._backward = _backward
        return output_T

    @classmethod
    def zeros(cls, shape)-> 'Tensor': return cls(np.zeros(shape))
        
    @classmethod
    def ones(cls, shape)-> 'Tensor': return cls(np.ones(shape))

    @classmethod
    def ones_like(cls, Tensor)-> 'Tensor' : return cls(np.ones(Tensor.shape))
       
    @classmethod
    def zeros_like(cls, Tensor)-> 'Tensor': return cls(np.zeros(Tensor.shape))

    RNG = np.random.default_rng() #https://numpy.org/doc/stable/reference/random/generator.html
    @classmethod
    def randn(cls, shape)-> 'Tensor': return cls(Tensor.RNG.standard_normal(size = shape))
        
    @classmethod
    def uniform(cls, shape)-> 'Tensor': return cls(Tensor.RNG.uniform(low = -1 , high =  1, size = shape))
       
    @classmethod
    def arange(cls, start, stop, step)-> 'Tensor': return cls(np.arange(start = start, stop = stop , step = step ))

    #-                                           ENGINE                                                  -
    def backward(self):
        
        topo = []
        visited = set()
    
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = Tensor([1.0])

        for node in reversed(topo):
            node._backward()
