import numpy as np
import math
 
class Tensor():

    def __init__(self, data, _children = ()):

        self.data = np.array(data, dtype =np.float32)
        self.shape = self.data.shape
        self._backward = lambda : None
        self.prev = set(_children)
        self.grad = 0.0

    def __repr__(self):
        return f"<Tensor data = {self.data}>"
    
    def shape(self): return self.shape

    def size(self): return self.data.size
    #-                                            BINARY                                                 -
    def __add__(self, other): 
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T  = Tensor(self.data + other.data, (self,other))

        def _backward():
            self.grad += output_T.grad   # chain rule 
            other.grad += output_T.grad

        output_T._backward = _backward
        return output_T

    def __mul__(self, other): 

        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T = Tensor(self.data * other.data,(self, other))

        def _backward():
            self.grad += other * output_T.grad 
            other.grad += self * output_T.grad
    
        output_T._backward = _backward
        return output_T
    
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T = Tensor(self.data ** other.data, (self, other))

        def _backward():
            self.grad += other * (self ** (other - 1)) * output_T.grad
        
        output_T._backward = _backward
        return output_T

    def __sub__(self, other):
        other  = other if isinstance(other , Tensor) else Tensor(other)
        
        output_T = Tensor(self.data - other.data, (self, other))

        def _backward():
            self.grad += output_T.grad
            other.grad += -output_T.grad

        output_T._backward = _backward
        return output_T 

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other 
    
    def __rsub__(self, other):
        return other + (self * -1)
    
    def __truediv__(self, other):
        return self * (other **-1)
    
    def __rtruediv__(self, other):
        return other * (self**-1)
    #-                                             UNARY                                               -
    def sum(self):
        output_T = Tensor(self.data.sum(), (self, ))
        
        def _backward():
            self.grad = Tensor.ones_like(self) * output_T.grad

        output_T._backward = _backward
        return output_T
    
    def log(self):
        output_T = Tensor(np.log(self.data), (self, ))

        def _backward():
            self.grad = Tensor.ones_like(self) / self * output_T.grad
        
        output_T._backward = _backward
        return output_T
    
    def mean(self):
        output_T = Tensor(np.mean(self.data), (self, ))

        def _backward():
            t = Tensor.ones_like(self)  
            self.grad = t / self.size() * output_T.grad

        output_T._backward = _backward
        return output_T
    
    def sqrt(self):
        output_T = Tensor(np.sqrt(self.data), (self, ))

        def _backward():
            self.grad = 1 / (2 * output_T) * output_T.grad

        output_T._backward = _backward
        return output_T
    
    def __neg__(self):
        return self * -1 

    @classmethod
    def zeros(cls, shape): return cls(np.zeros(shape))
        
    @classmethod
    def ones(cls , shape):  return cls(np.ones(shape))

    @classmethod
    def ones_like(cls,Tensor) : return cls(np.ones(Tensor.shape))
       
    @classmethod
    def zeros_like(cls, Tensor): return cls(np.zeros(Tensor.shape))

    RNG = np.random.default_rng() #https://numpy.org/doc/stable/reference/random/generator.html
    @classmethod
    def randn(cls, shape): return cls(Tensor.RNG.standard_normal(size = shape))
        
    @classmethod
    def uniform(cls, shape):   return cls(Tensor.RNG.uniform(low = -1 , high =  1, size = shape))
       
    @classmethod
    def arange(cls, start, stop, step): return cls(np.arange(start = start, stop = stop , step = step ))

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
