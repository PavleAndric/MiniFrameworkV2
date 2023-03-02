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
    # -----------------------------------------  binary OPS ----------------------------------------------------
    def __add__(self, other):   #addition of Tensor by a int(float) or a Tensror of the same size(element wise)
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T  = Tensor(self.data + other.data, (self,other))

        def _backward():
            self.grad += output_T.grad  * 1.0  # chain rule 
            other.grad += output_T.grad * 1.0

        self._backward = _backward
        return output_T

    def __mul__(self, other): #multiplication of Tensor by a int(float) or a Tensror of the same size(element wise)

        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T = Tensor(self.data * other.data,(self, other))

        def _backward():
            self.grad += (other.data * output_T.grad) 
            other.grad += (self.data * output_T.grad)
    
        self._backward = _backward
        return output_T
    
    #def __rmul__(self, other):
    #
    #    if isinstance(other, (int, float)): return Tensor(self.data * other)
    #    elif isinstance(other, Tensor) and self.shape == other.shape : return Tensor(self.data * other.data)
    #    raise TypeError(f"Operation is not suppeorted for {type(other)} and <class 'Tensor'>")
    #
    #def __sub__(self, other):
#
    #    if isinstance(other, (int, float)): return Tensor(self.data - other)
    #    elif isinstance(other, Tensor) and self.shape == other.shape : return Tensor(self.data - other.data)
    #    raise TypeError(f"Operation is not suppeorted for {type(other)} and <class 'Tensor'>")
    #
    #def __rsub__(self,other):
#
    #    if isinstance(other, (int, float)): return Tensor(other  - self.data)
    #    elif isinstance(other, Tensor) and self.shape == other.shape : return Tensor(other.data - self.data)
    #    raise TypeError(f"Operation is not suppeorted for {type(other)} and <class 'Tensor'>")
    #
    #def __pow__(self, other):
#
    #    if isinstance(other, (int, float)): return Tensor(np.power(self.data, other))
    #    elif isinstance(other, Tensor) and self.shape == other.shape: return Tensor(np.power(self.data, other.data))
    #    raise TypeError(f"Operation is not supported for {type(other)} and <class 'Tensor'>")
    #
    #def __rpow__(self, other):
#
    #    if isinstance(other, (int, float)): return Tensor(np.power(other, self.data))
    #    elif isinstance(other, Tensor) and self.shape == other.shape: return Tensor(np.power(other.data , self.data))
    #    raise TypeError(f"Operation is not supported for {type(other)} and <class 'Tensor'>")
#
    #def __truediv__(self, other):
#
    #    if isinstance(other, (int, float)): return Tensor(self.data * other**-1)
    #    elif isinstance(other, Tensor) and self.shape == other.shape: return Tensor(self.data * other.data**-1)
    #    raise TypeError(f"Operation is not supported for {type(other)} and <class 'Tensor'>")
    #
    #def __rtruediv__(self,other):
#
    #    if isinstance(other, (int, float)): return Tensor(other * self.data**-1)
    #    elif isinstance(other, Tensor) and self.shape == other.shape: return Tensor(other.data * self.data**-1)
    #    raise TypeError(f"Operation is not supported for {type(other)} and <class 'Tensor'>")
    #
    #def __neg__(self):
    #    return self * -1.0 
    ## ----------------------- creation--------------------------------------------------------------------------------

    ### -------------------------- unary  OPS-------------------------------------------------------------------------------
    def suma(self,axis = None):
        output = Tensor(self.data.sum(axis = axis))
        print("USLO")
        def _backward():
            self.grad = self.data * output.grad

        self._backward = _backward
        return output 
    

    @classmethod
    def zeros(cls, shape): return cls(np.zeros(shape))
        
    @classmethod
    def ones(cls , shape):  return cls(np.ones(shape))

    @classmethod
    def ones_like(cls,Tensor) : return cls(np.ones(Tensor.shape))
       
    @classmethod
    def zeros_like(cls, Tensor): return cls(np.zeros(Tensor.shape))

    RNG = np.random.default_rng() # rng for Tensors with randrom values
    @classmethod
    def randn(cls, shape): return cls(Tensor.RNG.standard_normal(size = shape))
        
    @classmethod
    def uniform(cls, shape):   return cls(Tensor.RNG.uniform(low = -1 , high =  1, size = shape))
       
    @classmethod
    def arange(cls, start, stop, step): return cls(np.arange(start = start, stop = stop , step = step ))
    
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
        
        self.grad = 1

        for node in reversed(topo):
            node._backward()


print("------------")

t = Tensor([[5], [0.2]])
k = Tensor([[9], [2]])
y = Tensor([[1.5], [12.]])
p = k * t 
h = p + y
o = h * 0.1
L = o.suma()
L.backward()

print(L.grad)
print(o.grad)
print(y.grad)

print(p.grad)
print(t.grad)
print(k.grad)