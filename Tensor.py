import numpy as np

class Tensor():

    def __init__(self, data):
        self.data = np.array(data, dtype =np.float32)
        self.shape = self.data.shape
        self.grad = 0

    def __repr__(self):
        return f"<Tensor data = {self.data}>"
    
    def shape(self): return self.shape

    def __add__(self, other):
 
        if isinstance(other, (int, float)):  return Tensor(self.data + other)
        
        elif isinstance(other, Tensor) and self.shape == other.shape: return Tensor(self.data + other.data)
            
        else: raise TypeError(f"Operation is not supported for {type(other)} and <class 'Tensor'>") #### FIX THIS
    
    def __mul__(self, other):

        if isinstance(other, (int, float)): return Tensor(self.data * other)
            
        elif isinstance(other, Tensor) and self.shape == other.shape: return Tensor(self.data * other.data)
         
        else: raise TypeError(f"Operation is not supported for {type(other)} and <class 'Tensor'>") ### AND THIS
     
    @classmethod
    def zeros(cls, shape): return cls(np.zeros(shape))
        
    @classmethod
    def ones(cls , shape):  return cls(np.ones(shape))
       
    @classmethod
    def zeros_like(cls, Tensor): return cls(np.zeros(Tensor.shape))

    RNG = np.random.default_rng() # rng for Tensors with randrom values
    @classmethod
    def randn(cls, shape): return cls(Tensor.RNG.standard_normal(size = shape))
        
    @classmethod
    def uniform(cls, shape):   return cls(Tensor.RNG.uniform(low = -1 , high =  1, size = shape))
       
    @classmethod
    def arange(cls, start, stop, step): return cls(np.arange(start = start, stop = stop , step = step ))
        
t = Tensor.ones((3,3))
k = Tensor.ones((3,3))
p = t * k
print(p) 
