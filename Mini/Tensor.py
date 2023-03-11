import numpy as np

np.set_printoptions(precision = 5)  # this is ugly here

class Tensor():

    def __init__(self, data, _children = () , label = ""):
        self.label = label
        self.data = data if isinstance(data, (np.ndarray, np.generic)) else np.array(data, dtype = np.float32)
        self.shape = self.data.shape
        self._backward = lambda : None
        self.prev = set(_children)
        self.grad = 0.0

    def __repr__(self):
        return f"<Tensor = {self.data}>"

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
    
    def __pow__(self, other) -> 'Tensor':                                       #https://testbook.com/learn/maths-derivative-of-exponential-function
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
    
    def __radd__(self, other) -> 'Tensor': return self + other
    def __rmul__(self, other)-> 'Tensor': return self * other
    def __rsub__(self, other)-> 'Tensor': return other + (self * -1)
    def __truediv__(self, other)-> 'Tensor': return self * (other **-1)
    def __rtruediv__(self, other)-> 'Tensor': return other * (self**-1)
    def __neg__(self)-> 'Tensor':  return self * -1    # TODO this may couse errors

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
    
    def abs(self) -> 'Tensor':   
        output_T =  Tensor(np.abs(self.data), (self, ))

        def _backward():
            self.grad += Tensor(np.sign(self.data)) * output_T.grad

        output_T._backward = _backward
        return output_T 
                 
    def T(self) -> 'Tensor':                                                  
        output_T = Tensor(np.transpose(self.data), (self, ))

        def _backward():
            t =  np.inner(output_T.grad.data, np.ones_like(self.data))
            self.grad += Tensor(np.transpose(t))                                #TODO find a nicer way to do this

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
    def ReLU(self) -> 'Tensor':
        output_T = Tensor(np.maximum(0, self.data), (self, ))

        def _backward():
            self.grad += Tensor(output_T.data > 0) * output_T.grad

        output_T._backward = _backward
        return output_T
    
    def Sigmoid(self) -> 'Tensor':
        exp = np.exp(-self.data)
        output_T = Tensor((1/(1 + exp)), (self, ))

        def _backwrad():
            self.grad += Tensor(output_T.data - output_T.data**2) * output_T.grad 

        output_T._backward = _backwrad
        return output_T
    
    def Tanh(self) -> 'Tensor':
        output_T = Tensor(np.tanh(self.data), (self, ))

        def _backward():
            self.grad = Tensor(1- output_T.data**2) * output_T.grad
        
        output_T._backward = _backward
        return output_T
    
    def Softmax(self, axis = None) -> 'Tensor' :                             # https://stackoverflow.com/questions/42599498/numerically-stable-softmax https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy

        z = self.data - np.max(self.data)
        exp = np.exp(z)
        output_T = Tensor(exp / np.sum(exp, axis = axis, keepdims = True), (self, ))

        def _backward():                                        #Tensor(np.matmul((np.diagflat(third_.data) - np.dot(third_.data, third_.data.T)), third_.grad.data))
            self.grad += Tensor(np.matmul((np.diagflat(output_T.data) - np.dot(output_T.data, output_T.data.T)), output_T.grad.data))

        output_T._backward = _backward
        return output_T
    
    # TODO maybe add logsoftmax
    #def Log_Softmax(self):
    #
    #    z = np.exp(self.data  - np.max(self.data))
    #    softmax_ = z / np.sum(z)
    #    output_T = np.log(softmax_)
    #
    #    def _backward():
    #        pass
    #        #self.grad  +=  # * output_T.grad
            

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

        assert self.data.size == 1
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
