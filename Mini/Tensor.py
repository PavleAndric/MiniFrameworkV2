import numpy as np
import TensorOps
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

    def __add__(self, other)-> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T = Tensor(TensorOps.ADD.forward(self, other), (self,other))

        def _backward():      
            self.grad , other.grad = TensorOps.ADD.backward(self, output_T.grad)

        output_T._backward = _backward
        return output_T

    def __mul__(self, other)-> 'Tensor': 
        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T = Tensor(TensorOps.MUL.forward(self, other), (self, other))

        def _backward():
            self.grad, other.grad = TensorOps.MUL.backward(self, output_T.grad)

        output_T._backward = _backward
        return output_T
    
    def __pow__(self, other) -> 'Tensor':                                       #https://testbook.com/learn/maths-derivative-of-exponential-function
        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T = Tensor(TensorOps.POW.forward(self, other), (self, other))

        def _backward():
            self.grad , other.grad = TensorOps.POW.backward(self, output_T, other)

        output_T._backward = _backward
        return output_T

    def __sub__(self, other)-> 'Tensor':
        other  = other if isinstance(other , Tensor) else Tensor(other)
        output_T = Tensor(TensorOps.SUB.forward(self, other), (self, other))

        def _backward():
            self.grad, other.grad = TensorOps.SUB.backward(self, output_T.grad)

        output_T._backward = _backward
        return output_T 
    
    def __radd__(self, other) -> 'Tensor': return self + other
    def __rmul__(self, other)-> 'Tensor': return self * other
    def __rsub__(self, other)-> 'Tensor': return other + (self * -1)
    def __truediv__(self, other)-> 'Tensor': return self * (other **-1)
    def __rtruediv__(self, other)-> 'Tensor': return other * (self**-1)
    def __neg__(self)-> 'Tensor':  return self * -1                             # TODO this may couse errors

    #-                                             UNARY      math                                     -
    def sum(self) -> 'Tensor':
        output_T = Tensor(TensorOps.SUM.forward(self), (self,))
  
        def _backward():
            self.grad = TensorOps.SUM.backward(self, output_T.grad)

        output_T._backward = _backward
        return output_T
    
    def log(self)-> 'Tensor':
        output_T = Tensor(TensorOps.LOG.forward(self), (self,))

        def _backward():
            self.grad = TensorOps.LOG.backward(self, output_T.grad)
        
        output_T._backward = _backward
        return output_T
    
    def mean(self)-> 'Tensor':
        output_T = Tensor(TensorOps.MEAN.forward(self), (self, ))

        def _backward():
            self.grad = TensorOps.MEAN.backward(self, output_T.grad)

        output_T._backward = _backward
        return output_T
    
    def sqrt(self)-> 'Tensor':
        output_T = Tensor(TensorOps.SQRT.forward(self), (self,))

        def _backward():
            self.grad  = TensorOps.SQRT.backward(self, output_T.grad , output_T.data)

        output_T._backward = _backward
        return output_T
    #                                            UNARY transformation                                          -
    def abs(self) -> 'Tensor':   
        output_T =  Tensor(TensorOps.ABS.forward(self), (self,))
        def _backward():
            self.grad  = TensorOps.ABS.backward(self, output_T.grad)

        output_T._backward = _backward
        return output_T 
                 
    def T(self) -> 'Tensor':                                                  
        output_T = Tensor(TensorOps.TR.forward(self), (self,))
        def _backward():
            self.grad = TensorOps.TR.backward(self, output_T.grad)                                        #TODO find a nicer way to do this

        output_T._backward  = _backward
        return output_T
    
    def unsqueeze(self, axis) -> 'Tensor':
        return Tensor(np.expand_dims(self.data, axis = axis))
    #                                                DOT                                                     - 
    def dot(self, other) -> 'Tensor':
        other = other if isinstance(other , Tensor) else Tensor(other)
        output_T  = Tensor(TensorOps.DOT.forward(self, other), (self, other))

        def _backward():
            self.grad ,other.grad = TensorOps.DOT.backward(self, output_T.grad)

        output_T._backward = _backward
        return output_T
    #                                               Activation functions                                      - 
    def ReLU(self) -> 'Tensor':
        output_T = Tensor(TensorOps.RELU.forward(self), (self,))

        def _backward():
            self.grad = TensorOps.RELU.backward(self, output_T.grad)

        output_T._backward = _backward
        return output_T
    
    def Sigmoid(self) -> 'Tensor':
        output_T = Tensor(TensorOps.SIGMOID.forward(self), (self,))

        def _backwrad():
            self.grad = TensorOps.SIGMOID.backward(self, output_T.grad)

        output_T._backward = _backwrad
        return output_T
    
    def Tanh(self) -> 'Tensor':
        output_T =Tensor(TensorOps.TANH.forward(self,), (self,))

        def _backward():
            self.grad = TensorOps.TANH.backward(self, output_T.grad)
        
        output_T._backward = _backward
        return output_T
    
    def Softmax(self, axis = None) -> 'Tensor' :                             # https://stackoverflow.com/questions/42599498/numerically-stable-softmax https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy
        output_T = Tensor(TensorOps.SOFTMAX.forward(self, axis = axis), (self, ))

        def _backward():                                       
            self.grad = TensorOps.SOFTMAX.backward(self, output_T.grad)

        output_T._backward = _backward
        return output_T
    
    # TODO maybe add logsoftmax

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
        
        self.grad = 1
        for node in reversed(topo):
            node._backward()
            node.grad = node.grad  if isinstance(node.grad, Tensor) else Tensor(node.grad)
