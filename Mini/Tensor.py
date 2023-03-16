import numpy as np
import TensorOps
np.set_printoptions(precision = 5)  # this is ugly here

class Tensor():

    def __init__(self, data, _children = () , label = ""):
        self.ALL_PARAMS = []
        self.label = label
        self.data = data if isinstance(data, (np.ndarray, np.generic)) else np.array(data, dtype = np.float32)
        self.shape = self.data.shape
        self._backward = lambda : None
        self.prev = set(_children)
        self.grad = 0.0

    def __repr__(self):
        return f"<Tensor = {self.data}>"

    def size(self)-> int: return self.data.size

    def  __getitem__(self, index):
        return Tensor(self.data[index])

    def __add__(self, other)-> 'Tensor':

        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T = Tensor(np.add(self.data, other.data), (self, other))
        

        def _backward():      
            self.grad  += Tensor(output_T.grad.data)
            other.grad += Tensor(output_T.grad.data)

        output_T._backward = _backward
        return output_T

    def __mul__(self, other)-> 'Tensor': 
        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T = Tensor(np.multiply(self.data, other.data), (self, other))

        def _backward():
            self.grad += Tensor(np.multiply(other.data,output_T.grad.data))
            other.grad += Tensor(np.multiply(self.data, output_T.grad.data))

        output_T._backward = _backward
        return output_T
    
    def __pow__(self, other) -> 'Tensor':                                       #https://testbook.com/learn/maths-derivative-of-exponential-function
        other = other if isinstance(other, Tensor) else Tensor(other)
        output_T = Tensor(np.power(self.data, other.data), (self, other))

        def _backward():
            self.grad += Tensor(other.data * (np.power(self.data, (other.data - 1))) *output_T.grad.data)
            other.grad += Tensor(np.log(self.data + 1e-8) * output_T.grad.data * output_T.data)

        output_T._backward = _backward
        return output_T

    def __sub__(self, other)-> 'Tensor':
        other  = other if isinstance(other , Tensor) else Tensor(other)
        output_T = Tensor(self.data - other.data, (self, other))

        def _backward():
            self.grad += Tensor(output_T.grad.data)
            other.grad += Tensor(output_T.grad.data * -1)

        output_T._backward = _backward
        return output_T 
    
    def __radd__(self, other) -> 'Tensor': return self + other
    def __rmul__(self, other)-> 'Tensor': return self * other
    def __rsub__(self, other)-> 'Tensor': return other + (self * -1)
    def __truediv__(self, other)-> 'Tensor': return self * (other **-1)
    def __rtruediv__(self, other)-> 'Tensor': return other * (self**-1)
    def __neg__(self)-> 'Tensor':  return self * -1                            # TODO this may couse errors

    #-                                             UNARY      math                                     -
    def sum(self) -> 'Tensor':
        output_T = Tensor(np.sum(self.data), (self, ))
  
        def _backward():
            self.grad += Tensor(np.ones_like(self.data) * output_T.grad.data)

        output_T._backward = _backward
        return output_T
    
    def log(self)-> 'Tensor':
        output_T = Tensor(np.log(self.data + 1e-8) (self,))

        def _backward():
            self.grad +=  Tensor((1 / output_T.data) * output_T.grad.data)
        
        output_T._backward = _backward
        return output_T
    
    def mean(self)-> 'Tensor':
        output_T = Tensor(np.mean(self.data), (self,))

        def _backward():
            self.grad += Tensor(np.ones_like(self.data) / self.data.size() * output_T.grad.data)

        output_T._backward = _backward
        return output_T
    
    def sqrt(self)-> 'Tensor':
        output_T = Tensor(np.sqrt(self.data), (self, ))

        def _backward():
            self.grad  += Tensor( 0.5 * self.data * output_T.grad.data)

        output_T._backward = _backward
        return output_T
    #                                            UNARY transformation                                          -
    def abs(self) -> 'Tensor':   
        output_T =  Tensor(np.abs(self.data) , (self,))
        def _backward():
            self.grad  = Tensor(np.sign(self.data) * output_T.grad.data)

        output_T._backward = _backward
        return output_T 
                 
    def T(self) -> 'Tensor':                                                  
        output_T = Tensor(np.transpose(self.data), (self, ))
        def _backward():
            self.grad = Tensor(np.transpose(np.inner(output_T.grad.data, np.ones_like(self.data))))                                   #TODO find a nicer way to do this

        output_T._backward  = _backward
        return output_T
    
    def unsqueeze(self, axis) -> 'Tensor':
        return Tensor(np.expand_dims(self.data, axis = axis))
    #                                                DOT                                                     - 
    def dot(self, other) -> 'Tensor':
        other = other if isinstance(other , Tensor) else Tensor(other)
        output_T  = Tensor(self.data.dot(other.data), (self, other))
        
        def _backward():
            if np.ndim(self.data) < 2 :
                self.data = np.reshape(self.data, (1, self.data.shape[0])) 
        
            if np.ndim(output_T.grad.data) < 2:
                output_T.grad.data = np.reshape(output_T.grad.data, (output_T.grad.data.shape[-1], 1)) ##THIS ONE
        
            self.grad += output_T.grad.data.dot(np.transpose(other.data))  # 1x3 * 3x1
            other.grad += np.transpose(self.data).dot(output_T.grad.data)  # 3x1 1x1 
    
            self.grad = Tensor(np.reshape(self.grad, (self.data.shape)))
            other.grad = Tensor(np.reshape(other.grad.data, (other.shape)))
    
            assert self.grad.shape == self.data.shape
            assert other.grad.shape == other.shape  

        output_T._backward = _backward
        return output_T
    #                                               Activation functions                                      - 
    def ReLU(self) -> 'Tensor':
        output_T = Tensor(np.maximum(0, self.data), (self, ))

        def _backward():
            self.grad += Tensor((output_T.data > 0) * output_T.grad.data)

        output_T._backward = _backward
        return output_T
    
    def Sigmoid(self) -> 'Tensor':
        output_T = Tensor(1/(1 + np.exp(-self.data)), (self, ))

        def _backwrad():
            self.grad +  (output_T.data - output_T.data**2) * output_T.grad.data 

        output_T._backward = _backwrad
        return output_T
    
    def Tanh(self) -> 'Tensor':
        output_T =Tensor(np.tanh(self.data))

        def _backward():
            self.grad += Tensor((1- output_T.data**2) * output_T.grad.data)
        
        output_T._backward = _backward
        return output_T
    
    def Softmax(self, axis = None) -> 'Tensor' :                             # https://stackoverflow.com/questions/42599498/numerically-stable-softmax https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy
        output_T = Tensor(TensorOps.SOFTMAX.forward(self, axis = axis), (self, ))

        def _backward():                                       
            self.grad = TensorOps.SOFTMAX.backward(self, output_T.grad)   # FIX THIS

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
        
        self.grad = Tensor([1.])
        
        for node in reversed(topo):
            #print(f"class_grad = {type(node.grad)}, class_node = {type(node)}, label  = {node.label}" )
            node._backward()
            #node.grad = node.grad  if isinstance(node.grad, Tensor) else Tensor(node.grad)
        self.ALL_PARAMS = [x for x in topo]
    
class Optim():

    def __init__(self , parameters, learning_rate):
        self.params = [x for x in parameters]
        self.lr = learning_rate
        
    def zero_grad(self, ALL_PARAMS):
        
        for x in ALL_PARAMS:
            x.grad = 0.0
    
    def step(self):
    
        for param in self.params:
            param.data  = param.data - (self.lr  * param.grad)
            #print(f"{param.data - (self.lr  * param.grad)}")
            
            
