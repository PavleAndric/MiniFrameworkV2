#import numpy as np
#np.set_printoptions(precision = 5)
#
#class ADD():
#    
#    def forward(self, other):
#        self.other = other
#        return np.add(self.data, other.data)
#    
#    def backward(self, chain):
#        self.grad += chain
#        self.other.grad += chain
#        return self.grad, self.other.grad
#    
#class MUL():
#    def forward(self, other):
#        self.other = other
#        return np.multiply(self.data, other.data)
#    
#    def backward(self, chain):
#        self.grad  += np.multiply(self.other.data, chain)
#        self.other.grad += np.multiply(self.data, chain)
#        return self.grad, self.other.grad
#    
#class POW():
#    def forward(self, other):
#        self.other = other
#        self.out = np.power(self.data, other.data)  
#        return self.out 
#    
#    def backward(self, chain):
#        self.grad += self.other * (np.power(self.data, (self.other.data - 1))) *chain  
#        self.other.grad += np.log(self.data + 1e-8) * chain * self.out
#        return self.grad , self.other.grad
#    
#class SUB():
#    def forward(self, other):
#        self.other = other 
#        return np.subtract(self.data, other.data)
#    
#    def backward(self, chain):
#        self.grad += chain
#        self.other.grad += -chain
#        return self.grad, self.other.grad
#    
#class DOT():
#    def forward(self, other):                           # Z = 3 x 1 -> first.dot(z) get 3 x 1 back  for z  and 3, for first
#        self.other = other
#        self.output = np.dot(self.data, self.other.data) 
#        return self.output
#    
#    def backward(self, chain):
#        if np.ndim(self.data) < 2 :
#            self.data = np.reshape(self.data, (1, self.data.shape[0])) 
#        
#        if np.ndim(chain) < 2:
#            chain = np.reshape(chain, (chain.shape[-1], 1)) ##THIS ONE
#
#        self.grad += chain.dot(np.transpose(self.other.data))  # 1x3 * 3x1
#        self.other.grad += np.transpose(self.data).dot(chain)  # 3x1 1x1 
#
#        self.grad = np.reshape(self.grad, (self.data.shape))
#        self.other.grad = np.reshape(self.other.grad, (self.other.shape))
#
#        assert self.grad.shape == self.data.shape
#        assert self.other.grad.shape == self.other.shape
#        return self.grad , self.other.grad    
#
## ------------------------------UNARY_TRANSFORM-------------------------------#
#class SUM():
#    def forward(self): 
#        self.lol = self.data
#        return np.sum(self.data) ####changed 
#
#    def backward(self, chain):
#        
#        self.grad += np.ones_like(self.data) * chain
#        return self.grad
#
#class LOG():
#    def forward(self):
#        self.input = self.data + 1e-8
#        return np.log(self.input)
#    
#    def backward(self, chain):
#        self.grad += (1 / self.input) * chain
#        return self.grad
#    
#class MEAN():
#    def forward(self):
#        return np.mean(self.data)
#    
#    def backward(self, chain):
#        self.grad += np.ones_like(self.data) / self.data.size() * chain
#        return self.grad
#    
#class SQRT():
#    def forward(self):
#        self.output = np.sqrt(self.data)
#        return self.output
#    
#    def backward(self, chain):
#        self.grad += 0.5 * self.data * chain
#        return self.grad
#    
#class ABS():
#    def forward(self):
#        return np.abs(self.data)
#    
#    def backward(self, chain):
#        self.grad += np.sign(self.data) * chain
#        return self.grad
#        
#class TR():                                           #TRANSPOSE
#    def  forward(self):
#        return np.transpose(self.data)
#    
#    def backward(self, chain):
#        self.grad += np.transpose(np.inner(chain, np.ones_like(self.data)))
#        return self.grad
##------------------------------ACTIAVTIIONS--------------------------------------#
#class RELU():
#    def  forward(self):
#        self.output = np.maximum(0, self.data)
#        return self.output
#        
#    def backward(self, chain):
#             
#        self.grad += (self.output > 0) * chain
#        return self.grad
#
#class TANH():
#    def forward(self):
#        self.output = np.tanh(self.data)
#        return self.output
#    
#    def backward(self, chain):  
#        self.grad += (1-self.output**2) * chain
#        return self.grad
#    
#class SIGMOID():
#    def forward(self):
#        self.output = 1/(1 + np.exp(-self.data))
#        return self.output
#    
#    def  backward(self, chain):
#        self.grad +=  (self.output - self.output**2) * chain 
#        return self.grad
#    
#class SOFTMAX():
#    def forward(self, axis = None):
#        exp = np.exp(self.data  - np.max(self.data, axis = axis , keepdims = True )) 
#        self.output = exp / np.sum(exp, axis = axis, keepdims = True)
#        return self.output
#    
#    def backward(self, chain):
#        p = self.output.squeeze() ; ch = chain
#        p = p.reshape(-1,1)
#
#        if ch.shape[0] == 1:
#            ch = ch.T
#
#        self.grad += np.matmul(np.diagflat(p) - np.dot(p, p.T), ch)
#        if self.grad.shape != self.output.shape:
#            self.grad = self.grad.T
#
#        return self.grad
#
       

   

			