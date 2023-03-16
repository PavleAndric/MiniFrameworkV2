
from Tensor import Tensor, Optim
class xor_net_mini():

    def __init__(self):
        self.L1 = Tensor.randn((2,3)) * 0.01
        self.L2 = Tensor.randn((3,1)) * 0.01
        
    def forward(self, x):
        out = (x.dot(self.L1)).ReLU().dot(self.L2)
        return out

xor_mini = xor_net_mini()
optim1 = Optim((xor_mini.L1, xor_mini.L2,), learning_rate = 0.1)


X_data_mini = Tensor([[1.,0.], [1., 1.], [0., 0.], [0.,1.]])
Y_data_mini = Tensor([[1.], [0.], [0.], [1.]])

for i in range(1):
    for x, y in zip(X_data_mini ,Y_data_mini):
        logits = xor_mini.forward(x)
        
        loss = 1/4 * ((y -logits)**2).sum()

        optim1.zero_grad(loss.ALL_PARAMS)
        
        loss.backward()
        
        optim1.step()

    
data = Tensor([1,0]) ; data.label = "data"
target = Tensor([1])    ; target.label = "target"

L1= Tensor.uniform((2,3)) * 0.1  ; L1.label = "L1"
L2= Tensor.uniform((3,1)) * 0.1   ; L2.label = "L1"

first = data.dot(L1) ; first.label = "first"
second = first.ReLU() ; second.label = "second"
out = second.dot(L2)  ; out.label = 'out'

loss_one = target - out         ; loss_one.label = "loss_one"
loss_two = loss_one ** 2            ; loss_two.label = "loss_two"
loss_thre  = loss_two.sum()         ; loss_thre.label = 'loss_thre'
loss_final = loss_thre * 0.25         ; loss_final.label = 'loss_final'

loss_final.backward()


opt = Optim((L1, L2), 0.1)
opt.step()
opt.zero_grad(loss_final.ALL_PARAMS)