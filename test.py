import torch
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.functional as F


x=torch.randn([3,10]).cuda()
x=Variable(x,requires_grad=True)#生成变量

fc=nn.Linear(10, 3).cuda()

y = fc(x)
print(y.size())

label=torch.randn([3,3]).cuda()
y.backward(label)
print(x.grad)#求对x的梯度

