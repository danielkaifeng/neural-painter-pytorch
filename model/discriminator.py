import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
   
class discriminator_2d(nn.Module):
    def __init__(self, w, h, in_channels=3, mid_ch=64):
        super(discriminator_2d, self).__init__()
        self.w = w
        self.h = h

        self.mid_ch = mid_ch
        kernel_size = (4, 4)
        self.c0 = nn.Conv2d(in_channels, mid_ch, kernel_size, 2, 1)
        self.c1 = nn.Conv2d(mid_ch, mid_ch * 2, kernel_size, 2, 1)
        self.c2 = nn.Conv2d(mid_ch * 2, mid_ch * 4, kernel_size, 2, 1)
        self.c3 = nn.Conv2d(mid_ch * 4, mid_ch * 8, kernel_size, 2, 1)
        self.bn0 = nn.BatchNorm2d(mid_ch)
        self.bn1 = nn.BatchNorm2d(mid_ch * 2)
        self.bn2 = nn.BatchNorm2d(mid_ch * 4)
        self.bn3 = nn.BatchNorm2d(mid_ch * 8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 3, self.w, self.h)
        
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        
        h = self.sigmoid(h)

        h = h.view(h.size(0), -1)
        return torch.mean(h, 1)

if __name__ == "__main__":
    x = Variable(torch.randn(1, 3, 256, 256), requires_grad=True)
    dis = discriminator_2d()
    dis.cuda()

    dis.zero_grad()

    out = dis(x.cuda())

    #one = torch.ones(out.size()).cuda()
    one = torch.FloatTensor([1]).cuda()
    out.backward(one)
    
    print(x.size(), out.size())
    print(x.grad)

