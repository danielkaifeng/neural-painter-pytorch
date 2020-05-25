import torch
import torch as T
import torch.nn.functional as F
from torch import nn
#from torch.optim import Adam
#from torch.autograd import Variable, grad


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size=100, nr_hidden=3):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        kernel_size = (3, 3)
        self.conv1 = nn.Conv2d(3, hidden_size, kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.bn3 = nn.BatchNorm2d(hidden_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return x

class linear_net(nn.Module):
    def __init__(self, nonlin, hidden_size=100, nr_hidden=3,
             w=100, h=100,
             input_dim=2,
             output_dim=1, recurrent=False,
             output_nonlin=lambda x: x):

        super(linear_net, self).__init__()

        self.w = w
        self.h = h
        self.nonlin = nonlin
        self.hidden_size = hidden_size
        self.nr_hidden = nr_hidden
        self.output_nonlin = output_nonlin

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

        kernel_size = (3, 3)
        self.conv1 = nn.Conv2d(output_dim, hidden_size, kernel_size, 1, 1)
        #self.conv1 = nn.Conv2d(output_dim, output_dim, kernel_size, 1, 1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size, 1, 1)
        self.conv3 = nn.Conv2d(hidden_size, output_dim, kernel_size, 1, 1)
        self.bn0 = nn.BatchNorm2d(hidden_size)
        self.bn1 = nn.BatchNorm2d(output_dim)

    def forward(self, v):
        v = v.view(-1, self.w, self.h, 3)
        v = v.permute(0, 3, 1, 2)
        v = self.bn0(self.conv1(v))
        v = self.nonlin(v)
        v = self.conv2(v)
        v = self.nonlin(v)
        v = self.conv2(v)
        v = self.nonlin(v)
        v = self.bn1(self.conv3(v))
        v = self.nonlin(v)
        #v = self.conv1(v)
        #v = self.conv3(v)
        #v = self.nonlin(v)
        v = v.permute(0, 2, 3, 1)

        v = v.view(self.w*self.h, 3)

        v = self.fc1(v)
        
        for i in range(self.nr_hidden):
            v = self.fc2(v)
            v = self.nonlin(v)

        v = self.fc3(v)
        v = self.nonlin(v)

        v = (v - v.min(dim=0, keepdim=True).values) / (
            v.max(dim=0).values - v.min(dim=0).values + 1e-8)

        #v = T.sigmoid(v)
        v = v.view(-1, self.w, self.h, 3)

        return v
