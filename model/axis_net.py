import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable, grad


class Axis(nn.Module):
    def __init__(self, hidden_size, w, h):
        super().__init__()
        kernel_size = (3, 3)
        output_dim = 3

        self.conv1 = nn.Conv2d(3, hidden_size, kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_size, hidden_size, kernel_size, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.bn3 = nn.BatchNorm2d(hidden_size)

        self.fc0 = nn.Linear(w//16 * h//16 * 3, hidden_size)
        self.fc01 = nn.Linear(hidden_size, 2)
        self.fc02 = nn.Linear(1, hidden_size)
        self.fc03 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))

        x = torch.flatten(x, 1) #[w/16 * h/16 * output_dim]

        #x = F.relu(self.fc0(x)) #[1, hidden_size]
        x = x.permute(1,0)
        x = F.relu(self.fc02(x))
        #x = F.relu(self.fc01(x))
        #x = x.permute(1,0) #[input_dim, hidden_size]
        #x = torch.sigmoid(x)
        #x = torch.tanh(x)

        return x

class Pos(nn.Module):
    def __init__(self, nonlin, hidden_size=100, nr_hidden=3,
             w=100, h=100,
             input_dim=2,
             output_dim=1, 
             output_nonlin=lambda x: x):

        super().__init__()

        self.w = w
        self.h = h
        self.hidden_size = hidden_size
        self.nr_hidden = nr_hidden
        self.output_nonlin = output_nonlin
        self.input_dim = input_dim
        self.nonlin = nonlin

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc20 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

        #self.axis = Axis(hidden_size)
            

    def forward(self, v, w):
        #v = self.fc1(v)
        v = w
        v = F.relu(self.fc2(v))
        v = self.nonlin(v)
        #print(v.size(), w.size())
        #v = torch.cat((v,w), 1)
        #v = F.relu(self.fc20(v))
        v = self.nonlin(v)

        for i in range(self.nr_hidden):
            v = F.relu(self.fc2(v))
            v = self.nonlin(v)

        v = F.relu(self.fc3(v))

        v = (v - v.min(dim=0, keepdim=True).values) / (
            v.max(dim=0).values - v.min(dim=0).values + 1e-8)
        v = v.view(-1, self.w, self.h, 3)
        #v = torch.sigmoid(v)

        return v
