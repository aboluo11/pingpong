import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def conv3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
    )

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #input size: 80*80
        self.convs = nn.Sequential(
            conv3x3(1, 4),  # 40*40
            conv3x3(4, 8), # 20*20
            conv3x3(8, 16),    # 10*10
            conv3x3(16, 32),    # 5*5
            Flatten(),
            nn.Linear(32*5*5, 32),
            nn.ReLU()
        )
        self.policy = nn.Linear(32, 1)
        self.value = nn.Linear(32, 1)
        self.gru = nn.GRUCell(32, 32)

    def forward(self, x, hidden):
        x = (x-x.mean())/x.std()
        x = self.convs(x)
        x = hidden = self.gru(x, hidden)
        p = torch.sigmoid(self.policy(x)).view(-1)
        value = self.value(x).view(-1)
        return p, value, hidden