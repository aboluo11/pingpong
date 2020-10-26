import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def conv3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
        # nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    )

class Model(nn.Module):
    def __init__(self, gru_sz):
        super().__init__()
        #input size: 80*80
        self.convs = nn.Sequential(
            conv3x3(1, 8),  # 40*40
            conv3x3(8, 16), # 20*20
            conv3x3(16, 32),    # 10*10
            conv3x3(32, 64),    # 5*5
            Flatten(),
            nn.Linear(64*5*5, gru_sz),
            nn.ReLU()
        )
        self.policy = nn.Linear(gru_sz, 1)
        self.critic = nn.Linear(gru_sz, 1)
        self.gru = nn.GRUCell(gru_sz,gru_sz)

    def forward(self, x, state):
        x = (x-x.mean())/x.std()
        x = self.convs(x)
        x = state = self.gru(x, state)
        p = self.policy(x).view(-1)
        p = torch.sigmoid(p)
        value = self.critic(x).view(-1)
        return p, value, state
