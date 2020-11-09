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
    def __init__(self, gru_sz):
        super().__init__()
        #input size: 80*80
        first_layer_channel = 8
        self.convs = nn.Sequential(
            conv3x3(1, first_layer_channel),  # 40*40
            conv3x3(first_layer_channel, first_layer_channel*2), # 20*20
            conv3x3(first_layer_channel*2, first_layer_channel*4),    # 10*10
            conv3x3(first_layer_channel*4, first_layer_channel*8),    # 5*5
            Flatten(),
            nn.Linear(first_layer_channel*8*5*5, gru_sz),
            # nn.BatchNorm1d(gru_sz),
            nn.ReLU()
        )
        self.policy = nn.Linear(gru_sz, 1)
        self.critic = nn.Linear(gru_sz, 1)
        self.rnn = nn.Sequential(
            nn.Linear(2*gru_sz, gru_sz),
            nn.Tanh(),
        )

    def forward(self, x, state):
        # x = (x-x.mean())/x.std()
        # x = x / 255
        # x = (x-x.mean())/255
        # x = x/x.std()
        x = (x-144.07124)/4.6154547
        x = self.convs(x)
        state = self.rnn(torch.cat([x, state], dim=1))
        p = self.policy(state).view(-1)
        p = torch.sigmoid(p)
        value = self.critic(state).view(-1)
        return p, value, state
