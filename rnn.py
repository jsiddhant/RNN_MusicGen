import torch
from torch import nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, layer_size, output_size, computing_device, n_layers=1):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.computing_device = computing_device
        self.reset_state()

        self.rnn = nn.RNN(input_size=input_size,
                            hidden_size=layer_size,
                            num_layers=n_layers,
                            batch_first=True).to(self.computing_device)
        self.linear = nn.Linear(layer_size, output_size).to(self.computing_device)
        nn.init.xavier_normal_(self.linear.weight)

    def __call__(self, x):
        x, h_t = self.rnn(x, self.h_0)
        x = self.linear(x)

        self.h_0 = Variable(h_t, requires_grad=False)

        return x

    def reset_state(self):
        self.h_0 = Variable(torch.zeros(self.n_layers, 1, self.layer_size).to(self.computing_device), requires_grad=False)
