from torch import nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, layer_size, output_size, n_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, layer_size, n_layers, batch_first=True)
        self.h_0 = None
        self.c_0 = None
        self.linear = nn.Linear(layer_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, x):
        if self.h_0 is not None:
            x, (self.h_0, self.c_0) = self.lstm(x, (self.h_0, self.c_0))
        else:
            x, (self.h_0, self.c_0) = self.lstm(x)
        x = self.linear(x)
        g = self.softmax(x)

        self.h_0 = Variable(self.h_0.data, requires_grad=False)
        self.c_0 = Variable(self.c_0.data, requires_grad=False)

        return g

    def reset_state(self):
        self.h_0, self.c_0 = (None, None)
