import argparse
from torch import nn
from torch.optim import Adam

from lstm import LSTM
from music_dataloader import create_split_loaders
from torch_utils import setup_device
from train_eval_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--chkpt-every', type=int, dest='chkpt_every', default=100)
parser.add_argument('-d', '--n-layers', type=int, dest='n_layers', default=1)
parser.add_argument('-l', '--layer-size', type=int, dest='layer_size', default=100)
parser.add_argument('-lr', '--learning-rate', type=float, dest='lr', default=0.001)
parser.add_argument('-m', '--model-name', type=str, dest='model_name')
parser.add_argument('-n', '--n-epochs', type=int, dest='n_epochs', default=10)
parser.add_argument('-s', '--seq-lenght', type=int, dest='seq_length', default=100)
parser.add_argument('-u', '--update-hist', type=int, dest='update_hist', default=25)
parser.add_argument('-v', '--val_every', type=int, dest='val_every', default=1000)
args = parser.parse_args()

computing_device = setup_device()
train_loader, val_loader, test_loader, dictionary = create_split_loaders(args.seq_length)

lstm = LSTM(len(dictionary),
            args.layer_size,
            len(dictionary),
            computing_device,
            n_layers=args.n_layers)
lstm.to(computing_device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(lstm.parameters(), lr=0.001)

fit_rnn(lstm,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        args.n_epochs,
        args.model_name,
        computing_device,
        val_every=args.val_every,
        chkpt_every=args.chkpt_every,
        update_hist=args.update_hist)
