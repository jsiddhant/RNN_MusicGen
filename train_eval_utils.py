from datetime import datetime as dt
import itertools
import numpy as np
import os
import pickle
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from lstm import LSTM
from music_dataloader import create_split_loaders
from torch_utils import setup_device

def fit_rnn(model, criterion, optimizer, train_loader, val_loader, n_epochs, model_name, computing_device,
            seq_length=100, chkpt_every=25, update_hist=50, val_every=35000):
    train_losses = dict()
    val_losses = dict()
    total_seen = 0
    start_time = dt.now()
    min_val_loss = torch.tensor(np.inf)

    # Make the directory to save the model and the losses
    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('train-stats'):
        os.mkdir('train-stats')
    model_save_path = os.path.join('models', model_name + '.pt')
    train_save_path = os.path.join('train-stats', model_name + '_train.pkl')
    val_save_path = os.path.join('train-stats', model_name + '_val.pkl')

    for epoch in np.arange(n_epochs):
        train_losses[epoch] = []
        val_losses[epoch] = []
        for i, (x, y) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            x, y = x.to(computing_device), y.to(computing_device)
            g = torch.squeeze(model(torch.unsqueeze(x, 0)))
            loss = criterion(g, y)
            loss.backward()
            optimizer.step()
            train_losses[epoch].append(loss.detach().cpu().numpy())
            total_seen += 1

            # Report training stats
            avg_loss = np.mean(train_losses[epoch][-update_hist:])
            time_delta = dt.now() - start_time
            update_str = '[TIME ELAPSED]: {0} [EPOCH {1}]: Avg. loss for last {2} minibatches: {3:0.5f}'
            print(update_str.format(str(time_delta),epoch + 1, chkpt_every, avg_loss), end='\r')

            # Save the model and the training and validation losses
            if not total_seen % val_every:
                # Validate the model
                with open(train_save_path, 'wb') as f:
                    pickle.dump(train_losses, f)
                with open(val_save_path, 'wb') as f:
                    pickle.dump(val_losses, f)
                val_losses[epoch].append(evaluate_model(model, val_loader, criterion, computing_device,
                                                        start_time))
                
                if val_losses[epoch][-1] < min_val_loss:
                    torch.save(model.state_dict(), model_save_path)
                else:
                    return
                

        print(f'\n[EPOCH {epoch + 1}]: Avg. loss for epoch: {np.mean(train_losses[epoch])}\n')

def evaluate_model(model, loader, criterion, computing_device, start_time):
    model.eval()
    val_losses = []
    T = len(loader)

    print(' ' * 120, end='\r')

    for i, (x, y) in enumerate(loader):
        x = x.to(computing_device)
        y = y.to(computing_device)
        val_losses.append(criterion(torch.squeeze(model(torch.unsqueeze(x, 0))), y))
        print(f'Validating model: {i}/{T}', end='\r')

    return torch.mean(torch.tensor(val_losses))
