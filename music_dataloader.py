import itertools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ClassDictionary:
    def __init__(self, data_fp):
        self.vocab_set = None
        self.index_to_class = None
        self.class_to_index = None
        self.make_lookup(data_fp)
        self.len = len(self.vocab_set)

    def __len__(self):
        return self.len

    def __getitem__(self, id):
        if type(id) is torch.Tensor:
            id = int(id.detach().cpu().numpy())
        if type(id) is str:
            if id not in self.vocab_set:
                return self.class_to_index['<unk>']
            return self.class_to_index[id]
        elif type(id) is int:
            return self.index_to_class[id]
        elif type(id) is float:
            if not id % 1:
                 return self.index_to_class[int(id)]
            else:
                print('ERROR: NON INTEGER FLOAT USED AS LOOKUP ID')
                return None

        print('ERROR: INVALID DATA TYPE USED AS LOOKUP ID')

        return None

    def make_lookup(self, fp):
        with open(fp, 'r') as f:
            data = list(itertools.chain(*[l for l in f.readlines() if l != '<end>' and l != '<start>']))
        self.vocab_set = set(data)
        self.vocab_set.add('<start>')
        self.vocab_set.add('<end>')
        self.vocab_set.add('<unk>')
        self.index_to_class = list(self.vocab_set)
        self.class_to_index = {v:i for i, v in enumerate(self.index_to_class)}


class MusicDataset(Dataset):
    def __init__(self, dictionary, fp):
        self.tunes = None
        self.dictionary = dictionary
        self.len = None

        self.load_tunes(fp)

    def __len__(self):
        return self.len

    def __getitem__(self, ind):
        x = torch.tensor(self.make_onehot(ind))
        y = torch.tensor(self.tunes[ind + 1]).long()

        return (x, y)

    def load_tunes(self, fp):
        with open(fp, 'r') as f:
            tokenized = [self.get_ids(l) for l in f.readlines()]
        self.tunes = list(itertools.chain(*tokenized))
        self.len = len(self.tunes) - 1

    def get_ids(self, string):
        if string == '<start>\n' or string == '<end>\n':
            string = [string[:-1], string[-1]]

        return [self.dictionary[c] for c in string]

    def make_onehot(self, ind):
        onehot = np.float32(np.zeros(len(self.dictionary)))
        onehot[self.tunes[ind]] = 1.

        return onehot


def create_split_loaders(seq_length):
    fp_train = 'data/train.txt'
    fp_val = 'data/val.txt'
    fp_test = 'data/test.txt'

    dictionary = ClassDictionary(fp_train)
    train_dataset = MusicDataset(dictionary, fp_train)
    val_dataset = MusicDataset(dictionary, fp_val)
    test_dataset = MusicDataset(dictionary, fp_test)
    
    train_loader = DataLoader(train_dataset, batch_size=seq_length)
    val_loader = DataLoader(val_dataset, batch_size=seq_length)
    test_loader = DataLoader(test_dataset, batch_size=seq_length)

    return train_loader, val_loader, test_loader, dictionary
