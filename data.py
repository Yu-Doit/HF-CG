import os
from collections import Counter, OrderedDict

import torch
import numpy as np


class Vocab(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, vocab = None, vocab_size = None):
        self.vocab = Vocab()
        self.vocab_size = vocab_size
        self.vocab_prvd = False
        
        if vocab is not None:
            with open(vocab, 'r', encoding = 'utf8') as f:
                for word in f:
                    self.vocab.add_word(word.strip())
            self.vocab_prvd = True

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        

    def tokenize(self, path):
        if not self.vocab_prvd:
            counter = Counter()
            with open(path, 'r', encoding = 'utf8') as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    counter.update(words)
            for word, count in counter.most_common(self.vocab_size):
                self.vocab.add_word(word)
            self.vocab.add_word('<unk>')

        print('Build vocabulary of size {} done.'.format(len(self.vocab)))

        with open(path, 'r', encoding = 'utf8') as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    if word in self.vocab.word2idx.keys():
                        ids.append(self.vocab.word2idx[word])
                    else:
                        ids.append(self.vocab.word2idx['<unk>'])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


class DataIterator(object):
    def __init__(self, data, bsz, bptt):
        self.bsz = bsz
        self.bptt = bptt
        
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        self.data = data.view(bsz, -1).t().contiguous()
    
    def get_batch(self, i, bptt = None):
        if bptt is None:
            bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        data = self.data[i: i + seq_len]
        target = self.data[i + 1: i + 1 + seq_len].view(-1)

        return data, target, seq_len

    def get_fixlen_iter(self, start = 0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)
    
    def get_varlen_iter(self, start = 0, std = 5, min_len = 5, max_deviation = 3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()

    def __len__(self):
        l, _ = self.data.size()
        return l