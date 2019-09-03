from data_loader import *
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import random
import argparse
from torch.utils.data import DataLoader
from torch.multiprocessing import Process
import time
import math
from io import open
import glob
import os
import matplotlib.pyplot as plt
import unicodedata
import string
from torchtext.datasets import text_classification


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.005, type=int, help=
'learning rate')
parser.add_argument('--epoch', default=20, type=int, help=
'number of epochs to train')
parser.add_argument('--batch_size', default=16, type=int, help=
'batch size which will be divided to number of model instances')
parser.add_argument('--world_size', default=2, type=int, help=
'number of model instances to be run parallel')
parser.add_argument('--threshold', default=0.1, type=float, help=
'threshold for large gradients, range 0. - 1.')
parser.add_argument('--ngrams', default=2, type=int, help=
'ngrams, the words put together, usually 2-4')
args = parser.parse_args()

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# def partition_dataset():
#     train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
#         root='./data', ngrams=args.ngrams, vocab=None)
#     size = dist.get_world_size()
#     testloader = torch.utils.data.DataLoader(test_dataset,
#                                              batch_size=args.batch_size,
#                            shuffle=False, num_workers=args.world_size)
#     # split batch size in two equal parts
#     b_size = int(args.batch_size / float(size))
#     # partition dataset to the number of parallel instances
#     partition_size = [1.0 / size for _ in range(size)]
#     partition = DataPartitioner(train_dataset, partition_size)
#     partition = partition.use(dist.get_rank())
#     train_set = train_dataset.DataLoader(partition, batch_size=b_size,
#                                 shuffle=True)
#     return train_set, b_size, testloader

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=args.batchsize, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text, offsets, cls
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)

def run(rank, size):
    """ Distributed function to be implemented later. """

    pass
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

# def init_processes(rank, size, fn, backend='tcp'):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29500'
#     dist.init_process_group(backend, rank=rank, world_size=size)
#     fn(rank, size)
#
#
# if __name__ == "__main__":
#     size = 2
#     processes = []
#     for rank in range(size):
#         p = Process(target=init_processes, args=(rank, size, run))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
