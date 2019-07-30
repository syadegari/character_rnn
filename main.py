from loaders import DataSet, DataLoader
import numpy as np
import importlib
import utils
import rnn_char
importlib.reload(utils)
importlib.reload(rnn_char)

from rnn_char import CharRNNv1, CharRNNv2, CharRNNv3, CharRNNv4
from utils import Train

with open('data/anna.txt', 'r') as f:
    text = f.read()

nBatch = 4
nSeq = 5
frac = 0.2


dSet = DataSet(text[:20000], nBatch, nSeq, frac)

train_loader = DataLoader(dSet('train'), dSet.nBatchSize, dSet.nSeqLength, dSet.nLabel, mode='many-1')
test_loader = DataLoader(dSet('test'), dSet.nBatchSize, dSet.nSeqLength, dSet.nLabel, mode='many-1')
rnnv1 = CharRNNv1(dSet.nLabel, 32, dSet.nLabel, dSet.encoder, nLayer=2, dropout=0.2)
train = Train(rnnv1, print_every=100, epochs=5)
hist_train, hist_test = train(train_loader, test_loader)

rnnv2 = CharRNNv2(dSet.nLabel, 32, dSet.nLabel, dSet.encoder, nLayer=2, dropout=0.2)
train = Train(rnnv2, print_every=100, epochs=5)
hist_train, hist_test = train(train_loader, test_loader)

rnnv3 = CharRNNv3(nSeq, dSet.nLabel, 32, dSet.nLabel, dSet.encoder, nLayer=2, dropout=0.2)
train = Train(rnnv3, print_every=100, epochs=5)
hist_train, hist_test = train(train_loader, test_loader)

train_loader = DataLoader(dSet('train'), dSet.nBatchSize, dSet.nSeqLength, dSet.nLabel, mode='many-many')
test_loader = DataLoader(dSet('test'), dSet.nBatchSize, dSet.nSeqLength, dSet.nLabel, mode='many-many')
rnnv4 = CharRNNv4(dSet.nLabel, 32, dSet.nLabel, dSet.encoder, nLayer=2, dropout=0.2)

train = Train(rnnv4, print_every=100, epochs=5)
hist_train, hist_test = train(train_loader, test_loader)

rnnv1.sample(400, prime='hello world', top_k=5)
rnnv2.sample(400, prime='hello world', top_k=5)
rnnv3.sample(400, prime='hello world', top_k=5)
rnnv4.sample(400, prime='hello world', top_k=5)
