from loaders import DataSet, DataLoader
import numpy as np
import imp
import utils
import rnn_char
imp.reload(utils)
imp.reload(rnn_char)

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

# def predictChar(model, seq, h, encoder, top_k=None):
#     decoder = {i: char for (char, i) in encoder.items()}
#     nLabel = len(encoder)
#
#     one_hot = one_hot_encode(np.array([encoder[ch] for ch in seq])[None, :], nLabel)
#     y, h = model(torch.from_numpy(one_hot), h)
#     h = tuple([_.detach() for _ in h])
#     p = F.softmax(y, dim=1)
#
#     if top_k == None:
#         top_ch = np.arange(nLabel)
#     else:
#         p, top_ch = p.topk(top_k)
#         top_ch = top_ch.squeeze()
#
#     p = p.detach().squeeze().numpy()
#     return decoder[np.random.choice(top_ch, p=p/np.sum(p))], h



rnnv1.sample(20)
rnnv2.sample(40, prime='hello world', top_k=5)
rnnv3.sample(40, prime='Hello', top_k=3)
rnnv4.sample(40, top_k=5)
