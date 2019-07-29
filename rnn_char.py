from abc import abstractmethod, ABC
from utils import Slider, one_hot_encode

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def sample_v1(self, length, prime='The', top_k=None):
    """samples characters, 1 char in, one char out"""
    chars = [_ for _ in prime]
    h = self.init_hidden(1)

    # prime the model
    for ch in prime:
        char, h = self.predict_char(ch, h, top_k=top_k)

    chars = char
    for i in range(length):
        char, h = self.predict_char(char, h, top_k=top_k)
        chars += char
    return chars


def sample_v2(self, length, prime, top_k=None):
    """samples characters, has to have a prime that is bigger than
    the length of sequence. Only used for CharRNNv3 where the sequence
    length has to be constant"""
    assert len(prime) >= self.seqLength
    L = self.seqLength
    h = self.init_hidden(1)
    slider = Slider(prime, self.seqLength)

    # prime the model
    for chars in slider:
        char, h = self.predict_char(chars, h, top_k=top_k)
        prime += char

    # generate the sequence
    for i in range(length):
        char, h = self.predict_char(prime[-L : ], h, top_k=top_k)
        prime += char

    return prime

class AbsCharRNN(nn.Module, ABC):
    def __init__(self, inputDim, hiddenDim, outputDim, encoder, **kwargs):
        """
        kwargs:      tpye    default
            nLayer   (int)   [1]
            dropout  (float) [0.0]
        """
        super(AbsCharRNN, self).__init__()

        # kwargs
        self.numLayers = kwargs.pop('nLayer', 1)
        self.dropout = kwargs.pop('dropout', 0.0)
        self.device = kwargs.pop('device', self.__device__())

        if kwargs:
            print('Unkonw argument', kwargs)

        self.input_size = inputDim
        self.hidden_size = hiddenDim
        self.output_size = outputDim
        self.encoder = encoder
        self.bidirectional = False
        self.batchFirst = True

        self.lstm = nn.LSTM(input_size=inputDim, hidden_size=hiddenDim, dropout=self.dropout,
                            num_layers=self.numLayers, batch_first=self.batchFirst)
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(hiddenDim, outputDim)

    def init_hidden(self, nBatchSize):
        h0 = torch.zeros(self.numLayers, nBatchSize, self.hidden_size)
        c0 = torch.zeros(self.numLayers, nBatchSize, self.hidden_size)
        return h0.to(self.device), c0.to(self.device)

    @staticmethod
    def __device__():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __lstmPass__(self, x, h0=None):
        if h0 == None:
            hs, hn = self.lstm(x)
        else:
            hs, hn = self.lstm(x, h0)
        return hs, hn

    def predict_char(self, seq, h, top_k=None):
        decoder = {i: char for (char, i) in self.encoder.items()}
        nLabel = len(self.encoder)

        one_hot = one_hot_encode(np.array([self.encoder[ch] for ch in seq])[None, :], nLabel)
        y, h = self(torch.from_numpy(one_hot), h)
        h = tuple([_.detach() for _ in h])
        p = F.softmax(y, dim=1)

        if top_k == None:
            top_ch = np.arange(nLabel)
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.squeeze()

        p = p.detach().squeeze().numpy()
        return decoder[np.random.choice(top_ch, p=p / np.sum(p))], h


    @abstractmethod
    def __call__(self, x, h=None):
        raise NotImplemented

    @abstractmethod
    def sample(self, size, prime, top_k=None):
        raise NotImplemented


class CharRNNv1(AbsCharRNN):
    """only the last hidden variable is passed to fc"""

    sample = sample_v1

    def __call__(self, x, h0=None):
        hs, hn = self.__lstmPass__(x, h0)
        hs = hs[:, -1, :].contiguous()
        # hs = hs.contiguous().view(-1, hs.size(2))
        return self.fc(self.dropout(hs)), hn


class CharRNNv2(AbsCharRNN):
    """average of hidden variables are passed to fc"""

    sample = sample_v1

    def __call__(self, x, h0=None):
        hs, hn = self.__lstmPass__(x, h0)
        hs = hs.mean(dim=1).contiguous()
        return self.fc(self.dropout(hs)), hn


class CharRNNv3(AbsCharRNN):
    """hidden variables are averaged using a weight vector alpha.
     requires a constant sequence length for both training and inference (equal to seqLength)"""
    def __init__(self, seqLength, *args, **kwargs):
        super(CharRNNv3, self).__init__(*args)
        self.seqLength = seqLength
        self.alpha = nn.Parameter(torch.rand(seqLength, dtype=torch.float, requires_grad=True))

    sample = sample_v2

    def __call__(self, x, h0=None):
        assert x.size(1) == self.seqLength, f'sequence length is {x.size(1)}, must be {self.seqLength}'
        hs, hn = self.__lstmPass__(x, h0)
        # each hs is multiplied with values from alpha along L (sequence) axis
        hs = hs * self.alpha[None, :, None]
        # sum of the sequence
        hs = torch.sum(hs, dim=1).contiguous()
        # hs is now averaged and can be fed into the fc layer
        return self.fc(self.dropout(hs)), hn

class CharRNNv4(AbsCharRNN):
    """all hidden variables are used without modification.
    batch size for both x and y has the same length"""

    sample = sample_v1

    def __call__(self, x, h0=None):
        hs, hn = self.__lstmPass__(x, h0)
        hs = hs.contiguous()
        return self.fc(self.dropout(hs.view(-1, hs.size(2)))), hn
