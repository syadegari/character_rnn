import numpy as np
import torch
from utils import one_hot_encode

def decodeToChar(encoded, decoder):
    return [decoder[x] for x in encoded.reshape(-1)]


def encodeToInt(text, encoder):
    return np.array([encoder[x] for x in text])


def splitText(text, testFrac=0.2):
    # split to train and test
    trainText = text[: int(len(text) * (1 - testFrac))]
    testText = text[int(len(text) * (1 - testFrac)):]
    return trainText, testText


def clipText(trainText, testText, nBatchSize, nSeqLength):
    # clip the remaining
    clip = lambda text, m: text[: (len(text) // m) * m]
    return clip(trainText, nBatchSize * nSeqLength), \
           clip(testText, nBatchSize * nSeqLength)


def encodeText(trainText, testText, encoder):
    trainEncoded = encodeToInt(trainText, encoder)
    testEncoded = encodeToInt(testText, encoder)
    return trainEncoded, testEncoded


def get_char_encoder_decoder(text):
    chars = tuple(set(text))
    char2int = {ch: i for i, ch in enumerate(chars)}
    int2char = {i: ch for ch, i in char2int.items()}
    return char2int, int2char, len(chars)


def dataSet(text:str, encoder, nBatchSize, nSeqLength, testFrac=0.2):
    train, test = splitText(text, testFrac)
    train, test = clipText(train, test, nBatchSize, nSeqLength)
    train, test = encodeText(train, test, encoder)
    return train, test


class DataSet:
    """
    creates a data set of training and testing out of a given set
    text -->
    split_text (train, test) -->
    clip (so we have sizes of [batch x seq_length] when reshape the data later -->
    encoded both train and test sets
    """
    def __init__(self, text:str, batchSize:int, seqLength:int, frac=0.2):
        self.nBatchSize = batchSize
        self.nSeqLength = seqLength
        self.frac = frac
        self.encoder, self.decoder, self.nLabel = get_char_encoder_decoder(text)
        self.trainSet, self.testSet = dataSet(text, self.encoder, batchSize, seqLength, frac)

    def __call__(self, setName:str):
        if setName == 'train':
            return self.trainSet
        if setName == 'test':
            return self.testSet
        raise TypeError

class DataLoader:
    """
    Wrapper around trainset and testset
    y-values are x-values shifted by 1
    mode:
        many-many: y is the same length as x and shifted by 1 : For RNNs that use many-to-many models
        many-1   : y is the last entry of x (already shifted) : For RNNs that use many-to-one models
    """
    def __init__(self, data:str, nBatchSize:int, nSeqlength:int, nLabel:int, mode='many-many'):
        assert mode == 'many-many' or mode == 'many-1'
        self.mode = mode
        self.nBatchSize = nBatchSize
        self.nSeqlength = nSeqlength
        self.nLabel = nLabel
        self.length = len(data)
        self.data = self._reshapeBatchFirst_(data, nBatchSize)
        self.pointer = 0


    def _reshapeBatchFirst_(self, data, nBatchSize):
        return data.reshape(nBatchSize, -1)

    def __len__(self):
        return int(self.length / (self.nBatchSize * self.nSeqlength))

    def __iter__(self):
        return self

    def __next__(self):
        while self.pointer < self.length / self.nBatchSize:
            n = self.pointer
            x = self.data[:, n:n + self.nSeqlength]
            if n == self.length / self.nBatchSize - self.nSeqlength:
                # one shift in x does not work, last numbers in each batch are
                # from the beginning of the batch
                y = np.concatenate((self.data[:, n + 1:], self.data[:, 0:1],), axis=1)
            else:
                # y is x shifted by one
                y = self.data[:, n + 1: n + 1 + self.nSeqlength]
            self.pointer += self.nSeqlength
            if self.mode == 'many-many':
                return torch.from_numpy(one_hot_encode(x, self.nLabel)), \
                       torch.from_numpy(y).contiguous().view(-1)
            else:
                # only the last ones from each batch in many-1 mode
                return torch.from_numpy(one_hot_encode(x, self.nLabel)), \
                       torch.from_numpy(y)[:, -1].contiguous().view(-1)
        self.pointer = 0
        raise StopIteration
