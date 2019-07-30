# character_rnn
Different variants of character level rnn together with loader and training routines

Four different rnn architectures are considered:
  - v1 uses only the hidden state of the rnn (only one hidden vector is passed to fc layer)
  - v2 uses the average of all hidden states (only one hidden vector, which is the average all hidden states of the last layer is passed to fc layer)
  - v3 uses a weighted average of hidden layers. There is an extra parameter, with the lenght of the input parameter nSeqLength, that is learned along with the rest of the parameters in the network. Because of this added, fixed size parameter, it is important that the inference is being done with the same length of text as training is performed. For this reasone, the sampling method for this architecture is different from the rest of the tested models
  - v4 uses a many-to-many training, i.e., all hidden vectors of the last layer are passed to fc layer
