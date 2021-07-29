import oneflow.experimental as flow
import oneflow.experimental.nn as nn
from eager.transformer import Transformer
import numpy as np


class TransLM(nn.Module):
    """
    Transformer Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, trans: Transformer, vocab_size):
        """
        :param trans: Transformer model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.trans = trans
        self.next_sentence = NextSentencePrediction(self.trans.hidden)
        self.mask_lm = MaskedLanguageModel(self.trans.hidden, vocab_size)

    def forward(self, x, segment_label):

        x = self.trans(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: Transformer model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):  # x.shape >> flow.Size([16, 20, 256])
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of Transformer model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
