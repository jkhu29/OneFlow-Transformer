import oneflow.experimental as flow
import oneflow.experimental.nn as nn
import numpy as np
import math


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = flow.Tensor(flow.zeros(size=(max_len, d_model)), requires_grad=False)

        position = flow.arange(0, max_len, dtype=flow.float).unsqueeze(1)
        div_term = (
            flow.arange(0, d_model, 2, dtype=flow.float)
            * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = flow.sin(position * div_term)
        pe[:, 1::2] = flow.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", flow.Tensor(pe))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)
