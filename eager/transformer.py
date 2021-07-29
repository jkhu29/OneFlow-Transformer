import oneflow.experimental.nn as nn
import oneflow.experimental as flow

import copy, math
from eager.embedding import PositionalEmbedding
from eager.attention.multi_head import MultiHeadedAttention
from eager.utils.sublayer import SublayerConnection
from eager.utils.feed_forward import PositionwiseFeedForward
from eager.utils.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    Encoder
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):

        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)
        )
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)

    
class EncoderLayer(nn.Module):
    def __init__(self, hidden, attn_heads, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(hidden)
        self.norm2 = LayerNorm(hidden)
        self.attn = MultiHeadedAttention(attn_heads, hidden, dropout=dropout)
        self.ff = PositionwiseFeedForward(hidden, d_ff=hidden*4, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x1 = self.norm1(x)
        x = x + self.dropout1(self.attn(x1, x1, x1, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden, attn_heads, dropout) -> None:
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()
        self.norm1 = LayerNorm(hidden)
        self.norm2 = LayerNorm(hidden)
        self.norm2 = LayerNorm(hidden)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attn1 = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)        
        self.attn2 = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)        

        self.ff = PositionwiseFeedForward(
            d_model=hidden, d_ff=hidden*4, dropout=dropout
        )

    def forward(self, x, encoder_output, src_mask, trg_mask):
        x1 = self.norm1(x)
        x = x + self.dropout1(self.attn1(x1, x1, x1, trg_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.attn2(x2, encoder_output, encoder_output, src_mask))
        x3 = self.norm3(x)
        x = x + self.dropout3(self.ff(x3))
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, hidden, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = flow.zeros((int(max_seq_len), int(hidden)))
        for pos in range(int(max_seq_len)):
            for i in range(0, hidden, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i) / hidden)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1)) / hidden)))
        pe = pe.unsqueeze(0)
        # self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.hidden)
        #add constant to embedding
        seq_len = x.size(1)
        pe = flow.Tensor(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, hidden, attn_heads, dropout, N=6) -> None:
        super().__init__()

        self.N = N
        self.embed = PositionalEmbedding(vocab_size, hidden)
        self.pe = PositionalEncoder(hidden, dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden, attn_heads, dropout)
                for _ in range(self.N)
            ]
        )
        self.norm = LayerNorm(hidden)

    def forward(self, x, mask):
        x = self.pe(self.embed(x))
        for encoder in range(self.layers):
            x = encoder.forward(x, mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden, attn_heads, dropout, N=6):
        super().__init__()

        self.N = N
        self.embed = PositionalEmbedding(vocab_size, hidden)
        self.pe = PositionalEncoder(hidden, dropout)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(hidden, attn_heads, dropout)
                for _ in range(self.N)
            ]
        )
        self.norm = LayerNorm(hidden)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.pe(self.embed(trg))
        for decoder in range(self.layers):
            x = decoder.forward(x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, hidden, attn_heads, dropout=0.1, N=6) -> None:
        super().__init__()

        self.encoder = TransformerEncoder(src_vocab, hidden, attn_heads, dropout, N)
        self.decoder = TransformerDecoder(trg_vocab, hidden, attn_heads, dropout, N)
        self.out = nn.Linear(hidden, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(trg, encoder_output, src_mask, trg_mask)
        return self.out(decoder_output)

