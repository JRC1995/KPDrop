import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.attentions import MultiHeadAttention
import math

class TransformerSeq2SeqEncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, dim_ff: int = 2048,
                 dropout: float = 0.1):
        """
        Self-Attention的Layer，
        :param int d_model: input和output的输出维度
        :param int n_head: 多少个head，每个head的维度为d_model/n_head
        :param int dim_ff: FFN的维度大小
        :param float dropout: Self-attention和FFN的dropout大小，0表示不drop
        """
        super(TransformerSeq2SeqEncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.ffn_layer_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.dim_ff, self.d_model),
                                 nn.Dropout(dropout))

    def forward(self, x, mask):
        """
        :param x: batch x src_seq x d_model
        :param mask: batch x src_seq，为0的地方为padding
        :return:
        """
        # attention
        residual = x
        x = self.attn_layer_norm(x)
        x, _ = self.self_attn(query=x,
                              key=x,
                              value=x,
                              key_mask=mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        x = self.ffn_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TransformerSeq2SeqEncoder(nn.Module):
    def __init__(self, config, embed, pos_embed):
        """
        基于Transformer的Encoder
        :param embed: encoder输入token的embedding
        :param nn.Module pos_embed: position embedding
        :param int num_layers: 多少层的encoder
        :param int d_model: 输入输出的维度
        :param int n_head: 多少个head
        :param int dim_ff: FFN中间的维度大小
        :param float dropout: Attention和FFN的dropout大小
        """
        super(TransformerSeq2SeqEncoder, self).__init__()
        self.embed = embed

        self.pos_embed = pos_embed
        self.num_layers = config["num_layers"]
        self.d_model = config["hidden_size"]
        self.n_head = config["heads"]
        self.dim_ff = config["ff_dim"]
        self.dropout = config["dropout"]
        self.embed_scale = math.sqrt(self.d_model)

        self.input_fc = nn.Linear(self.embed.embedding_dim, self.d_model)
        self.layer_stacks = nn.ModuleList([TransformerSeq2SeqEncoderLayer(self.d_model,
                                                                          self.n_head,
                                                                          self.dim_ff, self.dropout)
                                           for _ in range(self.num_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, src, src_mask):
        """
        :param tokens: batch x max_len
        :param seq_len: [batch]
        :return: bsz x max_len x d_model, bsz x max_len(为0的地方为padding)
        """
        x = self.embed(src) * self.embed_scale  # batch, seq, dim
        batch_size, max_src_len, _ = x.size()
        device = x.device
        if self.pos_embed is not None:
            position = T.arange(1, max_src_len + 1).unsqueeze(0).long().to(device)
            x += self.pos_embed(position)

        x = self.input_fc(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layer_stacks:
            x = layer(x, src_mask)

        x = self.layer_norm(x)

        return x
