import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def glorot_uniform_init(weight, fan_in, fan_out):
    v = 6 if (fan_in != 0 and fan_out != 0) else 3
    bound = float(math.sqrt(v / (fan_in + fan_out)))
    nn.init.uniform_(weight, a=-bound, b=bound)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    sinusoid的embedding，其中position的表示中，偶数维(0,2,4,...)是sin, 奇数(1,3,5...)是cos
    :param int n_position: 一共多少个position
    :param int d_hid: 多少维度，需要为偶数
    :param padding_idx:
    :return: torch.FloatTensor, shape为n_position x d_hid
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def generate_absolute_positional_embeddings(max_len, d_model, freeze=True):
    with T.no_grad():
        # Compute the positional encodings once in log space.
        pe = T.zeros(max_len, d_model)
        position = T.arange(1, max_len+1).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        assert pe.size() == (max_len, d_model)
    return pe.unsqueeze(0), nn.Embedding.from_pretrained(pe,
                                                         freeze=freeze)


def generate_relative_positional_embeddings(max_len, d_model):
    with T.no_grad():
        # Compute the positional encodings once in log space.
        pe = T.zeros(2 * max_len + 1, d_model)
        position = T.arange(-max_len, max_len + 1).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        assert pe.size() == (2 * max_len + 1, d_model)
        pe = nn.Embedding.from_pretrained(pe,
                                          freeze=True)
    return pe


def generate_temporal_encodings(time, d_model):
    with T.no_grad():
        pe = T.zeros(d_model).float()
        div_term = T.exp(T.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
        pe[0::2] = T.sin(time * div_term)
        pe[1::2] = T.cos(time * div_term)

        pe = pe.view(1, 1, d_model)

    return pe


# https://gist.github.com/GongXinyuu/3536da55639bd9bfdd5a905ebf3ab88e
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """
    u = logits.data.new(*logits.size()).uniform_()
    gumbel_noise = -T.log(-T.log(u + eps) + eps)

    y_soft = F.softmax((logits + gumbel_noise) / tau, dim=dim)  # ~Gumbel(logits,tau)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = T.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def gelu(x):
    return 0.5 * x * (1 + T.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
