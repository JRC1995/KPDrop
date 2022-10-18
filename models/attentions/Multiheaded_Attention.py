import math
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.layers.Linear import Linear


class Multiheaded_Attention(nn.Module):
    def __init__(self, config, query_dim: int = 512, value_dim: int = 512):
        super(Multiheaded_Attention, self).__init__()

        self.qD = query_dim
        self.vD = value_dim

        self.config = config
        self.heads = config["heads"]
        self.d = config["head_dim"]
        self.dropout = config["dropout"]
        self.attn_dropout = config["attn_dropout"]
        self.eps = 1e-32
        self.position_max_len = config["position_max_len"]
        self.scaling = self.d ** -0.5

        # initialize params
        self.init_QKV()
        self.init_position()
        self.init_head_compose()
        self.reset_parameters()

    """
    Parameter Initializers
    """

    def init_QKV(self):
        self.query_linear = nn.Linear(self.qD, self.heads * self.d)
        self.key_linear = nn.Linear(self.vD, self.heads * self.d)
        self.value_linear = nn.Linear(self.vD, self.heads * self.d)

    # %%
    def init_position(self):
        if self.config["relative_pos_enc"]:
            self.content_bias = nn.Parameter(T.zeros(self.heads))
            self.position_bias = nn.Parameter(T.zeros(self.heads))
            self.position_linear = nn.Linear(self.qD, self.heads * self.d)

    # %%
    def init_head_compose(self):
        self.head_compose_linear = nn.Linear(self.heads * self.d, self.qD)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query_linear.weight)
        nn.init.xavier_uniform_(self.key_linear.weight)
        nn.init.xavier_uniform_(self.value_linear.weight)
        if self.config["relative_pos_enc"]:
            nn.init.xavier_uniform_(self.position_linear.weight)
        nn.init.xavier_uniform_(self.head_compose_linear.weight)

    # %%
    def score_positions(self, Q, vS, relative_position_embeddings, query_pos):
        N, H, qS, d = Q.size()
        S = max([qS, vS])
        if query_pos is not None:
            S = max(query_pos, S)
        position_idx = T.arange(S).unsqueeze(0).repeat(S, 1)
        position_idx_t = position_idx.permute(1, 0).contiguous()
        relative_mat_idx = position_idx - position_idx_t + self.position_max_len
        if query_pos is None:
            relative_mat_idx = relative_mat_idx[0:qS, 0:vS]
        else:
            relative_mat_idx = relative_mat_idx[query_pos-1, 0:vS].unsqueeze(0)

        RE = relative_position_embeddings(relative_mat_idx.to(Q.device))
        assert RE.size() == (qS, vS, self.qD)
        RE = self.position_linear(RE)
        assert RE.size() == (qS, vS, self.heads * self.d)

        RE = RE.view(qS, vS, self.heads, self.d)
        RE = RE.permute(2, 0, 1, 3).contiguous()
        assert RE.size() == (self.heads, qS, vS, self.d)

        REt = RE.permute(0, 1, 3, 2).contiguous()
        assert REt.size() == (self.heads, qS, self.d, vS)

        assert Q.size() == (N, H, qS, d)
        Q = Q.permute(1, 2, 0, 3).contiguous()
        assert Q.size() == (H, qS, N, d)


        v = self.position_bias.view(self.heads, 1, 1, 1)
        position_scores = T.matmul(Q + v, REt)

        assert position_scores.size() == (H, qS, N, vS)
        position_scores = position_scores.permute(2, 0, 1, 3).contiguous()
        assert position_scores.size() == (N, H, qS, vS)

        return position_scores / self.attn_scalar.to(position_scores.device)

    # %%
    def sum_normalize(self, logits, dim=-1):
        return logits / T.sum(logits + self.eps, keepdim=True, dim=dim)

    # %%

    def score_contents(self, Q, K):
        N, _, qS, _ = Q.size()
        vS = K.size(2)

        assert Q.size() == (N, self.heads, qS, self.d)
        assert K.size() == (N, self.heads, vS, self.d)

        if self.config["relative_pos_enc"]:
            u = self.content_bias.view(1, self.heads, 1, 1)
        else:
            u = 0

        Kt = K.permute(0, 1, 3, 2).contiguous()
        content_scores = T.matmul(Q + u, Kt)
        assert content_scores.size() == (N, self.heads, qS, vS)

        return content_scores / self.attn_scalar.to(content_scores.device)

    """
    Forward Function
    """

    # %%
    def forward(self, Q, K, V,
                relative_position_embeddings,
                attention_mask,
                query_pos):
        N, qS, _ = Q.size()
        _, vS, _ = V.size()

        assert K.size() == V.size()

        Q = self.query_linear(Q)
        Q *= self.scaling
        K = self.key_linear(K)
        V = self.value_linear(V)

        """
        assert Q.size() == (N, qS, self.heads * self.d)
        assert V.size() == (N, vS, self.heads * self.d)
        assert K.size() == V.size()
        Q = Q.view(N, qS, self.heads, self.d)
        K = K.view(N, vS, self.heads, self.d)
        V = V.view(N, vS, self.heads, self.d)
        Q = Q.permute(0, 2, 1, 3).contiguous()
        K = K.permute(0, 2, 1, 3).contiguous()
        V = V.permute(0, 2, 1, 3).contiguous()
        attention_mask = attention_mask.unsqueeze(1)
        assert attention_mask.size() == (N, 1, qS, vS)
        content_scores = self.score_contents(Q, K)
        if self.config["relative_pos_enc"]:
            position_scores = self.score_positions(Q, vS, relative_position_embeddings, query_pos)
            edge_scores = content_scores + position_scores
        else:
            edge_scores = content_scores
        attention_scores = F.softmax(edge_scores + attention_mask, dim=-1)
        """
        Q = Q.reshape(N, qS, self.heads, self.d)
        K = K.reshape(N, vS, self.heads, self.d)
        V = V.reshape(N, vS, self.heads, self.d)
        attn_weights = T.einsum('bqnh,bknh->bqkn', Q, K)
        attention_mask = attention_mask.unsqueeze(-1)
        _key_mask = ~attention_mask.bool()
        attn_weights = attn_weights.masked_fill(_key_mask, -float('inf'))
        attention_scores = F.softmax(attn_weights, dim=-2)

        attention_scores = F.dropout(attention_scores, p=self.attn_dropout, training=self.training)

        attended_values = T.einsum('bqkn,bknh->bqnh', attention_scores, V)  # batch,q_len,n_head,head_dim
        attended_values = attended_values.reshape(N, qS, -1)

        #attended_values = T.matmul(attention_scores, V)

        #assert attended_values.size() == (N, qS, self.heads * self.d)

        #attended_values = attended_values.permute(0, 2, 1, 3).contiguous()
        #attended_values = attended_values.view(N, qS, self.heads * self.d)

        attended_values = self.head_compose_linear(attended_values)
        #attended_values = attended_values.view(N, qS, self.qD)

        #assert (attention_scores >= 0.0).all()
        attention_scores = attention_scores.permute(0, 3, 1, 2).contiguous()

        return {"attended_values": attended_values, "attention_scores": attention_scores}