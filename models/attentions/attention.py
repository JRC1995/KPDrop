import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.layers.Linear import Linear

class attention(nn.Module):
    def __init__(self, config):
        super(attention, self).__init__()
        self.config = config
        self.encoder_hidden_size = config["encoder_hidden_size"]
        self.decoder_hidden_size = config["decoder_hidden_size"]
        if self.config["coverage_mechanism"]:
            self.attn_linear1 = Linear(2 * self.encoder_hidden_size + self.decoder_hidden_size + 1,
                                       self.decoder_hidden_size)
        else:
            self.attn_linear1 = Linear(2 * self.encoder_hidden_size + self.decoder_hidden_size,
                                       self.decoder_hidden_size)
        self.attn_linear2 = Linear(self.decoder_hidden_size, 1)
        if self.config["scratchpad"]:
            self.write_prob_linear = Linear(4 * self.encoder_hidden_size + self.decoder_hidden_size, 1)
            self.update_linear = Linear(2 * self.encoder_hidden_size + self.decoder_hidden_size, 2 * self.encoder_hidden_size)
        self.eps = 1e-9

    # %%
    def forward(self, key_encoder_states, value_encoder_states,
                decoder_state, attention_mask, input_mask,
                coverage_attn):

        N, S, _ = key_encoder_states.size()
        decoder_state = decoder_state.unsqueeze(1).repeat(1, S, 1)

        if self.config["coverage_mechanism"]:
            assert coverage_attn.size() == (N, S, 1)
            energy = self.attn_linear2(T.tanh(self.attn_linear1(T.cat([key_encoder_states,
                                                                       decoder_state,
                                                                       coverage_attn], dim=-1))))
        else:
            energy = self.attn_linear2(T.tanh(self.attn_linear1(T.cat([key_encoder_states, decoder_state], dim=-1))))

        attention_scores = F.softmax(energy + attention_mask, dim=1) * input_mask.unsqueeze(-1)

        assert (attention_scores <= 1.0).all()
        context_vector = T.sum(attention_scores * value_encoder_states, dim=1)
        # create attention weighted vector

        if self.config["coverage_mechanism"]:
            coverage_loss = T.sum(T.min(attention_scores, coverage_attn).squeeze(-1) * input_mask, dim=1)
            coverage_attn = coverage_attn + attention_scores
        else:
            coverage_loss = None

        if self.config["scratchpad"]:
            not_write_prob = T.sigmoid(self.write_prob_linear(T.cat([decoder_state,
                                                                     context_vector.unsqueeze(1).repeat(1, S, 1),
                                                                     key_encoder_states], dim=-1)))
            u = T.tanh(self.update_linear(T.cat([decoder_state,
                                                 context_vector.unsqueeze(1).repeat(1, S, 1)], dim=-1)))
            key_encoder_states = not_write_prob * key_encoder_states + (1 - not_write_prob) * u

        return {"context_vector": context_vector,
                "attention_scores": attention_scores,
                "key_encoder_states": key_encoder_states,
                "coverage_attn": coverage_attn,
                "coverage_loss": coverage_loss}
