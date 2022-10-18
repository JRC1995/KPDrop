import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.attentions import attention
import numpy as np
from models.layers.Linear import Linear
import copy


class GRUEncoderDecoder(nn.Module):
    def __init__(self, config):
        super(GRUEncoderDecoder, self).__init__()

        self.pad_inf = -1.0e10
        self.vocab_len = config["vocab_len"]
        self.config = config
        self.dropout = config["dropout"]
        self.UNK_id = config["UNK_id"]
        self.sep_id = config["vocab2idx"][";"]

        self.embed_layer = nn.Embedding(self.vocab_len, config["embd_dim"],
                                        padding_idx=config["PAD_id"])

        self.encoder_hidden_size = config["encoder_hidden_size"]
        self.decoder_hidden_size = config["decoder_hidden_size"]

        self.encoder = nn.GRU(input_size=config["embd_dim"],
                              hidden_size=config["encoder_hidden_size"],
                              num_layers=config["encoder_layers"],
                              batch_first=True,
                              bidirectional=True)

        self.decodercell = nn.GRUCell(input_size=config["embd_dim"],
                                      hidden_size=config["decoder_hidden_size"],
                                      bias=True)

        self.attention = attention(config)
        self.out_linear1 = Linear(2 * self.encoder_hidden_size + self.decoder_hidden_size, self.decoder_hidden_size)
        self.out_linear2 = Linear(self.decoder_hidden_size, self.vocab_len)
        self.pointer_linear = Linear(2 * self.encoder_hidden_size + self.decoder_hidden_size + config["embd_dim"], 1)
        self.eps = 1e-9
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embed_layer.weight.data.uniform_(-initrange, initrange)

    def one_step_decode(self, input_dict):

        ex_window_size = self.config["ex_window_size"]
        vocab2idx = self.config["vocab2idx"]
        separator_id = vocab2idx[";"]
        eos_id = vocab2idx["<eos>"]

        i = input_dict["time_step"]
        N0 = input_dict["N"]
        input = input_dict["input"]
        input_id = input_dict["input_id"]
        h = input_dict["decoder_hidden_state"]
        output_dists = input_dict["output_dists"]
        key_encoded_src = input_dict["key_encoded_src"]
        value_encoded_src = input_dict["value_encoded_src"]
        ptr_src_idx = input_dict["ptr_src_idx"]
        input_mask = input_dict["input_mask"]
        attention_mask = input_dict["attention_mask"]
        trg = input_dict["trg"]
        output_mask = input_dict["output_mask"]
        teacher_force = input_dict["teacher_force"]
        coverage_attn = input_dict["coverage_attn"]
        coverage_loss = input_dict["coverage_loss"]
        past_probs = input_dict["past_probs"]
        cumprob = input_dict["cummulative_prob"]
        past_predictions = input_dict["past_predictions"]
        covered_idx = input_dict["covered_idx"]
        eos_mask = input_dict["eos_mask"]
        beam_filter_mask = input_dict["beam_filter_mask"]
        beam_lengths = input_dict["beam_lengths"]
        beam_width = input_dict["beam_width"]
        max_beam_size = input_dict["max_beam_size"]
        child_mask = input_dict["child_mask"]

        N, S1 = ptr_src_idx.size()

        if input is None:
            if (not self.config["generate"]) and teacher_force:
                input = trg[:, i - 1, :]
            else:
                input = self.embed_layer(input_id)

        if i == 0:
            first_word_mask = T.ones(N).float().to(input.device)
        elif i > 0:
            first_word_mask = T.where(input_id == separator_id,
                                      T.ones(N).float().to(input.device),
                                      T.zeros(N).float().to(input.device))

        h = self.decodercell(input, h)

        if not self.config["key_value_attention"]:
            value_encoded_src = key_encoded_src.clone()

        attention_dict = self.attention(key_encoder_states=key_encoded_src,
                                        value_encoder_states=value_encoded_src,
                                        decoder_state=h,
                                        attention_mask=attention_mask,
                                        input_mask=input_mask,
                                        coverage_attn=coverage_attn)

        context_vector = attention_dict["context_vector"]
        pointer_attention_scores = attention_dict["attention_scores"].squeeze(-1)
        key_encoded_src = attention_dict["key_encoder_states"]
        coverage_attn = attention_dict["coverage_attn"]
        coverage_loss_t = attention_dict["coverage_loss"]

        if self.training:
            if self.config["coverage_mechanism"]:
                coverage_loss = coverage_loss + (coverage_loss_t * output_mask[:, i])

        concat_out = T.cat([h, context_vector], dim=-1)
        gen_dist_intermediate = F.dropout(self.out_linear1(concat_out), p=self.dropout, training=self.training)
        gen_dist = F.softmax(self.out_linear2(gen_dist_intermediate), dim=-1)

        if self.max_oov_num > 0:
            potential_extra_vocab = T.zeros(N, self.max_oov_num).float().to(input.device)
            gen_dist_extended = T.cat([gen_dist, potential_extra_vocab], dim=-1)
        else:
            gen_dist_extended = gen_dist

        p_gen = T.sigmoid(self.pointer_linear(T.cat([input, context_vector, h], dim=-1)))

        assert gen_dist_extended.size() == (N, self.vocab_len + self.max_oov_num)
        assert p_gen.size() == (N, 1)
        assert ptr_src_idx.size() == (N, S1)
        assert pointer_attention_scores.size() == (N, S1)

        output_dist = (p_gen * gen_dist_extended).scatter_add(dim=-1,
                                                              index=ptr_src_idx,
                                                              src=((1.0 - p_gen) * pointer_attention_scores))

        if i > 0:
            if self.config["generate"] and self.config["hard_exclusion"]:
                vocab_len = self.vocab_len + self.max_oov_num
                dist_masks = T.ones(N, vocab_len).float().to(input.device)
                for j in range(N):
                    if first_word_mask[j] == 1:
                        dist_masks[j, separator_id] = 0
                        if covered_idx[j] is not None:
                            for id in covered_idx[j]:
                                dist_masks[j, id] = 0
                output_dist = dist_masks * output_dist
                output_dist = output_dist / (T.sum(output_dist, dim=-1, keepdim=True) + self.eps)

        output_dists.append(output_dist)

        if self.config["beam_search"] and self.config["generate"]:
            max_B = max_beam_size
            B = beam_width  # self.config["beam_width"]
        else:
            max_B = 1
            B = 1
        #print("B: ", B)

        if (max_B == 1) or (B == 1):
            prediction = T.argmax(output_dist, dim=-1, keepdim=False)
            input_id = T.where(prediction >= self.vocab_len,
                               T.empty(N).fill_(self.UNK_id).long().to(input.device),
                               prediction)
            if i == 0:
                past_predictions = prediction.unsqueeze(-1)
            else:
                assert past_predictions.size() == (N, i)
                past_predictions = T.cat([past_predictions, prediction.unsqueeze(-1)], dim=-1)

            if self.config["generate"] and self.config["hard_exclusion"]:
                for j in range(N):
                    if first_word_mask[j] == 1:
                        if covered_idx[j] is None:
                            covered_idx[j] = [prediction[j]]
                        else:
                            covered_idx[j].append(prediction[j])
                            covered_idx_sample = covered_idx[j]
                            covered_idx[j] = covered_idx_sample[-ex_window_size:]
        else:
            vocab_len = self.vocab_len + self.max_oov_num
            output_dist.size() == (N, vocab_len)
            top_B_val, top_B_idx = T.topk(output_dist, k=B, dim=-1, sorted=True)
            assert top_B_val.size() == (N, B)
            assert top_B_idx.size() == (N, B)

            BS = N // N0
            top_B_idx = top_B_idx.view(N0, BS, B)
            top_B_idx = top_B_idx.view(N0, BS * B)

            top_B_val = top_B_val.view(N0, BS, B)
            top_B_val = top_B_val.view(N0, BS * B)

            first_word_mask = first_word_mask.view(N0, BS, 1).repeat(1, 1, B).view(N0, BS * B)


            if i == 0:
                threshold = self.config["beam_threshold"]
            else:
                threshold = 0.0

            beam_filter_mask_ = T.where(top_B_val <= threshold,
                                        T.zeros_like(top_B_val).float().to(top_B_val.device),
                                        T.ones_like(top_B_val).float().to(top_B_val.device))

            beam_filter_mask = beam_filter_mask.view(N0, BS).unsqueeze(-1).repeat(1, 1, B).view(N0, BS * B)
            beam_filter_mask = beam_filter_mask * beam_filter_mask_



            if i > 0 and self.config["beam_threshold"] > 0:
                #print("child mask i+1: ", child_mask.size())
                assert child_mask.size() == (N0, B)
                child_mask_ = child_mask.unsqueeze(1).repeat(1, BS, 1)
                assert child_mask_.size() == (N0, BS, B)
                child_mask_ = child_mask_.view(N0, BS * B)
                beam_filter_mask = beam_filter_mask * child_mask_

            top_B_val = beam_filter_mask * top_B_val

            log_top_B_val = T.log(top_B_val + self.eps)
            if self.config["beam_threshold"] > 0:
                assert beam_filter_mask.size() == (N0, BS * B)
                assert log_top_B_val.size() == (N0, BS * B)
                non_select_val = T.ones(N0, BS * B).float().to(log_top_B_val.device) * (-99999)
                log_top_B_val = beam_filter_mask * log_top_B_val + (1-beam_filter_mask) * non_select_val


            log_top_B_val = log_top_B_val.view(N0, BS, B)

            eos_mask = eos_mask.view(N0, BS, 1)
            for_eosed_val = T.tensor([0] + [-99999] * (B - 1)).float().to(top_B_val.device)
            for_eosed_val = for_eosed_val.view(1, 1, B)
            log_top_B_val = eos_mask * for_eosed_val + (1 - eos_mask) * log_top_B_val
            # we want to ignore the children on beams which have reached eos.
            # We want to keep only one children to preserve the overall beam

            log_top_B_val = log_top_B_val.view(N0, BS * B)
            eos_mask = eos_mask.view(N0 * BS)

            eos_mask = eos_mask.view(N0, BS, 1).repeat(1, 1, B)
            eos_mask = eos_mask.view(N0, BS * B)

            beam_lengths = beam_lengths.view(N0, BS, 1).repeat(1, 1, B).view(N0, BS * B)
            beam_lengths = eos_mask * beam_lengths + (1 - eos_mask) * (beam_lengths + 1)
            assert beam_lengths.size() == (N0, BS * B)

            if self.config["length_normalization"]:
                alpha = self.config["length_coefficient"]
                lp = T.pow(5 + beam_lengths, alpha) / (6 ** alpha)
                log_top_B_val = log_top_B_val / lp


            cumprob = cumprob.view(N0, BS, 1).repeat(1, 1, B).view(N0, BS * B)
            cumprob = cumprob + log_top_B_val

            BS_ = min(BS * B, max_B)
            cumprob, beam_idx = T.topk(cumprob, dim=-1, k=BS_, sorted=True)
            assert cumprob.size() == (N0, BS_)
            assert beam_idx.size() == (N0, BS_)


            beam_filter_mask = T.gather(beam_filter_mask, index=beam_idx, dim=-1)

            if i == 0 and self.config["beam_threshold"] > 0:
                assert beam_filter_mask.size() == (N0, BS_)
                DynamicTopKs = T.sum(beam_filter_mask.int(), dim=1)
                max_k = T.max(DynamicTopKs).item()
                max_beam_size = max_k
                beam_width = max_k
                child_mask = []
                assert DynamicTopKs.size(0) == N0 and len(DynamicTopKs.size()) == 1
                for j in range(N0):
                    num = DynamicTopKs[j].item()
                    mask = [1 for _ in range(num)]
                    if num < max_k:
                        mask2 = [0 for _ in range(max_k - num)]
                        mask = mask + mask2
                        assert len(mask) == max_k
                    child_mask.append(mask)
                child_mask = T.tensor(child_mask).float().to(beam_filter_mask.device)
                assert child_mask.size() == (N0, max_k)
                BS_ = max_k
                beam_idx = beam_idx[:, 0:max_k]
                beam_filter_mask = beam_filter_mask[:, 0:max_k]
                cumprob = cumprob[:, 0:max_k]
                #print("max_k: ", max_k)
                #print("child mask i: ", child_mask.size())


            cumprob = cumprob.reshape(N0 * BS_)
            if self.config["beam_threshold"] > 0:
                beam_filter_mask = beam_filter_mask * child_mask
            beam_filter_mask = beam_filter_mask.view(N0 * BS_)

            beam_lengths = T.gather(beam_lengths, index=beam_idx, dim=-1)
            beam_lengths = beam_lengths.view(N0 * BS_)

            prediction = T.gather(top_B_idx, index=beam_idx, dim=-1)
            assert prediction.size() == (N0, BS_)
            prediction = prediction.view(N0 * BS_)

            input_id = T.where(prediction >= self.vocab_len,
                               T.empty(N0 * BS_).fill_(self.UNK_id).long().to(input.device),
                               prediction)

            eos_mask = T.gather(eos_mask, index=beam_idx, dim=-1)
            assert eos_mask.size() == (N0, BS_)
            eos_mask = eos_mask.view(N0 * BS_)

            if self.config["one2one"] and self.config["generate"]:
                new_eos_mask = T.where((prediction == separator_id) | (prediction == eos_id),
                                       T.ones(N0 * BS_).float().to(prediction.device),
                                       T.zeros(N0 * BS_).float().to(prediction.device))
            else:
                new_eos_mask = T.where(prediction == eos_id,
                                       T.ones(N0 * BS_).float().to(prediction.device),
                                       T.zeros(N0 * BS_).float().to(prediction.device))

            eos_mask = eos_mask + (1 - eos_mask) * new_eos_mask

            probs = T.gather(top_B_val, index=beam_idx, dim=-1)
            probs = probs.view(N0 * BS_)

            if i == 0:
                past_predictions = prediction.unsqueeze(-1)
                past_probs = probs.unsqueeze(-1)
            else:
                assert past_predictions.size() == (N, i)
                past_predictions = past_predictions.view(N0, BS, i)
                past_predictions = past_predictions.unsqueeze(2).repeat(1, 1, B, 1)
                assert past_predictions.size() == (N0, BS, B, i)
                past_predictions = past_predictions.view(N0, BS * B, i)
                past_predictions = T.gather(past_predictions, index=beam_idx.unsqueeze(-1).repeat(1, 1, i), dim=1)
                assert past_predictions.size() == (N0, BS_, i)
                past_predictions = past_predictions.view(N0 * BS_, i)
                past_predictions = T.cat([past_predictions, prediction.unsqueeze(-1)], dim=-1)

                assert past_probs.size() == (N, i)
                past_probs = past_probs.view(N0, BS, i)
                past_probs = past_probs.unsqueeze(2).repeat(1, 1, B, 1)
                assert past_probs.size() == (N0, BS, B, i)
                past_probs = past_probs.view(N0, BS * B, i)
                past_probs = T.gather(past_probs, index=beam_idx.unsqueeze(-1).repeat(1, 1, i), dim=1)
                assert past_probs.size() == (N0, BS_, i)
                past_probs = past_probs.view(N0 * BS_, i)
                past_probs = T.cat([past_probs, probs.unsqueeze(-1)], dim=-1)

            D = h.size(-1)
            h = h.view(N0, BS, D).unsqueeze(-2).repeat(1, 1, B, 1)
            assert h.size() == (N0, BS, B, D)
            h = h.view(N0, BS * B, D)

            h = T.gather(h, index=beam_idx.unsqueeze(-1).repeat(1, 1, D), dim=1)
            assert h.size() == (N0, BS_, D)
            h = h.view(N0 * BS_, D)

            if self.config["generate"] and self.config["hard_exclusion"]:
                first_word_mask = T.gather(first_word_mask, index=beam_idx, dim=-1)
                assert first_word_mask.size() == (N0, BS_)
                first_word_mask = first_word_mask.view(N0 * BS_)

                new_covered_idx = []
                for j in range(N0):
                    for k in range(BS_):
                        id = beam_idx[j, k].item() // B
                        new_covered_idx.append(covered_idx[j * BS + id])

                covered_idx = new_covered_idx

                for j in range(N0 * BS_):
                    if first_word_mask[j] == 1:
                        if covered_idx[j] is None:
                            covered_idx[j] = [prediction[j]]
                        else:
                            covered_idx[j].append(prediction[j])
                            covered_idx_sample = covered_idx[j]
                            covered_idx[j] = covered_idx_sample[-ex_window_size:]

            if self.config["coverage_mechanism"]:
                coverage_attn = coverage_attn.view(N0, BS, S1).unsqueeze(2).repeat(1, 1, B, 1).view(N0, BS * B, S1)
                coverage_attn = T.gather(coverage_attn, index=beam_idx.view(N0, BS_, 1).repeat(1, 1, S1), dim=1)
                coverage_attn = coverage_attn.view(N0 * BS_, S1, 1)

            if self.config["scratchpad"]:
                D_ = key_encoded_src.size(-1)
                key_encoded_src = key_encoded_src.view(N0, BS, S1, D_).unsqueeze(2).repeat(1, 1, B, 1, 1)
                key_encoded_src = key_encoded_src.view(N0, BS * B, S1, D_)
                key_encoded_src = T.gather(key_encoded_src,
                                           index=beam_idx.view(N0, BS_, 1, 1).repeat(1, 1, S1, D_), dim=1)
            elif BS != BS_:
                D_ = key_encoded_src.size(-1)
                key_encoded_src = key_encoded_src.view(N0, BS, S1, D_)[:, 0, :, :].unsqueeze(1)
                assert key_encoded_src.size() == (N0, 1, S1, D_)
                key_encoded_src = key_encoded_src.repeat(1, BS_, 1, 1)
                key_encoded_src = key_encoded_src.view(N0 * BS_, S1, D_)

            if BS != BS_:
                D_ = value_encoded_src.size(-1)
                value_encoded_src = value_encoded_src.view(N0, BS, S1, D_)[:, 0, :, :].unsqueeze(1)
                assert value_encoded_src.size() == (N0, 1, S1, D_)
                value_encoded_src = value_encoded_src.repeat(1, BS_, 1, 1)
                value_encoded_src = value_encoded_src.view(N0 * BS_, S1, D_)

                ptr_src_idx = ptr_src_idx.view(N0, BS, S1)[:, 0, :].unsqueeze(1).repeat(1, BS_, 1).view(N0 * BS_, S1)
                input_mask = input_mask.view(N0, BS, S1)[:, 0, :].unsqueeze(1).repeat(1, BS_, 1).view(N0 * BS_, S1)
                attention_mask = attention_mask.view(N0, BS, S1)[:, 0, :].unsqueeze(1).repeat(1, BS_, 1)
                attention_mask = attention_mask.view(N0 * BS_, S1, 1)

        output_dict = {"input_id": input_id,
                       "beam_width": beam_width,
                       "max_beam_size": max_beam_size,
                       "input": None,
                       "decoder_hidden_state": h,
                       "output_dists": output_dists,
                       "key_encoded_src": key_encoded_src,
                       "value_encoded_src": value_encoded_src,
                       "ptr_src_idx": ptr_src_idx,
                       "input_mask": input_mask,
                       "attention_mask": attention_mask,
                       "coverage_attn": coverage_attn,
                       "coverage_loss": coverage_loss,
                       "past_probs": past_probs,
                       "cummulative_prob": cumprob,
                       "past_predictions": past_predictions,
                       "covered_idx": covered_idx,
                       "eos_mask": eos_mask,
                       "beam_filter_mask": beam_filter_mask,
                       "beam_lengths": beam_lengths,
                       "child_mask": child_mask}

        return output_dict

    # %%
    def forward(self, src_idx, max_oov_num, ptr_src_idx, input_mask, trg_idx=None, output_mask=None):
        self.max_oov_num = max_oov_num
        src = self.embed_layer(src_idx)

        tfr = self.config["teacher_force_ratio"]
        teacher_force = np.random.choice([True, False], p=[tfr, 1 - tfr])
        if self.config["generate"]:
            teacher_force = False
        if not self.config["generate"] and not self.training:  # this condition is only true during validation
            teacher_force = True  # we need teacher forcing true for ground truth ppl calculation during validation

        if trg_idx is not None:
            assert output_mask is not None
            trg = self.embed_layer(trg_idx)
        else:
            trg = None

        N, S1, D = src.size()
        if trg is not None:
            N, S2, D = trg.size()

        assert input_mask.size() == (N, S1)
        attention_mask = T.where(input_mask == 0.0,
                                 T.empty_like(input_mask).fill_(self.pad_inf).float().to(input_mask.device),
                                 T.zeros_like(input_mask).float().to(input_mask.device))

        attention_mask = attention_mask.unsqueeze(-1)
        assert attention_mask.size() == (N, S1, 1)

        """
        ENCODING
        """
        lengths = T.sum(input_mask, dim=1).long().view(N).cpu()
        packed_sequence = nn.utils.rnn.pack_padded_sequence(src, lengths, batch_first=True, enforce_sorted=False)
        encoded_src, hn = self.encoder(packed_sequence)
        encoded_src, _ = nn.utils.rnn.pad_packed_sequence(encoded_src, batch_first=True)
        assert encoded_src.size() == (N, S1, 2 * self.encoder_hidden_size)
        assert hn.size() == (2, N, self.encoder_hidden_size)  # forward and backward hidden states

        hn = hn.permute(1, 0, 2).contiguous()
        assert hn.size() == (N, 2, self.encoder_hidden_size)
        hn = hn.view(N, 2 * self.encoder_hidden_size)

        """
        PREPARING FOR DECODING
        """
        sos_input_id = T.ones(N).long().to(src.device) * self.config["vocab2idx"]["<sos>"]
        sos_input = self.embed_layer(sos_input_id)

        h = hn  # initial decoder hidden state
        initial_hidden_state = hn.clone()

        if not self.config["generate"]:
            S = S2
        else:
            S = self.config["max_decoder_len"]

        if self.config["coverage_mechanism"]:
            coverage_attn = T.zeros(N, S1, 1).float().to(src.device)
            coverage_loss = T.zeros(N).float().to(src.device)
        else:
            coverage_attn = None
            coverage_loss = None

        key_encoded_src = encoded_src.clone()
        value_encoded_src = encoded_src.clone()

        input_dict = {}
        input_dict["N"] = N
        input_dict["input"] = sos_input.clone()
        input_dict["input_id"] = None
        input_dict["decoder_hidden_state"] = h
        input_dict["output_dists"] = []
        input_dict["max_beam_size"] = self.config["max_beam_size"]
        input_dict["beam_width"] = self.config["beam_width"]

        input_dict["key_encoded_src"] = key_encoded_src
        input_dict["value_encoded_src"] = value_encoded_src
        input_dict["ptr_src_idx"] = ptr_src_idx
        input_dict["input_mask"] = input_mask
        input_dict["attention_mask"] = attention_mask

        input_dict["trg"] = trg
        input_dict["output_mask"] = output_mask
        input_dict["teacher_force"] = teacher_force
        input_dict["coverage_attn"] = coverage_attn
        input_dict["coverage_loss"] = coverage_loss

        # Special Inputs for Beam Search
        input_dict["past_probs"] = None  # keeps sequence of probabilities of keywords
        input_dict["cummulative_prob"] = T.zeros(N).float().to(src.device)
        # keeps cummulative probabilities of a whole beam
        input_dict["past_predictions"] = None  # keeps past predictions
        input_dict["covered_idx"] = [None] * (N)  # keeps keyphrase first words covered within a window size
        input_dict["eos_mask"] = T.zeros(N).float().to(src.device)  # 1 if eos is reach 0 otherwise
        input_dict["beam_filter_mask"] = T.ones(N).long().to(src.device)
        input_dict["beam_lengths"] = T.zeros(N).float().to(src.device)
        input_dict["child_mask"] = None
        # beam_filter_mask: 1 if we want to keep the beam 0 if we don't

        """
        DECODE AUTOREGRESSIVELY
        """
        for i in range(S):
            input_dict["time_step"] = i
            if self.config["one2one"] and i > 0 and not self.config["generate"]:
                input_id = input_dict["input_id"]
                if teacher_force:
                    input = trg[:, i - 1, :]
                    trg_id = trg_idx[:, i - 1]
                    sep_mask = T.where(trg_id == self.sep_id,
                                       T.ones_like(trg_id).float().to(input_id.device),
                                       T.zeros_like(trg_id).float().to(input_id.device))
                else:
                    input = self.embed_layer(input_id)
                    sep_mask = T.where(input_id == self.sep_id,
                                       T.ones_like(input_id).float().to(input_id.device),
                                       T.zeros_like(input_id).float().to(input_id.device))
                sep_mask = sep_mask.unsqueeze(-1)
                X, D = input.size()
                assert sep_mask.size() == (X, 1)
                assert initial_hidden_state.size() == (X, self.decoder_hidden_size)
                h = input_dict["decoder_hidden_state"]
                assert h.size() == (X, self.decoder_hidden_size)

                if sos_input.size(0) != X:
                    sos_input = sos_input[0, :].unsqueeze(0).repeat(X, 1)
                assert sos_input.size() == (X, D)

                input = sep_mask * sos_input + (1 - sep_mask) * input
                h = sep_mask * initial_hidden_state + (1 - sep_mask) * h

                input_dict["input"] = input
                input_dict["decoder_hidden_state"] = h

            output_dict = self.one_step_decode(input_dict)
            for key in output_dict:
                input_dict[key] = output_dict[key]
            if T.sum(output_dict["eos_mask"]) == output_dict["eos_mask"].size(0):
                break

        """
        PREPARE OUTPUT
        """
        if self.config["generate"]:
            logits = None
        else:
            output_dists = output_dict["output_dists"]
            logits = T.stack(output_dists, dim=1)
            assert logits.size() == (N, S, self.vocab_len + self.max_oov_num)

        if self.config["coverage_mechanism"] and self.training:
            coverage_loss = coverage_loss / (T.sum(output_mask, dim=1) + self.eps)
        else:
            coverage_loss = None

        return {"logits": logits,
                "penalty_item": coverage_loss,
                "predictions": output_dict["past_predictions"],
                "probs": output_dict["past_probs"],
                "beam_filter_mask": output_dict["beam_filter_mask"]}
