import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers.Linear import Linear
from models.layers.TransformerSeq2SeqDecoder import TransformerSeq2SeqDecoder
from models.layers.TransformerSeq2SeqEncoder import TransformerSeq2SeqEncoder
from models.utils import get_sinusoid_encoding_table
from scipy.optimize import linear_sum_assignment


class TransformerSetDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerSetDecoder, self).__init__()
        self.vocab_len = config["vocab_len"]
        embed = nn.Embedding(self.vocab_len, config["embd_dim"], padding_idx=config["PAD_id"])
        self.init_emb(embed)
        pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(3000, config["embd_dim"]),
            freeze=True)
        self.encoder = TransformerSeq2SeqEncoder(config, embed, pos_embed)
        self.decoder = TransformerSeq2SeqDecoder(config, embed, pos_embed)
        self.sos_id = config["SOS_id"]
        self.max_decoder_len = config["max_decoder_len"]
        self.decoder_hidden_size = config["hidden_size"]
        self.config = config
        self.eps = 1e-9
        if self.config["contextualized_control_codes"]:
            self.aux_loss_criterion = nn.BCELoss(reduction='none')

    def init_emb(self, embed):
        """Initialize weights."""
        initrange = 0.1
        embed.weight.data.uniform_(-initrange, initrange)

    def generator(self, input_dict):

        ex_window_size = self.config["ex_window_size"]
        vocab2idx = self.config["vocab2idx"]
        separator_id = vocab2idx[";"]
        eos_id = vocab2idx["<eos>"]
        unk_id = vocab2idx["<unk>"]
        max_kp_num = self.config["max_kp_num"]

        i = input_dict["time_step"]
        N0 = input_dict["N"]
        input_idx = input_dict["input_idx"]
        output_dists = input_dict["output_dists"]
        ptr_src_idx = input_dict["ptr_src_idx"]
        past_probs = input_dict["past_probs"]
        cumprob = input_dict["cummulative_prob"]
        past_predictions = input_dict["past_predictions"]
        eos_mask = input_dict["eos_mask"]
        beam_filter_mask = input_dict["beam_filter_mask"]
        beam_lengths = input_dict["beam_lengths"]
        state = input_dict["state"]
        control_embed = input_dict["control_embed"]
        if self.config["pointer"]:
            vocab_len = self.vocab_len + self.max_oov_num
        else:
            vocab_len = self.vocab_len

        N = input_idx.size(0)

        if i == 0:
            first_word_mask = T.ones(N).float().to(input_idx.device)
        elif i > 0:
            first_word_mask = T.where(input_idx[:, -1] == separator_id,
                                      T.ones(N).float().to(input_idx.device),
                                      T.zeros(N).float().to(input_idx.device))

        B = N // N0
        N00 = N0 // max_kp_num

        input_idx_ = input_idx.view(N00, max_kp_num, B, i + 1)
        input_idx_ = input_idx_.permute(0, 2, 1, 3).contiguous()
        assert input_idx_.size() == (N00, B, max_kp_num, i + 1)
        input_idx_ = input_idx_.view(N00 * B, max_kp_num, i + 1)

        decoder_dist, _ = self.decoder(input_idx_, state, ptr_src_idx, self.max_oov_num, control_embed)

        assert decoder_dist.size() == (N00 * B, max_kp_num, vocab_len)
        decoder_dist = decoder_dist.view(N00, B, max_kp_num, vocab_len)
        decoder_dist = decoder_dist.permute(0, 2, 1, 3).contiguous()
        assert decoder_dist.size() == (N00, max_kp_num, B, vocab_len)
        output_dist = decoder_dist.view(N, vocab_len)

        output_dists.append(output_dist)

        if self.config["beam_search"] and self.config["generate"]:
            max_B = self.config["max_beam_size"]
            B = self.config["beam_width"]
        else:
            max_B = 1
            B = 1

        if (max_B == 1) or (B == 1):
            prediction = T.argmax(output_dist, dim=-1, keepdim=False)
            # print("step: {}, prediction: ".format(i), prediction[0])
            input_id = T.where(prediction >= self.vocab_len,
                               T.empty(N).fill_(unk_id).long().to(input_idx.device),
                               prediction)
            if i == 0:
                past_predictions = prediction.unsqueeze(-1)
            else:
                assert past_predictions.size() == (N, i)
                past_predictions = T.cat([past_predictions, prediction.unsqueeze(-1)], dim=-1)

            # print("step: {}, past_predictions: ".format(i), past_predictions[0])

        else:
            output_dist.size() == (N, vocab_len)
            top_B_val, top_B_idx = T.topk(output_dist, k=B, dim=-1)
            assert top_B_val.size() == (N, B)
            assert top_B_idx.size() == (N, B)

            BS = N // N0
            top_B_idx = top_B_idx.view(N0, BS, B)
            top_B_idx = top_B_idx.view(N0, BS * B)

            top_B_val = top_B_val.view(N0, BS, B)
            top_B_val = top_B_val.view(N0, BS * B)

            first_word_mask = first_word_mask.view(N0, BS, 1).repeat(1, 1, B).view(N0, BS * B)

            top_B_val_ = T.where(top_B_val < self.config["beam_threshold"],
                                 T.zeros_like(top_B_val).float().to(top_B_val.device),
                                 top_B_val)

            top_B_val = first_word_mask * top_B_val_ + (1 - first_word_mask) * top_B_val

            beam_filter_mask_ = T.where(top_B_val < self.config["beam_threshold"],
                                        T.zeros_like(top_B_val).float().to(top_B_val.device),
                                        T.ones_like(top_B_val).float().to(top_B_val.device))

            beam_filter_mask = beam_filter_mask.view(N0, BS).unsqueeze(-1).repeat(1, 1, B).view(N0, BS * B)
            beam_filter_mask = first_word_mask * (beam_filter_mask * beam_filter_mask_) \
                               + (1 - first_word_mask) * beam_filter_mask

            log_top_B_val = T.log(top_B_val + self.eps)
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
            cumprob, beam_idx = T.topk(cumprob, dim=-1, k=BS_)
            assert cumprob.size() == (N0, BS_)
            assert beam_idx.size() == (N0, BS_)
            cumprob = cumprob.view(N0 * BS_)

            beam_filter_mask = T.gather(beam_filter_mask, index=beam_idx, dim=-1)
            beam_filter_mask = beam_filter_mask.view(N0 * BS_)

            beam_lengths = T.gather(beam_lengths, index=beam_idx, dim=-1)
            beam_lengths = beam_lengths.view(N0 * BS_)

            prediction = T.gather(top_B_idx, index=beam_idx, dim=-1)
            assert prediction.size() == (N0, BS_)
            prediction = prediction.view(N0 * BS_)

            input_id = T.where(prediction >= self.vocab_len,
                               T.empty(N0 * BS_).fill_(unk_id).long().to(input_idx.device),
                               prediction)

            eos_mask = T.gather(eos_mask, index=beam_idx, dim=-1)
            assert eos_mask.size() == (N0, BS_)
            eos_mask = eos_mask.view(N0 * BS_)

            if self.config["one2one"]:
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

            assert input_idx.size() == (N, i + 1)
            input_idx = input_idx.view(N0, BS, i + 1)
            input_idx = input_idx.unsqueeze(2).repeat(1, 1, B, 1)
            assert input_idx.size() == (N0, BS, B, i + 1)
            input_idx = input_idx.view(N0, BS * B, i + 1)
            input_idx = T.gather(input_idx,
                                 index=beam_idx.view(N0, BS_, 1).repeat(1, 1, i + 1),
                                 dim=1)
            assert input_idx.size() == (N0, BS_, i + 1)
            input_idx = input_idx.view(N0 * BS_, i + 1)

        input_idx = T.cat([input_idx, input_id.unsqueeze(-1)], dim=1)

        output_dict = {"input_idx": input_idx,
                       "output_dists": output_dists,
                       "state": state,
                       "ptr_src_idx": ptr_src_idx,
                       "past_probs": past_probs,
                       "cummulative_prob": cumprob,
                       "past_predictions": past_predictions,
                       "eos_mask": eos_mask,
                       "beam_filter_mask": beam_filter_mask,
                       "beam_lengths": beam_lengths}

        return output_dict

    def hungarian_assign(self, decode_dist, target, ignore_indices, random=False):
        '''
        :param decode_dist: (batch_size, max_kp_num, kp_len, vocab_size)
        :param target: (batch_size, max_kp_num, kp_len)
        :return:
        '''

        batch_size, max_kp_num, kp_len = target.size()
        reorder_rows = T.arange(batch_size)[..., None]
        if random:
            reorder_cols = np.concatenate([np.random.permutation(max_kp_num).reshape(1, -1) for _ in range(batch_size)],
                                          axis=0)
        else:
            score_mask = target.new_zeros(target.size()).bool()
            for i in ignore_indices:
                score_mask |= (target == i)
            score_mask = score_mask.unsqueeze(1)  # (batch_size, 1, max_kp_num, kp_len)

            score = decode_dist.new_zeros(batch_size, max_kp_num, max_kp_num, kp_len)
            for b in range(batch_size):
                for l in range(kp_len):
                    score[b, :, :, l] = decode_dist[b, :, l][:, target[b, :, l]]
            score = score.masked_fill(score_mask, 0)
            score = score.sum(-1)  # batch_size, max_kp_num, max_kp_num

            reorder_cols = []
            for b in range(batch_size):
                row_ind, col_ind = linear_sum_assignment(score[b].detach().cpu().numpy(), maximize=True)
                reorder_cols.append(col_ind.reshape(1, -1))
                # total_score += sum(score[b][row_ind, col_ind])
            reorder_cols = np.concatenate(reorder_cols, axis=0)
        return tuple([reorder_rows, reorder_cols])

    def forward(self, src_idx, ptr_src_idx, input_mask, max_oov_num, first_mask=None, trg_idx=None, labels=None,
                output_mask=None):
        src = src_idx
        src_oov = ptr_src_idx
        src_mask = input_mask
        trg_mask = output_mask
        max_num_oov = max_oov_num
        max_kp_num = self.config["max_kp_num"]
        assign_steps = self.config["assign_steps"]
        null_id = self.config["vocab2idx"]["<null>"]
        pad_id = self.config["vocab2idx"]["<pad>"]
        # Encoding
        kp_len = trg_idx.size(2)

        batch_size = src.size(0)

        if not self.config["generate"]:
            if self.training:
                supposed_to_train = True
            else:
                supposed_to_train = False
            self.eval()
            with T.no_grad():
                y_t_init = T.ones(batch_size, max_kp_num, 1).to(src.device).long() * self.sos_id

                memory_bank = self.encoder(src, src_mask)
                state = self.decoder.init_state(memory_bank, src_mask)

                control_embed, scores = self.decoder.forward_seg(state, first_mask)
                input_tokens = src.new_zeros(batch_size, self.config["max_kp_num"], self.config["assign_steps"] + 1)
                decoder_dists = []
                input_tokens[:, :, 0] = self.sos_id
                self.config["generate"] = True
                for t in range(1, self.config["assign_steps"] + 1):
                    decoder_inputs = input_tokens[:, :, :t]
                    decoder_inputs = decoder_inputs.masked_fill(decoder_inputs.gt(self.vocab_len - 1),
                                                                self.config["UNK_id"])

                    decoder_dist, _ = self.decoder(decoder_inputs, state, src_oov, max_num_oov, control_embed)
                    input_tokens[:, :, t] = decoder_dist.argmax(-1)
                    decoder_dists.append(decoder_dist.reshape(batch_size, max_kp_num, 1, -1))
                self.config["generate"] = False

                decoder_dists = T.cat(decoder_dists, -2)

                mid_idx = max_kp_num // 2
                pre_reorder_index = self.hungarian_assign(decoder_dists[:, :mid_idx],
                                                          labels[:, :mid_idx, :assign_steps],
                                                          ignore_indices=[null_id, pad_id])
                labels[:, :mid_idx] = labels[:, :mid_idx][pre_reorder_index]
                trg_idx[:, :mid_idx] = trg_idx[:, :mid_idx][pre_reorder_index]
                trg_mask[:, :mid_idx] = trg_mask[:, :mid_idx][pre_reorder_index]

                ab_reorder_index = self.hungarian_assign(decoder_dists[:, mid_idx:],
                                                         labels[:, mid_idx:, :assign_steps],
                                                         ignore_indices=[null_id, pad_id])
                labels[:, mid_idx:] = labels[:, mid_idx:][ab_reorder_index]
                trg_idx[:, mid_idx:] = trg_idx[:, mid_idx:][ab_reorder_index]
                trg_mask[:, mid_idx:] = trg_mask[:, mid_idx:][ab_reorder_index]

            if supposed_to_train:
                self.train()

            input_tgt = T.cat([y_t_init, trg_idx[:, :, :-1]], dim=-1)
            memory_bank = self.encoder(src, src_mask)
            state = self.decoder.init_state(memory_bank, src_mask)
            control_embed, scores = self.decoder.forward_seg(state, first_mask)

            decoder_dist_all, _ = self.decoder(input_tgt, state, src_oov, max_num_oov, control_embed)

            decoder_dist_all = decoder_dist_all.view(batch_size, max_kp_num, kp_len, -1)

            if self.config["contextualized_control_codes"]:
                first_token_label = labels[..., 0]
                null_mask = T.where(first_token_label == self.config["NULL_id"],
                                    T.zeros_like(first_token_label).float().to(labels.device),
                                    T.ones_like(first_token_label).float().to(labels.device))

                # assert scores.size() == (N, S2)
                penalty_item = None #self.aux_loss_criterion(scores, null_mask)
            else:
                penalty_item = None

            return {"logits": decoder_dist_all,
                    "penalty_item": penalty_item,
                    "predictions": T.argmax(decoder_dist_all, dim=-1),
                    "probs": None,
                    "beam_filter_mask": None,
                    "labels": labels,
                    "output_mask": trg_mask}
        else:
            y_t_init = T.ones(batch_size * max_kp_num, 1).to(src.device).long() * self.sos_id

            memory_bank = self.encoder(src, src_mask)
            state = self.decoder.init_state(memory_bank, src_mask)
            #control_embed = self.decoder.forward_seg(state)
            control_embed, scores = self.decoder.forward_seg(state, first_mask)

            input_dict = {}
            input_dict["N"] = batch_size * max_kp_num
            input_dict["input_idx"] = y_t_init
            input_dict["output_dists"] = []

            input_dict["ptr_src_idx"] = ptr_src_idx

            # Special Inputs for Beam Search
            input_dict["past_probs"] = None  # keeps sequence of probabilities of keywords
            input_dict["cummulative_prob"] = T.zeros(batch_size * max_kp_num).float().to(src.device)
            # keeps cummulative probabilities of a whole beam
            input_dict["past_predictions"] = None  # keeps past predictions
            input_dict["eos_mask"] = T.zeros(batch_size * max_kp_num).float().to(
                src.device)  # 1 if eos is reach 0 otherwise
            input_dict["beam_filter_mask"] = T.ones(batch_size * max_kp_num).long().to(src.device)
            input_dict["beam_lengths"] = T.zeros(batch_size * max_kp_num).float().to(src.device)
            input_dict["state"] = state
            input_dict["control_embed"] = control_embed
            self.max_oov_num = max_oov_num
            # beam_filter_mask: 1 if we want to keep the beam 0 if we don't

            for t in range(self.max_decoder_len):
                input_dict["time_step"] = t
                output_dict = self.generator(input_dict)
                for key in output_dict:
                    input_dict[key] = output_dict[key]
                if T.sum(output_dict["eos_mask"]) == output_dict["eos_mask"].size(0):
                    break

            """
            PREPARE OUTPUT
            """
            predictions = output_dict["past_predictions"]
            NX = predictions.size(0)
            B = NX // (batch_size * max_kp_num)
            S = predictions.size(-1)

            assert predictions.size() == (batch_size * max_kp_num * B, S)
            predictions = predictions.view(batch_size, max_kp_num, B, S)
            predictions = predictions.permute(0, 2, 1, 3).contiguous()
            assert predictions.size() == (batch_size, B, max_kp_num, S)
            predictions = predictions.view(batch_size * B, max_kp_num, S)

            if output_dict["past_probs"] is not None:
                probs = output_dict["past_probs"].view(batch_size, max_kp_num, B, S)
                probs = probs.permute(0, 2, 1, 3).contiguous()
                assert probs.size() == (batch_size, B, max_kp_num, S)
                probs = probs.view(batch_size * B, max_kp_num, S)
            else:
                probs = None

            beam_filter_mask = output_dict["beam_filter_mask"].view(batch_size, max_kp_num, B)
            beam_filter_mask = beam_filter_mask.permute(0, 2, 1).contiguous()
            assert beam_filter_mask.size() == (batch_size, B, max_kp_num)
            beam_filter_mask = beam_filter_mask.view(batch_size * B, max_kp_num)
            return {"logits": None,
                    "penalty_item": None,
                    "predictions": predictions,
                    "probs": probs,
                    "beam_filter_mask": beam_filter_mask,
                    "labels": labels,
                    "output_mask": trg_mask}
