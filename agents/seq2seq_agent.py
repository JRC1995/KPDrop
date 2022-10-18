import torch as T
import torch.nn as nn
from torch.optim import *
from utils.evaluation_utils import evaluate
import copy
import math
import nltk
import torch.nn.functional as F
from nltk.stem import PorterStemmer
import numpy as np


class seq2seq_agent:
    def __init__(self, model, config, device):
        self.model = model
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = eval(config["optimizer"])
        self.config = config
        if self.config["custom_betas"]:
            self.optimizer = optimizer(self.parameters,
                                       lr=config["lr"],
                                       weight_decay=config["weight_decay"],
                                       betas=(0.9, 0.998), eps=1e-09)
        else:
            self.optimizer = optimizer(self.parameters,
                                       lr=config["lr"],
                                       weight_decay=config["weight_decay"])

        with open("predictions.txt", "w+") as fp:
            pass

        with open("srcs.txt", "w+") as fp:
            pass

        with open("trgs.txt", "w+") as fp:
            pass

        self.key = "none"
        # self.label_smoothing = self.config["label_smoothing"]
        self.device = device
        self.DataParallel = config["DataParallel"]
        self.optimizer.zero_grad()
        self.vocab2idx = config["vocab2idx"]
        self.idx2vocab = {id: token for token, id in self.vocab2idx.items()}
        self.vocab_len = len(config["vocab2idx"])
        self.eps = 1e-9
        self.epoch_level_scheduler = True
        self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                mode='max',
                                                                factor=0.5,
                                                                patience=config["scheduler_patience"])

    def loss_fn(self, logits, labels, output_mask, penalty_item=None):
        vocab_len = logits.size(-1)
        N, S = labels.size()
        assert logits.size() == (N, S, vocab_len)
        assert output_mask.size() == (N, S)
        assert (logits >= 0.0).all()
        #assert (logits <= 1.0).all()

        """
        confidence = T.empty(N, S)
        confidence = confidence.fill_(1.0 - self.label_smoothing)
        true_dist = T.empty(N, S, vocab_len).to(logits.device)
        true_dist.fill_(self.label_smoothing / (vocab_len - 1))

        true_dist = true_dist.scatter(-1, labels.view(N, S, 1), confidence.view(N, S, 1))
        """
        true_dist = F.one_hot(labels, num_classes=vocab_len)
        assert true_dist.size() == (N, S, vocab_len)
        assert (true_dist >= 0).all()

        logits = T.where(logits == 0.0,
                         T.empty_like(logits).fill_(self.eps).float().to(logits.device),
                         logits)

        neg_log_logits = -T.log(logits)
        #assert (neg_log_logits >= 0).all()

        assert true_dist.size() == neg_log_logits.size()

        cross_entropy = T.sum(neg_log_logits * true_dist, dim=-1)

        assert cross_entropy.size() == (N, S)
        masked_cross_entropy = cross_entropy * output_mask

        #mean_ce = T.sum(masked_cross_entropy, dim=1) / (T.sum(output_mask, dim=1) + self.eps)
        #loss = T.mean(mean_ce)
        loss = T.sum(masked_cross_entropy) / T.sum(output_mask)

        if penalty_item is not None:
            loss = loss + self.config["penalty_gamma"] * penalty_item.mean()

        return loss

    def decode(self, prediction_idx, src):
        src_token_dict = {}
        for token in src:
            if token not in src_token_dict and token not in self.vocab2idx:
                src_token_dict[token] = self.vocab_len + len(src_token_dict)

        src_token_dict_rev = {v: k for k, v in src_token_dict.items()}

        decoded_kps = []
        kp = ""
        kp_len = 0
        for id in prediction_idx:
            if id >= self.vocab_len:
                word = src_token_dict_rev.get(id, "<unk>")
            else:
                word = self.idx2vocab[id]
            if word == "<eos>" or word == "<sep>" or word == ";":
                if kp_len != 0:
                    decoded_kps.append(kp)
                    kp = ""
                    kp_len = 0
                elif word == "<eos>":
                    break
            else:
                if kp == "":
                    kp = kp + word
                else:
                    kp = kp + " " + word
                kp_len += 1
        prediction = " ; ".join(decoded_kps)
        return prediction

    def adv_decode(self, prediction_idx, src, probs):
        src_token_dict = {}
        for token in src:
            if token not in src_token_dict and token not in self.vocab2idx:
                src_token_dict[token] = self.vocab_len + len(src_token_dict)

        src_token_dict_rev = {v: k for k, v in src_token_dict.items()}

        decoded_kps = []
        decoded_kpps = []
        kp = ""
        kpp = 1
        kp_len = 0
        for id, prob in zip(prediction_idx, probs):
            if id >= self.vocab_len:
                word = src_token_dict_rev.get(id, "<unk>")
            else:
                word = self.idx2vocab[id]
            if word == "<eos>" or word == "<sep>" or word == ";":
                if kp_len != 0:
                    decoded_kps.append(kp)
                    decoded_kpps.append(kpp ** (1 / kp_len))
                    kp = ""
                    kpp = 1
                    kp_len = 0
                if self.config["one2one"] and self.config["generate"]:
                    break
                elif word == "<eos>":
                    break
            else:
                if kp == "":
                    kp = kp + word
                else:
                    kp = kp + " " + word
                kpp = kpp * prob
                kp_len += 1

        return decoded_kps, decoded_kpps

    def run(self, batch, train=True):

        if train:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        if not self.DataParallel:
            batch["src_vec"] = batch["src_vec"].to(self.device)
            batch["trg_vec"] = batch["trg_vec"].to(self.device)
            batch["ptr_src_vec"] = batch["ptr_src_vec"].to(self.device)
            batch["src_mask"] = batch["src_mask"].to(self.device)
            batch["trg_mask"] = batch["trg_mask"].to(self.device)
            batch["labels"] = batch["labels"].to(self.device)

        output_dict = self.model(batch)
        if self.config["generate"]:
            loss = None
        else:
            logits = output_dict["logits"]
            penalty_item = output_dict["penalty_item"]
            labels = batch["labels"].to(logits.device)
            loss = self.loss_fn(logits=logits,
                                labels=labels.to(logits.device),
                                output_mask=batch["trg_mask"].to(logits.device),
                                penalty_item=penalty_item)

        if not (self.config["beam_search"] and self.config["generate"] and not self.config["top_beam"]):
            predictions = output_dict["predictions"]
            if self.config["generate"] and self.config["top_beam"] and self.config["beam_search"]:
                N = batch["src_vec"].size(0)
                N2, S2 = predictions.size()
                predictions = predictions.view(N, -1, S2)
                predictions = predictions[:, 0, :]
            predictions = predictions.cpu().detach().numpy().tolist()
            predictions = [self.decode(prediction, src) for prediction, src in
                           zip(predictions, batch["src"])]


            if "generate_txt_files" in self.config and self.config["generate_txt_files"]:
                display_predictions = []
                for prediction in predictions:
                    prediction = prediction.split(" ; ")
                    prediction = [k for k in prediction if k != "<unk>"]
                    prediction = ";".join(prediction)
                    display_predictions.append(prediction)

                trgs = batch["trg"]
                display_trgs = []
                for trg in trgs:
                    trg = trg[0:-1] # remove eos
                    trg = " ".join(trg).replace(" ; ", ";")
                    display_trgs.append(trg)

                srcs = batch["src"]
                display_srcs = []
                for src in srcs:
                    src = " ".join(src)
                    display_srcs.append(src)

                with open("predictions.txt", "a", encoding="utf-8") as fp:
                    for prediction in display_predictions:
                        fp.write(prediction + "\n")

                with open("trgs.txt", "a", encoding="utf-8") as fp:
                    for trg in display_trgs:
                        fp.write(trg + "\n")

                with open("srcs.txt", "a", encoding="utf-8") as fp:
                    for src in display_srcs:
                        fp.write(src + "\n")
        else:
            predictions = output_dict["predictions"]
            probs = output_dict["probs"]
            beam_filter_masks = output_dict["beam_filter_mask"]
            N2, S2 = predictions.size()
            N = batch["src_vec"].size(0)
            B = N2 // N
            predictions = predictions.view(N, B, S2)
            probs = probs.view(N, B, S2)
            beam_filter_masks = beam_filter_masks.view(N, B)
            predictions = predictions.cpu().detach().numpy().tolist()
            probs = probs.cpu().detach().numpy().tolist()
            beam_filter_masks = beam_filter_masks.cpu().detach().numpy().tolist()

            predictions_ = []
            for j in range(N):
                src = batch["src"][j]
                beam_filter_mask = beam_filter_masks[j]
                prob_beams = probs[j]
                kp_predictions = []
                kpps = []
                for k, beam_prediction in enumerate(predictions[j]):
                    if beam_filter_mask[k] == 1:
                        decoded_kps, decoded_kpps = self.adv_decode(prediction_idx=beam_prediction,
                                                                    src=src,
                                                                    probs=prob_beams[k])
                        kp_predictions = kp_predictions + decoded_kps
                        kpps = kpps + decoded_kpps

                if self.config["rerank"]:
                    sorted_kpp_idx = np.flip(np.argsort(kpps), axis=0)
                    kp_predictions = [kp_predictions[l] for l in sorted_kpp_idx]
                predictions_.append(";".join(kp_predictions))

            predictions = predictions_

        if self.model.training:
            metrics = {"total_data": batch["batch_size"],
                       "total_present_precision": 0,
                       "total_present_recall": 0,
                       "total_absent_recall": 0,
                       "total_absent_precision": 0}
        else:
            metrics = evaluate(copy.deepcopy(batch["src"]), copy.deepcopy(batch["trg"]), copy.deepcopy(predictions),
                               beam=False, key=self.key)

        if loss is not None:
            metrics["loss"] = loss.item()
        else:
            metrics["loss"] = 0.0

        item = {"display_items": {"source": batch["src"],
                                  "target": batch["trg"],
                                  "predictions": predictions},
                "loss": loss,
                "metrics": metrics,
                "stats_metrics": metrics}

        return item

    def backward(self, loss):
        loss.backward()

    def step(self):
        if self.config["max_grad_norm"] is not None:
            T.nn.utils.clip_grad_norm_(self.parameters, self.config["max_grad_norm"])
        self.optimizer.step()
        self.optimizer.zero_grad()
