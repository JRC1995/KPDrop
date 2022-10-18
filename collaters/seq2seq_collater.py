import torch as T
import numpy as np
import random
import copy
from nltk.stem import PorterStemmer


class seq2seq_collater:
    def __init__(self, PAD, config, train):
        self.PAD = PAD
        self.config = config
        self.train = train
        self.UNK_id = config["UNK_id"]
        self.vocab_len = config["vocab_len"]
        self.separator_id = config["vocab2idx"][";"]
        self.eos_id = config["vocab2idx"]["<eos>"]
        self.vocab2idx = config["vocab2idx"]
        self.stemmer = PorterStemmer()

    def pad(self, items, PAD):
        # max_len = max([len(item) for item in items])
        max_len = max([len(item) for item in items])

        padded_items = []
        item_masks = []
        for item in items:
            mask = [1] * len(item)
            while len(item) < max_len:
                item.append(PAD)
                mask.append(0)
            padded_items.append(item)
            item_masks.append(mask)

        return padded_items, item_masks

    def text_vectorize(self, text):
        return [self.vocab2idx.get(word, self.vocab2idx['<unk>']) for word in text]

    def keyphrase_dropout_fn(self, src, trg):
        p = self.config["kpdrop"]
        # print("predrop trg: ", trg)
        trg = " ".join(copy.deepcopy(trg[0:-1]))
        kps = trg.split(" ; ")
        tokenized_kps = [kp.split(" ") for kp in kps]
        stemmed_kps = []
        for tokenized_kp in tokenized_kps:
            stemmed_kp = " ".join([self.stemmer.stem(kw.lower().strip()) for kw in copy.deepcopy(tokenized_kp)])
            stemmed_kps.append(stemmed_kp)

        stemmed_src = " ".join([self.stemmer.stem(token) for token in copy.deepcopy(src)])

        present_keyphrases = []
        original_present_keyphrases = []
        for kp, stemmed_kp in zip(kps, stemmed_kps):
            if stemmed_kp in stemmed_src:
                present_keyphrases.append(stemmed_kp)
                original_present_keyphrases.append(kp)

        # print("\n\noriginal present keyphrases: ", present_keyphrases)

        copy_present = copy.deepcopy(original_present_keyphrases)
        present_kp = []
        orig_present_kp = []
        for X, Y in sorted(zip(present_keyphrases, original_present_keyphrases), key=lambda x: len(x[0]), reverse=True):
            present_kp.append(X)
            orig_present_kp.append(Y)
        present_keyphrases = present_kp
        original_present_keyphrases = orig_present_kp

        # print("post sort present keyphrases: ", present_keyphrases)

        kps_drop = np.random.binomial(1, p, len(present_keyphrases))

        # print("kps_drop: ", kps_drop)

        i = 0
        for kp, kp_drop in zip(present_keyphrases, kps_drop):
            if kp_drop == 1:
                marker = ' '.join(['#' for _ in kp.split(" ")])
                stemmed_src = stemmed_src.replace(kp, marker)
            i += 1

        tokenized_stemmed_src = stemmed_src.split(" ")

        recover_src = []
        for token_stem, token in zip(tokenized_stemmed_src, src):
            if token_stem != '#':
                recover_src.append(token)
            elif recover_src:
                if recover_src[-1] != "<pad>":
                    recover_src.append("<pad>")
            else:
                recover_src.append("<pad>")

        new_src = recover_src

        original_absent_keyphrases = [kp for kp in kps if kp not in original_present_keyphrases]
        new_absent_keyphrases = [kp for kp, stemmed_kp in zip(original_present_keyphrases, present_keyphrases) if
                                 stemmed_kp not in stemmed_src]
        new_present_keyphrases = [kp for kp in copy_present if kp not in new_absent_keyphrases]

        # print("\n\nnew present keyphrases: ", new_present_keyphrases)
        # print("new absent keyphrases: ", new_absent_keyphrases)
        # print("original absent keyphrases: ", original_absent_keyphrases)
        # print("-------------------------------------------------------")

        keyphrases = new_present_keyphrases + new_absent_keyphrases + original_absent_keyphrases

        new_trg = " ; ".join(keyphrases).split(" ")
        new_trg = new_trg + ["<eos>"]

        # print("post drop trg: ", new_trg)

        src_vec = self.text_vectorize(new_src)
        trg_vec = self.text_vectorize(new_trg)

        if self.config["one2one"]:
            trg_vec = trg_vec[:-1] + [self.vocab2idx[";"]]

        src_token_dict = {}
        for token in new_src:
            if token not in src_token_dict and token not in self.vocab2idx:
                src_token_dict[token] = self.vocab_len + len(src_token_dict)

        ptr_src_vec = [self.vocab2idx.get(token, src_token_dict.get(token, self.UNK_id)) for token in new_src]
        label = [self.vocab2idx.get(token, src_token_dict.get(token, self.UNK_id)) for token in new_trg]

        if self.config["one2one"]:
            label = label[:-1] + [self.vocab2idx[";"]]

        return new_src, new_trg, src_vec, trg_vec, ptr_src_vec, label, len(src_token_dict)

    def sort_list(self, objs, idx):
        return [objs[i] for i in idx]

    def src_truncate(self, src_vec):
        return src_vec[0:self.config["truncate_src_len"]]

    def trg_truncate(self, trg_vec):
        return trg_vec[0:self.config["truncate_trg_len"]]

    def collate_fn(self, batch):
        batch = copy.deepcopy(batch)
        srcs = [obj['src'] for obj in batch]
        trgs = [obj['trg_no_peos'] for obj in batch]
        oov_nums = [obj["oov_num"] for obj in batch]
        srcs_vec = [obj['src_vec'] for obj in batch]
        ptr_srcs_vec = [obj["ptr_src_vec"] for obj in batch]
        labels = [obj['label'] for obj in batch]
        trgs_vec = [obj['trg_no_peos_vec'] for obj in batch]
        bucket_size = len(srcs)

        if self.config["one2one"]:
            trgs_vec = [trg_vec[:-1] + [self.vocab2idx[";"]] for trg_vec in trgs_vec]
            labels = [label[:-1] + [self.vocab2idx[";"]] for label in labels]

        if self.train:
            batch_size = self.config["train_batch_size"]
        else:
            batch_size = self.config["dev_batch_size"]

        original_batch_size = batch_size

        # max_oov_num = max(len(src) for src in srcs)

        if self.config["kpdrop"] > 0 and self.train:
            new_srcs = []
            new_trgs = []
            new_srcs_vec = []
            new_trgs_vec = []
            new_ptr_srcs_vec = []
            new_labels = []
            new_oov_nums = []
            for src, trg in zip(srcs, trgs):
                new_src, new_trg, src_vec, trg_vec, \
                ptr_src_vec, label, oov_num = self.keyphrase_dropout_fn(copy.deepcopy(src),
                                                                        copy.deepcopy(trg))
                new_srcs.append(new_src)
                new_trgs.append(new_trg)
                new_srcs_vec.append(src_vec)
                new_trgs_vec.append(trg_vec)
                new_ptr_srcs_vec.append(ptr_src_vec)
                new_labels.append(label)
                new_oov_nums.append(oov_num)
            if self.config["replace"]:
                srcs = new_srcs
                trgs = new_trgs
                srcs_vec = new_srcs_vec
                ptr_srcs_vec = new_ptr_srcs_vec
                trgs_vec = new_trgs_vec
                labels = new_labels
                oov_nums = new_oov_nums
            else:
                srcs += new_srcs
                trgs += new_trgs
                srcs_vec += new_srcs_vec
                ptr_srcs_vec += new_ptr_srcs_vec
                trgs_vec += new_trgs_vec
                labels += new_labels
                # oov_nums += new_oov_nums
                batch_size = 2 * batch_size
                bucket_size = 2 * bucket_size


        if self.config["truncate"]:
            srcs_vec = [self.src_truncate(src) for src in srcs_vec]
            ptr_srcs_vec = [self.src_truncate(src) for src in ptr_srcs_vec]
            trgs_vec = [self.trg_truncate(trg) for trg in trgs_vec]
            labels = [self.trg_truncate(trg) for trg in labels]

        max_oov_num = max(oov_nums)

        if not self.config["pointer"]:
            labels = trgs_vec

        meta_batches = []

        # print("bucket_size: ", bucket_size)
        # print("batch_size: ", batch_size)

        i = 0
        while i < bucket_size:
            batches = []

            inr = batch_size
            if i + inr > bucket_size:
                inr = bucket_size - i

            inr_ = inr

            j = copy.deepcopy(i)
            while j < i + inr:
                srcs_vec_, src_masks = self.pad(srcs_vec[j:j + inr_], PAD=self.PAD)
                trgs_vec_, trg_masks = self.pad(trgs_vec[j:j + inr_], PAD=self.PAD)
                labels_, _ = self.pad(labels[j:j + inr_], PAD=self.PAD)
                ptr_srcs_vec_, _ = self.pad(ptr_srcs_vec[j:j + inr_], PAD=self.PAD)

                batch = {}
                batch["src_vec"] = T.tensor(srcs_vec_).long()
                trg_vec = T.tensor(trgs_vec_).long()
                batch["trg_vec"] = trg_vec
                batch["ptr_src_vec"] = T.tensor(ptr_srcs_vec_).long()
                batch["src"] = srcs[j:j + inr_]
                batch["trg"] = trgs[j:j + inr_]
                batch["src_mask"] = T.tensor(src_masks).float()
                batch["trg_mask"] = T.tensor(trg_masks).float()
                labels_ = T.tensor(labels_).long()
                assert labels_.size() == trg_vec.size()
                batch["labels"] = labels_
                batch["batch_size"] = min(inr_, original_batch_size)
                batch["max_oov_num"] = max_oov_num
                batches.append(batch)
                j += inr_
            i += inr
            meta_batches.append(batches)

        random.shuffle(meta_batches)

        batches = []
        for batch_list in meta_batches:
            batches = batches + batch_list

        return batches
