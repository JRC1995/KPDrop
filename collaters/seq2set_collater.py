import torch as T
import numpy as np
import random
import copy
from nltk.stem import PorterStemmer


class seq2set_collater:
    def __init__(self, PAD, config, train):
        self.PAD = PAD
        self.config = config
        self.train = train
        self.UNK_id = config["UNK_id"]
        self.vocab_len = config["vocab_len"]
        self.separator_id = config["vocab2idx"][";"]
        self.eos_id = config["vocab2idx"]["<eos>"]
        self.idx2vocab = config["idx2vocab"]
        self.vocab2idx = config["vocab2idx"]
        if config["one2set"]:
            self.null_id = config["NULL_id"]
        self.stemmer = PorterStemmer()

    def pad(self, items, PAD):
        # max_len = max([len(item) for item in items])
        item_lens = [len(item) for item in items]
        max_len = max(item_lens)
        item_pad = [PAD] * max_len
        zeros = [0] * max_len
        ones = [1] * max_len

        padded_items = []
        item_masks = []
        for item, item_len in zip(items, item_lens):
            if item_len < max_len:
                item = item + item_pad[0:max_len - item_len]
                mask = ones[0:item_len] + zeros[0:max_len - item_len]
            else:
                mask = ones
            padded_items.append(item)
            item_masks.append(mask)

        return padded_items, item_masks

    def pad_no_mask(self, items, PAD):
        # max_len = max([len(item) for item in items])
        item_lens = [len(item) for item in items]
        max_len = max(item_lens)
        item_pad = [PAD] * max_len

        padded_items = []
        for item, item_len in zip(items, item_lens):
            if item_len < max_len:
                item = item + item_pad[0:max_len - item_len]
            padded_items.append(item)

        return padded_items

    def sort_list(self, objs, idx):
        return [objs[i] for i in idx]

    def text_vectorize(self, text):
        return [self.vocab2idx.get(word, self.vocab2idx['<unk>']) for word in text]

    def reorg_trg(self, trg, trg_vec, src):

        max_decoder_len = 6
        eos_id = self.vocab2idx["<eos>"]
        PAD = self.vocab2idx["<pad>"]
        UNK_id = self.vocab2idx["<unk>"]
        null_id = self.vocab2idx["<null>"]
        vocab_len = len(self.vocab2idx)
        max_kp_num = 20

        src_token_dict = {}
        for pos, token in enumerate(src):
            if token not in src_token_dict:
                src_token_dict[token] = len(src_token_dict)

        present_sep_kps = []
        absent_sep_kps = []
        present_sep_label_kps = []
        absent_sep_label_kps = []
        kp = []
        label_kp = []
        absent_flag = 0
        for token_id, token in zip(trg_vec, trg):
            if token == "<peos>":
                absent_flag = 1
            elif token == ";" or token == "<eos>" or token == "<sep>":
                if kp:
                    kp.append(eos_id)
                    label_kp.append(eos_id)
                    kp = kp[0:max_decoder_len]
                    label_kp = label_kp[0:max_decoder_len]
                    while len(kp) < max_decoder_len:
                        kp.append(PAD)
                        label_kp.append(PAD)
                    if absent_flag == 1:
                        absent_sep_kps.append(kp)
                        absent_sep_label_kps.append(label_kp)
                    else:
                        present_sep_kps.append(kp)
                        present_sep_label_kps.append(label_kp)
                    kp = []
                    label_kp = []
            else:
                kp.append(token_id)
                ptr_token_id = None
                if token_id == UNK_id:
                    if token in src_token_dict:
                        ptr_token_id = vocab_len + src_token_dict[token]
                if ptr_token_id is None:
                    label_kp.append(token_id)
                else:
                    label_kp.append(ptr_token_id)

        present_sep_kps = present_sep_kps[0:max_kp_num // 2]
        absent_sep_kps = absent_sep_kps[0:max_kp_num // 2]
        present_sep_label_kps = present_sep_label_kps[0:max_kp_num // 2]
        absent_sep_label_kps = absent_sep_label_kps[0:max_kp_num // 2]
        PAD_KP = [null_id] + [PAD] * (max_decoder_len - 1)
        while len(present_sep_kps) < (max_kp_num // 2):
            present_sep_kps.append(PAD_KP)
            present_sep_label_kps.append(PAD_KP)
        while len(absent_sep_kps) < (max_kp_num // 2):
            absent_sep_kps.append(PAD_KP)
            absent_sep_label_kps.append(PAD_KP)

        sep_kps = present_sep_kps + absent_sep_kps
        sep_label_kps = present_sep_label_kps + absent_sep_label_kps

        return sep_kps, sep_label_kps

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

        keyphrases = new_present_keyphrases + ["<peos>"] + new_absent_keyphrases + original_absent_keyphrases
        keyphrases_no_peos = new_present_keyphrases + new_absent_keyphrases + original_absent_keyphrases

        new_trg = " ; ".join(keyphrases).split(" ")
        new_trg = new_trg + ["<eos>"]
        # print("post drop trg: ", new_trg)

        new_trg_no_peos = " ; ".join(keyphrases_no_peos).split(" ")
        new_trg_no_peos = new_trg_no_peos + ["<eos>"]

        src_vec = self.text_vectorize(new_src)
        trg_vec = self.text_vectorize(new_trg)

        set_trg_vec, set_label = self.reorg_trg(new_trg, trg_vec, new_src)

        src_token_dict = {}
        for token in new_src:
            if token not in src_token_dict:
                src_token_dict[token] = self.vocab_len + len(src_token_dict)

        ptr_src_vec = [self.vocab2idx.get(token, src_token_dict.get(token, self.UNK_id)) for token in new_src]

        return new_src, new_trg_no_peos, src_vec, set_trg_vec, ptr_src_vec, set_label, len(src_token_dict)

    def collate_fn(self, batch):
        batch = copy.deepcopy(batch)
        srcs_vec = [obj['src_vec'] for obj in batch]
        srcs = [obj['src'] for obj in batch]
        ptr_srcs_vec = [obj["ptr_src_vec"] for obj in batch]
        set_trgs_vec = [obj['set_trg_vec'] for obj in batch]
        set_labels = [obj['set_label'] for obj in batch]
        trgs = [obj['trg'] for obj in batch]
        first_masks = [obj["first_mask"] for obj in batch]
        oov_nums = [obj["oov_num"] for obj in batch]

        if not self.config["pointer"]:
            set_labels = copy.deepcopy(set_trgs_vec)

        bucket_size = len(srcs)
        if self.train:
            batch_size = self.config["train_batch_size"]
        else:
            batch_size = self.config["dev_batch_size"]

        original_batch_size = batch_size

        if self.config["kpdrop"] > 0 and self.train:
            new_srcs = []
            new_trgs = []
            new_srcs_vec = []
            new_set_trgs_vec = []
            new_ptr_srcs_vec = []
            new_set_labels = []
            new_oov_nums = []
            for src, trg in zip(srcs, trgs):
                new_src, new_trg, src_vec, set_trg_vec, \
                ptr_src_vec, set_label, oov_num = self.keyphrase_dropout_fn(copy.deepcopy(src),
                                                                            copy.deepcopy(trg))
                new_srcs.append(new_src)
                new_trgs.append(new_trg)
                new_srcs_vec.append(src_vec)
                new_set_trgs_vec.append(set_trg_vec)
                new_ptr_srcs_vec.append(ptr_src_vec)
                new_set_labels.append(set_label)
                new_oov_nums.append(oov_num)
            if self.config["replace"]:
                srcs = new_srcs
                trgs = new_trgs
                srcs_vec = new_srcs_vec
                ptr_srcs_vec = new_ptr_srcs_vec
                set_trgs_vec = new_set_trgs_vec
                set_labels = new_set_labels
                oov_nums = new_oov_nums
            else:
                srcs += new_srcs
                trgs += new_trgs
                srcs_vec += new_srcs_vec
                ptr_srcs_vec += new_ptr_srcs_vec
                set_trgs_vec += new_set_trgs_vec
                set_labels += new_set_labels
                oov_nums += new_oov_nums
                batch_size = 2 * batch_size
                bucket_size = 2 * bucket_size

        max_oov_num = max(oov_nums)
        if not self.config["pointer"]:
            set_labels = set_trgs_vec

        meta_batches = []

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
                ptr_srcs_vec_ = self.pad_no_mask(ptr_srcs_vec[j:j + inr_], PAD=self.PAD)
                first_masks_ = self.pad_no_mask(first_masks[j:j + inr_], PAD=0)
                set_labels_ = T.tensor(set_labels[j:j + inr_]).long()
                batch = {}
                batch["src_vec"] = T.tensor(srcs_vec_).long()
                batch["trg_vec"] = T.tensor(set_trgs_vec[j:j + inr_]).long()
                batch["ptr_src_vec"] = T.tensor(ptr_srcs_vec_).long()
                batch["src"] = srcs[j:j + inr_]
                batch["trg"] = trgs[j:j + inr_]
                batch["first_mask"] = T.tensor(first_masks_).float()
                batch["src_mask"] = T.tensor(src_masks).float()
                batch["labels"] = set_labels_
                batch["trg_mask"] = T.where(set_labels_ == self.PAD,
                                            T.zeros_like(set_labels_).long(),
                                            T.ones_like(set_labels_).long())
                batch["batch_size"] = batch["batch_size"] = min(inr_, original_batch_size)
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
