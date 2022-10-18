import pickle
import random
from os import fspath
from pathlib import Path
import numpy as np
from preprocess_tools.process_utils import jsonl_save
import copy
import nltk
import jsonlines
from nltk.stem import PorterStemmer
ps = PorterStemmer()

metadata_path = Path("../processed_data/kp20k_big_unsup/metadata.pkl")
with open(metadata_path, 'rb') as fp:
    metadata = pickle.load(fp)

vocab2idx = metadata["vocab2idx"]
print(len(vocab2idx))

trgs = []

prediction1_path = Path("../processed_data/kp20k_big_unsup1/predicted.txt")
prediction2_path = Path("../processed_data/kp20k_big_unsup2/predicted.txt")
prediction3_path = Path("../processed_data/kp20k_big_unsup3/predicted.txt")
prediction4_path = Path("../processed_data/kp20k_big_unsup4/predicted.txt")

paths = [prediction1_path, prediction2_path, prediction3_path, prediction4_path]

for filepath in paths:
    with open(filepath) as fp:
        lines = fp.readlines()
        for line in lines:
            line = ";".join(line.split(";")[0:10])
            sample = line.lower().strip().replace(";", " ; ").split(" ")
            sample = sample + ["<eos>"]
            sample = [kp.strip() for kp in sample if kp.strip() != ""]
            trgs.append(sample)


def text_vectorize(text):
    return [vocab2idx.get(word, vocab2idx['<unk>']) for word in text]

def create_first_mask(src):
    fm = []
    item_dict = {}
    for item in src:
        if item in item_dict:
            fm.append(0)
        else:
            fm.append(1)
            item_dict[item] = 1
    return fm

def create_ptr_src(srcs, srcs_vec):
    ptr_srcs_vec = []
    oov_nums = []

    global vocab2idx
    vocab_len = len(vocab2idx)
    UNK_id = vocab2idx["<unk>"]

    for src, src_vec in zip(srcs, srcs_vec):
        token_dict = {}
        ptr_src_vec = []
        for token, token_id in zip(src, src_vec):
            if token not in token_dict:
                token_dict[token] = len(token_dict)
            if token_id == UNK_id:
                ptr_src_vec.append(vocab_len + token_dict[token])
            else:
                ptr_src_vec.append(token_id)
        ptr_srcs_vec.append(ptr_src_vec)
        oov_nums.append(len(token_dict))
    return ptr_srcs_vec, oov_nums


def create_labels(trg, trg_vec, src):
    global vocab2idx
    vocab_len = len(vocab2idx)
    UNK_id = vocab2idx["<unk>"]
    label = []
    src_token_dict = {}
    for pos, token in enumerate(src):
        if token not in src_token_dict:
            src_token_dict[token] = len(src_token_dict)
    for token, id in zip(trg, trg_vec):
        if (id == UNK_id) and (token in src_token_dict):
            label.append(vocab_len + src_token_dict[token])
        else:
            label.append(id)

    assert len(label) == len(trg_vec)

    return label

def reorg_trg(trg, trg_vec, src):

    global vocab2idx

    max_decoder_len = 6
    eos_id = vocab2idx["<eos>"]
    PAD = vocab2idx["<pad>"]
    UNK_id = vocab2idx["<unk>"]
    null_id = vocab2idx["<null>"]
    vocab_len = len(vocab2idx)
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

trgs_vec = [text_vectorize(trg) for trg in trgs]
trgs_no_peos = trgs
trgs_no_peos_vec = trgs_vec


input_path = Path('../processed_data/kp20k_big_unsup/train.jsonl')
samples = []
with jsonlines.open(input_path, "r") as Reader:
    for id, obj in enumerate(Reader):
        if id % 1000 == 0:
            print("retrieving id", id)
        samples.append(obj)

srcs = []
new_samples = []
for id, obj in enumerate(samples):
    src = obj["src"]
    src_vec = obj["src_vec"]
    srcs.append(src)
    label = create_labels(trgs_no_peos[id], trgs_no_peos_vec[id], src)
    first_mask = create_first_mask(src_vec)
    set_trg_vec, set_label = reorg_trg(trgs_no_peos[id], copy.deepcopy(trgs_no_peos_vec[id]), src)

    obj_ = {}
    for key in obj:
        obj_[key] = obj[key]

    obj_["trg"] = trgs[id]
    obj_["trg_no_peos"] = trgs_no_peos[id]
    obj_["trg_no_peos_vec"] = trgs_no_peos_vec[id]
    obj_["first_mask"] = first_mask
    obj_["label"] = label
    obj_["set_trg_vec"] = set_trg_vec
    obj_["set_label"] = set_label


    new_samples.append(obj_)

    if len(srcs) == 400000:
        break

assert len(srcs) == len(trgs)


rand_idx = [i for i in range(len(srcs))]
random.shuffle(rand_idx)

for id in rand_idx[0:100]:
    sample = new_samples[id]
    print("id: ", id)
    print("src: ", sample["src"])
    print("trg: ", sample["trg"])
    print("\n\n")

train_save_path_big_unsup = Path('../processed_data/kp20k_big_unsup/train.jsonl')
with jsonlines.open(train_save_path_big_unsup, mode='w') as writer:
    writer.write_all(new_samples)



