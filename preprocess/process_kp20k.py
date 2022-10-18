import copy
import pickle
from os import fspath
from pathlib import Path
import numpy as np
from preprocess_tools.process_utils import jsonl_save
import copy

dev_keys = ["kp20k"]
test_keys = ["kp20k", "krapivin", "inspec", "semeval", "nus"]

train_src_path = Path('../dataset/kp20k_separated/train_src.txt')
train_trg_path = Path('../dataset/kp20k_separated/train_trg.txt')

dev_src_path = {}
dev_trg_path = {}
dev_src_path["kp20k"] = Path('../dataset/kp20k_separated/valid_src.txt')
dev_trg_path["kp20k"] = Path('../dataset/kp20k_separated/valid_trg.txt')

test_src_path = {}
test_trg_path = {}
test_src_path["kp20k"] = Path('../dataset/kp20k_sorted/test_src.txt')
test_trg_path["kp20k"] = Path('../dataset/kp20k_sorted/test_trg.txt')
for key in test_keys[1:]:
    test_src_path[key] = Path('../dataset/cross_domain_sorted/word_{}_testing_context.txt'.format(key))
    test_trg_path[key] = Path('../dataset/cross_domain_sorted/word_{}_testing_allkeywords.txt'.format(key))

Path('../processed_data/kp20k/').mkdir(parents=True, exist_ok=True)

train_save_path = Path('../processed_data/kp20k/train.jsonl')
dev_save_path = {}
for key in dev_keys:
    dev_save_path[key] = Path('../processed_data/kp20k/dev_{}.jsonl'.format(key))
test_save_path = {}
for key in test_keys:
    test_save_path[key] = Path('../processed_data/kp20k/test_{}.jsonl'.format(key))
metadata_save_path = fspath(Path("../processed_data/kp20k/metadata.pkl"))

vocab2count = {}


def process_data(src_filename, trg_filename, train=True, update_vocab=True):
    global vocab2count
    print("\n\nOpening directory: {}\n\n".format(src_filename))

    srcs = []
    trgs = []
    trgs_no_peos = []

    count = 0

    with open(src_filename) as reader:
        lines = reader.readlines()
        for sample in lines:
            sample = sample.lower().strip().replace(" <eos>", "").split(" ")
            srcs.append(sample)
            if update_vocab:
                for token in sample:
                    vocab2count[token] = vocab2count.get(token, 0) + 1
        count += 1

        """
        if count % 1000 == 0:
            print("Processing Data # {}...".format(count))
        """

    print("\n\nOpening directory: {}\n\n".format(trg_filename))

    count = 0

    with open(trg_filename) as reader:
        lines = reader.readlines()
        for sample in lines:
            sample = sample.lower().strip().replace(";", " ; ").split(" ")
            sample = sample + ["<eos>"]
            sample = [kp.strip() for kp in sample if kp.strip() != ""]
            sample_no_peos = []
            flag = 0
            for token in sample:
                token = token.strip()
                if token == "<peos>":
                    flag = 1
                else:
                    if flag == 1 and token == ";":
                        flag = 0
                    else:
                        if token != "":
                            sample_no_peos.append(token)
            # print(sample_no_peos)
            trgs.append(sample)
            trgs_no_peos.append(sample_no_peos)
            if update_vocab:
                for token in sample:
                    vocab2count[token] = vocab2count.get(token, 0) + 1
        count += 1

    srcs_dict = {}
    duplicate_count = 0
    full_duplicate_count = 0
    zero_trg_count = 0
    high_len_count = 0
    zero_count = 0
    duplicate_and_zero_count = 0
    srcs_ = []
    trgs_ = []
    trgs_no_peos_ = []

    for src, trg, trg_no_peos in zip(srcs, trgs, trgs_no_peos):
        flag = 0
        src_string = " ".join(src)
        trg_string = " ".join(trg[0:-1])
        kps = trg_string.split(" ; ")
        kps = [kp.strip() for kp in kps if kp.strip() != ""]
        if src_string not in srcs_dict:
            srcs_dict[src_string] = [trg_string]
            flag = 0
        else:
            duplicate_count += 1
            if trg_string in srcs_dict[src_string]:
                full_duplicate_count += 1
                flag = 1
            else:
                srcs_dict[src_string].append(trg_string)

        if len(kps) == 0:
            zero_trg_count += 1
            if train and flag != 1:
                flag = 1

        if len(src) > 512 and train and flag != 1:
            flag = 1

        if len(kps) > 20 and train and flag != 1:
            flag = 1

        if flag == 0:
            srcs_.append(src)
            trgs_.append(trg)
            trgs_no_peos_.append(trg_no_peos)
        # print("src: ", src)
        # print("trg: ", trg)

    srcs = srcs_
    trgs = trgs_
    trgs_no_peos = trgs_no_peos_

    print("Duplicate Counts: ", duplicate_count)
    print("Full Duplicate Counts: ", full_duplicate_count)
    print("zero trg count: ", zero_trg_count)
    print("total samples: ", len(srcs))

    assert len(srcs) == len(trgs)

    return srcs, trgs, trgs_no_peos


train_srcs, train_trgs, train_trgs_no_peos = process_data(train_src_path, train_trg_path, train=True, update_vocab=True)

dev_srcs = {}
dev_trgs = {}
dev_trgs_no_peos = {}
for key in dev_keys:
    dev_srcs[key], dev_trgs[key], \
    dev_trgs_no_peos[key] = process_data(dev_src_path[key], dev_trg_path[key], train=False, update_vocab=True)

test_srcs = {}
test_trgs = {}
test_trgs_no_peos = {}
for key in test_keys:
    print(key)
    test_srcs[key], test_trgs[key], \
    test_trgs_no_peos[key] = process_data(test_src_path[key], test_trg_path[key], train=False,
                                          update_vocab=False)

counts = []
vocab = []
for word, count in vocab2count.items():
    vocab.append(word)
    counts.append(count)

MAX_VOCAB = 50000 - 3
sorted_idx = np.flip(np.argsort(counts), axis=0)
vocab = [vocab[id] for id in sorted_idx]
if len(vocab) > MAX_VOCAB:
    vocab = vocab[0:MAX_VOCAB]

vocab2idx = {}
for i, token in enumerate(vocab):
    vocab2idx[token] = i

special_tags = [";", "<unk>", "<eos>", "<pad>", "<digit>", "<null>", "<sos>"]
for token in special_tags:
    if token not in vocab2idx:
        vocab2idx[token] = len(vocab2idx)


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

def vectorize_data(srcs, trgs, trgs_no_peos):
    data_dict = {}
    srcs_vec = [text_vectorize(src) for src in srcs]
    trgs_vec = [text_vectorize(trg) for trg in trgs]
    trgs_no_peos_vec = [text_vectorize(trg_no_peos) for trg_no_peos in trgs_no_peos]
    ptr_srcs_vec, oov_nums = create_ptr_src(srcs, srcs_vec)
    labels = [create_labels(trg, trg_vec, src) for trg_vec, src, trg in zip(trgs_no_peos_vec, srcs, trgs_no_peos)]
    first_masks = [create_first_mask(src) for src in srcs_vec]

    set_trgs_vec = []
    set_labels = []
    for trg, trg_vec, src in zip(trgs, trgs_vec, srcs):
        set_trg_vec, set_label = reorg_trg(trg, copy.deepcopy(trg_vec), src)
        set_trgs_vec.append(set_trg_vec)
        set_labels.append(set_label)

    data_dict["src"] = srcs
    data_dict["trg"] = trgs
    data_dict["trg_no_peos"] = trgs_no_peos

    data_dict["src_vec"] = srcs_vec
    data_dict["ptr_src_vec"] = ptr_srcs_vec
    data_dict["oov_num"] = oov_nums

    data_dict["trg_vec"] = trgs_vec
    data_dict["trg_no_peos_vec"] = trgs_no_peos_vec
    data_dict["set_trg_vec"] = set_trgs_vec
    data_dict["set_label"] = set_labels
    data_dict["label"] = labels
    data_dict["first_mask"] = first_masks
    return data_dict


train_data = vectorize_data(train_srcs, train_trgs, train_trgs_no_peos)
jsonl_save(filepath=train_save_path,
           data_dict=train_data)

dev_data = {}
for key in dev_keys:
    dev_data[key] = vectorize_data(dev_srcs[key], dev_trgs[key], dev_trgs_no_peos[key])
    jsonl_save(filepath=dev_save_path[key],
               data_dict=dev_data[key])

test_data = {}
for key in test_keys:
    test_data[key] = vectorize_data(test_srcs[key], test_trgs[key], test_trgs_no_peos[key])
    jsonl_save(filepath=test_save_path[key],
               data_dict=test_data[key])

metadata = {"dev_keys": dev_keys,
            "test_keys": test_keys,
            "vocab2idx": vocab2idx}

with open(metadata_save_path, 'wb') as outfile:
    pickle.dump(metadata, outfile)
