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

input_path = Path('../processed_data/kp20k_big_unsup/train.jsonl')
samples = []
with jsonlines.open(input_path, "r") as Reader:
    for id, obj in enumerate(Reader):
        if id % 1000 == 0:
            print("retrieving id", id)
        samples.append(obj)

new_samples = []
for id, obj in enumerate(samples):
    original_src = copy.deepcopy(obj["src"])
    src = obj["src"]
    src = " ".join(src)
    src_sents = nltk.sent_tokenize(src)
    src_sents = [src_sent.split(" ") for src_sent in src_sents]
    pos_tags = []
    stems = []
    if id % 1000 == 0:
        print("processing id: ", id)
    for src_sent in src_sents:
        stem = [ps.stem(w) for w in src_sent]
        pos_tag = nltk.pos_tag(src_sent)
        pos_tag = [p[-1] for p in pos_tag]
        pos_tags.append(pos_tag)
        stems.append(stem)

    trg = obj["trg_no_peos"][0:-1]
    trg = " ".join(trg)
    trg = trg.split(";")
    trg = [item.strip() for item in trg]

    new_obj = {"document_id": id, "tokens": src_sents, "tokens_pos": pos_tags, "tokens_stem": stems, "keyphrases": trg,
               "src": original_src}

    new_samples.append(new_obj)

sample_dict = {}
for item in new_samples:
    for key in item:
        if key in sample_dict:
            sample_dict[key].append(item[key])
        else:
            sample_dict[key] = [item[key]]

part1_sample_dict = {key: sample_dict[key][0:100000] for key in sample_dict}
part2_sample_dict = {key: sample_dict[key][100000:200000] for key in sample_dict}
part3_sample_dict = {key: sample_dict[key][200000:300000] for key in sample_dict}
part4_sample_dict = {key: sample_dict[key][300000:400000] for key in sample_dict}
part5_sample_dict = {key: sample_dict[key][400000:500000] for key in sample_dict}

Path('../processed_data/kp20k_big_unsup1/').mkdir(parents=True, exist_ok=True)
output_path1 = Path('../processed_data/kp20k_big_unsup1/get_keyphrases.json')
jsonl_save(output_path1, part1_sample_dict)

Path('../processed_data/kp20k_big_unsup2/').mkdir(parents=True, exist_ok=True)
output_path2 = Path('../processed_data/kp20k_big_unsup2/get_keyphrases.json')
jsonl_save(output_path2, part2_sample_dict)

Path('../processed_data/kp20k_big_unsup3/').mkdir(parents=True, exist_ok=True)
output_path3 = Path('../processed_data/kp20k_big_unsup3/get_keyphrases.json')
jsonl_save(output_path3, part3_sample_dict)

Path('../processed_data/kp20k_big_unsup4/').mkdir(parents=True, exist_ok=True)
output_path4 = Path('../processed_data/kp20k_big_unsup4/get_keyphrases.json')
jsonl_save(output_path4, part4_sample_dict)

Path('../processed_data/kp20k_big_unsup5/').mkdir(parents=True, exist_ok=True)
output_path5 = Path('../processed_data/kp20k_big_unsup5/get_keyphrases.json')
jsonl_save(output_path5, part5_sample_dict)