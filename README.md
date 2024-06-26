# Keyphrase Dropout
Official code for the paper: KPDROP: Improving Absent Keyphrase Generation (EMNLP Findings 2022)
https://arxiv.org/abs/2112.01476

Generally, we find that we can improve the absent performance of any baseline model that we have tried so far (T5, One2Set, CatSeq, One2One etc.)  with KPDROP-A, or KPDROP-R+beam search without harming present performance. 

Additionally, we were able to contribute to the semi-supervised settings for keyphrase generation. KPDRop allows better exploitation of synthetic data for enhanced absent performance. 

## Credits
* ```get_mebedding.py``` and ```ranker.py``` are taken from: https://github.com/xnliang98/uke_ccrank
* ```models/layers/TransformerSetDecoder.py```, ```models/layers/seq2seq_state.py```, ```models/layers/TransformerSeq2SeqEncoder.py```, and ```models/layers/TransformerSetDecoder.py``` are adapted from: https://github.com/jiacheng-ye/kg_one2set

## Requirements
Check environment.yml (All of them may not be required, but that's the environment I worked on.)

## Relevant Keyphrase Dropout Code:
If you just want to use Keyphrase Dropout in a different codebase you can refer to: https://github.com/JRC1995/KPDrop/blob/main/collaters/seq2seq_collater.py#L39

Its expected inputs, src and trg, are list of tokens. trg should be keyphrases deliminited by ";" and end with "\<eos\>". (\["keyphhrase1-first-word", "keyphrase1-second word", ";" "keyphrase-2", "\<eos\>"\]) (but you can make necessary minor modifications within the code to change these requirements). The main KPDropped outputs are new_src and new_trg which are in the same format as src and trg. You can remove the construction of other return variables and create whatever format of input and output you need from new_src and new_trg as needed for your task. 

## Datasets
* Download datasets from the link provided in https://github.com/kenchan0226/keyphrase-generation-rl.
* Extract and put the extracted folders under ```dataset/```

## Preprocess
* In the root project direction in cmd: ```cd preprocess```
* Then run the following commands sequentially:
* ```python process_kp20k.py```
* ```python process_kp20k2.py```
* ```python process_kp20k_split.py```
* ```python process_kp20k_split2.py```
* ```cd ..```
* ```bash get_embedding.sh``` (use Liang et al. 2021 extractor to generate synthetic labels)
* ```bash ranker.sh``` (use Liang et al. 2021 extractor to generate synthetic labels) 
* ```cd preprocess```
* ```python process_kp20k_big_unsup.py```

Alternatively see if you can download the processed data from [here](https://drive.google.com/file/d/1_3EhngogMD1X0djFwKVLFrb3QInqVJsd/view?usp=sharing) (Keep the processed_data folder in the root directory)

## Supervised Experiments
* (set device=[whatever gpu/cpu device you want] instead of cuda:0, if something else is needed)
* GRU One2Many (baseline): ```python train.py --model=GRUSeq2Seq --model_type=seq2seq --times=3 --dataset=kp20k2 --device=cuda:0```
* GRU One2Many (KPD-R): ```python train.py --model=GRUSeq2SeqKPD0_7 --model_type=seq2seq --times=3 --dataset=kp20k2 --device=cuda:0```
* GRU One2Many (KPD-A): ```python train.py --model=GRUSeq2SeqKPD0_7A --model_type=seq2seq --times=3 --dataset=kp20k2 --device=cuda:0```
* GRU One2One (baseline): ```python train.py --model=GRUSeq2One --model_type=seq2seq --times=3 --dataset=kp20k2 --device=cuda:0```
* GRU One2One (KPD-R): ```python train.py --model=GRUSeq2OneKPD0_7 --model_type=seq2seq --times=3 --dataset=kp20k2 --device=cuda:0```
* GRU One2One (KPD-A): ```python train.py --model=GRUSeq2OneKPD0_7A --model_type=seq2seq --times=3 --dataset=kp20k2 --device=cuda:0```
* Transformer One2Set (baseline): ```python train.py --model=TransformerSeq2Set --model_type=seq2set --times=3 --dataset=kp20k --device=cuda:0```
* Transformer One2Set (KPD-R): ```python train.py --model=TransformerSeq2SetKPD0_7 --model_type=seq2seq --times=3 --dataset=kp20k --device=cuda:0```
* Transformer One2Set (KPD-A): ```python train.py --model=TransformerSeq2SetKPD0_7A --model_type=seq2seq --times=3 --dataset=kp20k --device=cuda:0```

## Semi-Supervised Experiments:
* (set device=[whatever gpu/cpu device you want] instead of cuda:0, if something else is needed)
* GRU One2Many (PT): ```python train.py --model=GRUSeq2Seq --model_type=seq2seq --times=3 --dataset=kp20k_big_unsup --device=cuda:0```
* GRU One2Many (PT+KPD-R): ```python train.py --model=GRUSeq2SeqKPD0_7 --model_type=seq2seq --times=3 --dataset=kp20k_big_unsup --device=cuda:0```
* GRU One2Many (PT+KPD-A): ```python train.py --model=GRUSeq2SeqKPD0_7A --model_type=seq2seq --times=3 --dataset=kp20k_big_unsup --device=cuda:0```
* GRU One2Many (FT): ```python train.py --model=GRUSeq2SeqKPD0_7A --model_type=seq2seq --times=3 --dataset=kp20k_low_res --device=cuda:0```
* GRU One2Many (PT; FT): ```python train.py --model=GRUSeq2SeqKPD0_7Afrom_big --model_type=seq2seq --times=3 --dataset=kp20k_low_res --device=cuda:0```
* GRU One2Many (PT+KPD-R; FT) (only possible after PT+KPD-R): ```python train.py --model=GRUSeq2SeqKPD0_7Afrom_bigKPD0_7 --model_type=seq2seq --times=3 --dataset=kp20k_low_res --device=cuda:0```
* GRU One2Many (PT+KPD-A; FT) (only possible after PT+KPD-A): ```python train.py --model=GRUSeq2SeqKPD0_7Afrom_big0_7A --model_type=seq2seq --times=3 --dataset=kp20k_low_res --device=cuda:0```

(Use the same chain of commands for GRUSeq2One (GRU One2One) and TransformerSeq2Set (TransformerOne2Set) experiments just replace substring GRUSeq2Seq with GRUSeq2One or TransformerSeq2Set whenever applicable,
and replace model_type=seq2seq to model_type=seq2set when using TransformerSeq2Set)


## Decoding
* Add the following arguments when running train.py for Greedy decoding based test: ```--decode_mode=Greedy --test``` 
* Add the following arguments when running train.py for Beam decoding based test: ```--decode_mode=BeamLN --test```

(every other arguments should be same as that for training for the corresponding model)

## Evaluation
In the evaluation, @5R represents the evaluation in the style of [Chan et al.](https://aclanthology.org/P19-1208/) and others, where they do the equivalent of adding "dummy keyphrases" if the top (here top 5) selected keyphrases are less than k (i.e., 5). Our @5R is equivalent to theirs @5 and @5C in our paper (Appendix). 

## Citation

```
@inproceedings{ray-chowdhury-etal-2022-kpdrop,
    title = "{KPDROP}: Improving Absent Keyphrase Generation",
    author = "Ray Chowdhury, Jishnu  and
      Park, Seo Yeon  and
      Kundu, Tuhin  and
      Caragea, Cornelia",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.357",
    pages = "4853--4870",
    abstract = "Keyphrase generation is the task of generating phrases (keyphrases) that summarize the main topics of a given document. Keyphrases can be either present or absent from the given document. While the extraction of present keyphrases has received much attention in the past, only recently a stronger focus has been placed on the generation of absent keyphrases. However, generating absent keyphrases is challenging; even the best methods show only a modest degree of success. In this paper, we propose a model-agnostic approach called keyphrase dropout (or KPDrop) to improve absent keyphrase generation. In this approach, we randomly drop present keyphrases from the document and turn them into artificial absent keyphrases during training. We test our approach extensively and show that it consistently improves the absent performance of strong baselines in both supervised and resource-constrained semi-supervised settings.",
}
```

Contact: jishnu.ray.c@gmail.com for any issues or questions. 

