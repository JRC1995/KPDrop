# Keyphrase Dropout
This is code for XYZ2500 paper: KPDROP: Improving Absent Keyphrase Generation.


## Credits
* ```get_mebedding.py``` and ```ranker.py``` are taken from: https://github.com/xnliang98/uke_ccrank
* ```models/layers/TransformerSetDecoder.py```, ```models/layers/seq2seq_state.py```, ```models/layers/TransformerSeq2SeqEncoder.py```, and ```models/layers/TransformerSetDecoder.py``` are adapted from: https://github.com/jiacheng-ye/kg_one2set

## Requirements
Check environment.yml (All of them may not be required, but that's the environment I worked on.)

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
* ```python process_kp20k_big_unsup.py```

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
* Add the following arguments for Greedy decoding based test: ```--decode_mode=Greedy --test```
* Add the following arguments for Beam decoding based test: ```--decode_mode=BeamLN --test```
