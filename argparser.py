import argparse
from argparse import ArgumentParser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser(description="LanguageProcessors Arguments")
    parser.add_argument('--model', type=str, default="TransformerSeq2Seq",
                        choices=["GRUSeq2Seq",
                                 "GRUSeq2SeqKPD0_5",
                                 "GRUSeq2SeqKPD0_3tune",
                                 "GRUSeq2SeqKPD0_5tune",
                                 "GRUSeq2SeqKPD0_7tune",
                                 "GRUSeq2SeqKPD0_9tune",
                                 "GRUSeq2SeqKPD1_0tune",
                                 "GRUSeq2SeqKPD0_7",
                                 "GRUSeq2SeqKPD0_7Afrom_big",
                                 "GRUSeq2SeqKPD0_7Afrom_bigKPD",
                                 "GRUSeq2SeqKPD0_7Afrom_bigKPD2",
                                 "GRUSeq2SeqKPD0_7Afrom_bigKPD0_7",
                                 "GRUSeq2SeqKPD0_7A",
                                 "GRUSeq2One",
                                 "GRUSeq2OneKPD0_7",
                                 "GRUSeq2OneKPD0_7A",
                                 "GRUSeq2OneKPD0_7Afrom_big",
                                 "GRUSeq2OneKPD0_7Afrom_bigKPD0_7",
                                 "GRUSeq2OneKPD0_7Afrom_bigKPD0_7A",
                                 "TransformerSeq2Set",
                                 "TransformerSeq2SetKPD0_7",
                                 "TransformerSeq2SetKPD0_7A",
                                 "TransformerSeq2SetKPD0_7Afrom_big",
                                 "TransformerSeq2SetKPD0_7Afrom_bigKPD0_7",
                                 "TransformerSeq2SetKPD0_7Afrom_bigKPD0_7A"])
    parser.add_argument('--no_display', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--display_params', type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument('--test', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--model_type', type=str, default="seq2seq",
                        choices=["seq_label", "seq2seq", "seq2set"])
    parser.add_argument('--decode_mode', type=str, default="Greedy",
                        choices=["Greedy", "GreedyES1", "GreedyES4", "Beam", "BeamLN", "Beam11LN", "Beam20LN","Beam50LN",
                                 "TopBeam5LN", "TopBeam50LN",
                                 "AdaBeam50LN", "AdaBeam20LN", "BeamLN_ES1", "BeamLN_ES4"])
    parser.add_argument('--dataset', type=str, default="kp20k",
                        choices=["kp20k", "kp20k2", "kp20k_big_unsup", "kp20k_low_res"])
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--initial_time', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--example_display_step', type=int, default=500)
    parser.add_argument('--load_checkpoint', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--reproducible', type=str2bool, default=True, const=True, nargs='?')
    return parser
