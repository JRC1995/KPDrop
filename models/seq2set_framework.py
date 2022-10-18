import torch.nn as nn
from models.layers import *


class seq2set_framework(nn.Module):
    def __init__(self, data, config):

        super(seq2set_framework, self).__init__()

        self.config = config
        encoder_decoder_fn = eval(config["encoder_type"])
        self.encoder_decoder = encoder_decoder_fn(self.config)

    # %%
    def forward(self, batch):

        src = batch["src_vec"]
        ptr_src = batch["ptr_src_vec"]
        src_mask = batch["src_mask"]


        if "trg_vec" in batch:
            trg = batch["trg_vec"]
            trg_mask = batch["trg_mask"]
            labels = batch["labels"]
        else:
            trg = None
            labels = None
            trg_mask = None

        # EMBEDDING BLOCK
        sequence_dict = self.encoder_decoder(src_idx=src,
                                             ptr_src_idx=ptr_src,
                                             input_mask=src_mask,
                                             trg_idx=trg,
                                             output_mask=trg_mask,
                                             labels=labels,
                                             max_oov_num=batch["max_oov_num"],
                                             first_mask=batch["first_mask"])

        return sequence_dict
