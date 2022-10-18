
class optimizer_config:
    def __init__(self):
        # optimizer config
        self.max_grad_norm = 1.0
        self.batch_size = 12
        self.train_batch_size = 12
        self.dev_batch_size = 12
        self.truncate = False
        self.bucket_size_factor = 1
        self.DataParallel = False
        self.weight_decay = 0.0
        self.lr = 1e-3
        self.epochs = 20
        self.early_stop_patience = 4
        self.scheduler_patience = 0
        self.optimizer = "Adam"
        self.save_by = "loss"
        self.metric_direction = -1
        self.validation_interval = 1
        self.chunk_size = -1
        self.num_workers = 0
        self.display_metric = "loss"
        self.custom_betas = False
        self.use_dev_as_test = False



class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        # word embedding
        self.truncate = False

        self.word_embd_freeze = False
        self.embd_dim = 100
        self.encoder_hidden_size = 150
        self.decoder_hidden_size = 300
        self.encoder_layers = 1
        self.pointer = True

        self.encoder_type = "GRUEncoderDecoder"
        self.coverage_mechanism = False
        self.scratchpad = False
        self.key_value_attention = False

        self.dropout = 0.1
        self.teacher_force_ratio = 1.0
        self.penalty_gamma = 0.0
        self.kpdrop = 0.0

        self.one2one = False
        self.one2set = False

        self.max_decoder_len = 60
        self.hard_exclusion = False
        self.ex_window_size = 4
        self.beam_threshold = 0.0
        self.max_beam_size = 50
        self.beam_width = 6
        self.beam_search = False
        self.length_normalization = False
        self.length_coefficient = 0.8
        self.rerank = False
        self.top_beam = False

class GRUSeq2Seq_config(base_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.0
        self.truncate = True
        self.truncate_src_len = 800
        self.truncate_trg_len = 100
        self.model_name = "(GRU Seq2Seq)"

class GRUSeq2SeqKPD0_7A_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.model_name = "(GRU Seq2Seq KPDROP 0.7A)"


class GRUSeq2SeqKPD0_7Afrom_bigKPD0_7_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.fine_tune = True
        self.pretrained_path = "inference_weights/kp20k_big_unsup_GRUSeq2SeqKPD0_7_seq2seq/2.pt"
        self.model_name = "(GRU Seq2Seq KPDROP 0.7A from bigKPD 0.7)"


class GRUSeq2SeqKPD0_7Afrom_bigKPD2_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.fine_tune = True
        self.pretrained_path = "inference_weights/kp20k_big_unsup_GRUSeq2SeqKPD0_7A_seq2seq/2.pt"
        self.model_name = "(GRU Seq2Seq KPDROP 0.7A from bigKPD2)"


class GRUSeq2SeqKPD0_7Afrom_bigKPD_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.fine_tune = True
        self.pretrained_path = "inference_weights/kp20k_big_unsup_GRUSeq2SeqKPD0_5_seq2seq/2.pt"
        self.model_name = "(GRU Seq2Seq KPDROP 0.7A from bigKPD)"

class GRUSeq2SeqKPD0_7Afrom_big_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.fine_tune = True
        self.pretrained_path = "inference_weights/kp20k_big_unsup_GRUSeq2Seq_seq2seq/2.pt"
        self.model_name = "(GRU Seq2Seq KPDROP 0.7A from big)"


class GRUSeq2One_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.one2one = True
        self.max_decoder_len = 6
        self.model_name = "(GRU Seq2One)"


class GRUSeq2OneKPD0_7A_config(GRUSeq2One_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.model_name = "(GRU Seq2One KPDROP 0.7A)"

class GRUSeq2OneKPD0_7Afrom_big_config(GRUSeq2One_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.fine_tune = True
        self.pretrained_path = "inference_weights/kp20k_big_unsup_GRUSeq2One_seq2seq/2.pt"
        self.model_name = "(GRUSeq2One KPDROP 0.7A from big)"


class GRUSeq2OneKPD0_7Afrom_bigKPD0_7_config(GRUSeq2One_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.fine_tune = True
        self.pretrained_path = "inference_weights/kp20k_big_unsup_GRUSeq2OneKPD0_7_seq2seq/2.pt"
        self.model_name = "(GRUSeq2One KPDROP 0.7A from bigKPD 0.7)"

class GRUSeq2OneKPD0_7Afrom_bigKPD0_7A_config(GRUSeq2One_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.fine_tune = True
        self.pretrained_path = "inference_weights/kp20k_big_unsup_GRUSeq2OneKPD0_7A_seq2seq/2.pt"
        self.model_name = "(GRUSeq2One KPDROP 0.7A from bigKPD 0.7A)"

class TransformerSeq2Seq_config(base_config):
    def __init__(self):
        super().__init__()
        self.lr = 1e-4
        self.embd_dim = 512
        self.hidden_size = 512
        self.train_batch_size = 12
        self.batch_size = 12
        self.num_layers = 6
        self.ff_dim = 2048
        self.dropout = 0.1
        self.position_max_len = 3000
        self.chunk_size = self.batch_size * 8000
        self.loss_normalization = "token"
        self.encoder_type = "TransformerEncoderDecoder"
        self.custom_betas = True
        self.head_dim = 64
        self.heads = 8
        self.attn_dropout = 0.1
        self.scheduler_patience = 0
        self.early_stop_patience = 3
        self.gelu_act = False
        self.relative_pos_enc = False
        self.model_name = "(Transformer Seq2Seq)"

class TransformerSeq2Set_config(TransformerSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.one2set = True
        self.one2one = False
        self.max_decoder_len = 6
        self.max_kp_num = 20
        self.assign_steps = 2
        self.null_present_scale = 0.2
        self.null_absent_scale = 0.1
        self.encoder_type = "TransformerSetDecoder"
        self.specialized_filter = False
        self.contextualized_control_codes = False
        self.max_codes = False
        self.model_name = "(Transformer Seq2Set)"


class TransformerSeq2SetKPD0_7A_config(TransformerSeq2Set_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.model_name = "(Transformer Seq2Set KPD 0.7A)"

class TransformerSeq2SetKPD0_7Afrom_big_config(TransformerSeq2Set_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.fine_tune = True
        self.pretrained_path = "inference_weights/kp20k_big_unsup_TransformerSeq2Set_seq2set/2.pt"
        self.model_name = "(Transformer Seq2Set KPDROP 0.7A from big)"


class TransformerSeq2SetKPD0_7Afrom_bigKPD0_7_config(TransformerSeq2Set_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.fine_tune = True
        self.pretrained_path = "inference_weights/kp20k_big_unsup_TransformerSeq2SetKPD0_7_seq2set/2.pt"
        self.model_name = "(Transformer Seq2Set KPDROP 0.7A from bigKPD 0.7)"

class TransformerSeq2SetKPD0_7Afrom_bigKPD0_7A_config(TransformerSeq2Set_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.fine_tune = True
        self.pretrained_path = "inference_weights/kp20k_big_unsup_TransformerSeq2SetKPD0_7A_seq2set/2.pt"
        self.model_name = "(Transformer Seq2Set KPDROP 0.7A from bigKPD 0.7A)"