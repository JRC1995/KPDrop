
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
        self.early_stop_patience = 2
        self.scheduler_patience = 0
        self.optimizer = "Adam"
        self.save_by = "loss"
        self.metric_direction = -1
        self.validation_interval = 1
        self.chunk_size = self.batch_size * 4000
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

class GRUSeq2One_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.one2one = True
        self.max_decoder_len = 60
        self.model_name = "(GRU Seq2One)"

class GRUSeq2OneKPD0_7_config(GRUSeq2One_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = True
        self.model_name = "(GRU Seq2One KPDROP 0.7)"

class GRUSeq2OneKPD0_7A_config(GRUSeq2One_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.model_name = "(GRU Seq2One KPDROP 0.7A)"

class GRUSeq2SeqKPD0_7_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = True
        self.model_name = "(GRU Seq2Seq KPDROP 0.7)"

class GRUSeq2SeqKPD0_7tune_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = True
        self.use_dev_as_test = True
        self.epochs = 1
        self.model_name = "(GRU Seq2Seq KPDROP 0.7)"

class GRUSeq2SeqKPD0_5tune_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.5
        self.replace = True
        self.use_dev_as_test = True
        self.epochs = 1
        self.model_name = "(GRU Seq2Seq KPDROP 0.5)"

class GRUSeq2SeqKPD0_9tune_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.9
        self.replace = True
        self.use_dev_as_test = True
        self.epochs = 1
        self.model_name = "(GRU Seq2Seq KPDROP 0.9)"

class GRUSeq2SeqKPD0_3tune_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.3
        self.replace = True
        self.use_dev_as_test = True
        self.epochs = 1
        self.model_name = "(GRU Seq2Seq KPDROP 0.3)"

class GRUSeq2SeqKPD1_0tune_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 1.0
        self.replace = True
        self.use_dev_as_test = True
        self.epochs = 1
        self.model_name = "(GRU Seq2Seq KPDROP 1.0)"

class GRUSeq2SeqKPD0_7A_config(GRUSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = False
        self.model_name = "(GRU Seq2Seq KPDROP 0.7A)"
