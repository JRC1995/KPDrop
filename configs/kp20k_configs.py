
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
        self.custom_betas = False
        self.weight_decay = 0.0
        self.lr = 1e-3
        self.epochs = 20
        self.early_stop_patience = 2
        self.scheduler_patience = 0
        self.optimizer = "Adam"
        self.save_by = "loss"
        self.loss_normalization = "batch"
        self.metric_direction = -1
        self.validation_interval = 1
        self.chunk_size = self.batch_size * 4000
        self.num_workers = 6
        self.display_metric = "loss"



class base_config(optimizer_config):
    def __init__(self):
        super().__init__()
        # word embedding
        self.remove_eos = False

        self.word_embd_freeze = False
        self.embd_dim = 100
        self.encoder_hidden_size = 150
        self.decoder_hidden_size = 300
        self.encoder_layers = 1
        self.pointer = True
        self.kpdrop = 0.0

        self.encoder_type = "GRUEncoderDecoder"
        self.coverage_mechanism = False
        self.scratchpad = False
        self.key_value_attention = False
        self.contextualized_control_codes = False

        self.dropout = 0.1
        self.teacher_force_ratio = 1.0
        self.penalty_gamma = 0.0

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
        self.use_dev_as_test = False


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


class TransformerSeq2SeqPointless_config(TransformerSeq2Seq_config):
    def __init__(self):
        super().__init__()
        self.pointer = False
        self.model_name = "(Transformer Seq2Seq Pointless)"

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
        self.model_name = "(Transformer Seq2Set KPD 0.7 A)"

class TransformerSeq2SetKPD0_7_config(TransformerSeq2Set_config):
    def __init__(self):
        super().__init__()
        self.kpdrop = 0.7
        self.replace = True
        self.model_name = "(Transformer Seq2Set KPD 0.7)"

class TransformerSeq2SetCC_config(TransformerSeq2Set_config):
    def __init__(self):
        super().__init__()
        self.one2set = True
        self.one2one = False
        self.max_decoder_len = 6
        self.max_kp_num = 20
        self.penalty_gamma = 1.0
        self.encoder_type = "TransformerSetDecoder"
        self.contextualized_control_codes = True
        self.model_name = "(Transformer Seq2Set CC)"

class TransformerSeq2SetMCC_config(TransformerSeq2Set_config):
    def __init__(self):
        super().__init__()
        self.one2set = True
        self.one2one = False
        self.max_decoder_len = 6
        self.max_kp_num = 20
        self.penalty_gamma = 1.0
        self.null_present_scale = 0.3
        self.null_absent_scale = 0.3
        self.encoder_type = "TransformerSetDecoder"
        self.contextualized_control_codes = True
        self.max_codes = True
        self.model_name = "(Transformer Seq2Set MCC)"

class TransformerSeq2One_config(TransformerSeq2Set_config):
    def __init__(self):
        super().__init__()
        self.one2one = True
        self.one2set = False
        self.model_name = "(Transformer Seq2One)"
