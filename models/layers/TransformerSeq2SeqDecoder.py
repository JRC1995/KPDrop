import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.attentions import MultiHeadAttention
import math
from models.layers.seq2seq_state import TransformerState
from diffsort import DiffSortNet

class TransformerSeq2SeqDecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, dim_ff=2048, dropout=0.1, layer_idx=None,
                 fix_kp_num_len=False, max_kp_num=20):
        """
        :param int d_model: 输入、输出的维度
        :param int n_head: 多少个head，需要能被d_model整除
        :param int dim_ff:
        :param float dropout:
        :param int layer_idx: layer的编号
        """
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.layer_idx = layer_idx  # 记录layer的层索引，以方便获取state的信息


        self.self_attn = MultiHeadAttention(d_model, n_head, dropout, layer_idx,
                                            fix_kp_num_len, max_kp_num)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)

        self.encoder_attn = MultiHeadAttention(d_model, n_head, dropout, layer_idx)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(self.d_model, self.dim_ff),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(self.dim_ff, self.d_model),
                                 nn.Dropout(dropout))

        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, x, encoder_output, encoder_mask=None, self_attn_mask=None, state=None):
        """
        :param x: (batch, seq_len, dim), decoder端的输入
        :param encoder_output: (batch,src_seq_len,dim), encoder的输出
        :param encoder_mask: batch,src_seq_len, 为1的地方需要attend
        :param self_attn_mask: seq_len, seq_len，下三角的mask矩阵，只在训练时传入
        :param TransformerState state: 只在inference阶段传入
        :return:
        """

        # self attention part
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(query=x,
                              key=x,
                              value=x,
                              attn_mask=self_attn_mask)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # encoder attention part
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x, attn_weight = self.encoder_attn(query=x,
                                           key=encoder_output,
                                           value=encoder_output,
                                           key_mask=encoder_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        # ffn
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn_weight


class TransformerSeq2SeqDecoder(nn.Module):
    def __init__(self, config, embed, pos_embed,  max_kp_len=6, max_kp_num=20):
        """
        :param embed: 输入token的embedding
        :param nn.Module pos_embed: 位置embedding
        :param int d_model: 输出、输出的大小
        :param int num_layers: 多少层
        :param int n_head: 多少个head
        :param int dim_ff: FFN 的中间大小
        :param float dropout: Self-Attention和FFN中的dropout的大小
        """
        super().__init__()

        d_model = config["hidden_size"]
        self.decoder_hidden_size = d_model
        num_layers = config["num_layers"]
        n_head = config["heads"]
        dim_ff = config["ff_dim"]
        dropout = config["dropout"]
        copy_attn = config["pointer"]
        fix_kp_num_len = config["one2set"]
        self.config = config
        self.sorter = DiffSortNet('bitonic', 100, steepness=15, device=self.config["device"])

        self.embed = embed
        self.pos_embed = pos_embed

        self.num_layers = num_layers
        self.d_model = d_model
        self.n_head = n_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_fc = nn.Linear(self.embed.embedding_dim, d_model)
        self.layer_stacks = nn.ModuleList([TransformerSeq2SeqDecoderLayer(d_model, n_head, dim_ff, dropout, layer_idx,
                                                                          fix_kp_num_len, max_kp_num)
                                           for layer_idx in range(num_layers)])
        self.embed_scale = math.sqrt(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.vocab_size = self.embed.num_embeddings
        self.output_fc = nn.Linear(self.d_model, self.embed.embedding_dim)
        self.output_layer = nn.Linear(self.embed.embedding_dim, self.vocab_size, bias=False)

        self.copy_attn = copy_attn
        if copy_attn:
            self.p_gen_linear = nn.Linear(self.embed.embedding_dim, 1)

        self.fix_kp_num_len = fix_kp_num_len
        if self.fix_kp_num_len:
            self.max_kp_len = max_kp_len
            self.max_kp_num = max_kp_num
            if not self.config["contextualized_control_codes"]:
                self.control_code = nn.Embedding(max_kp_num, self.embed.embedding_dim)
                self.control_code.weight.data.uniform_(-0.1, 0.1)
            self.self_attn_mask = self._get_self_attn_mask(max_kp_num, max_kp_len)

            if self.config["contextualized_control_codes"]:
                self.absent_scorer1 = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
                self.absent_scorer2 = nn.Linear(self.decoder_hidden_size, 1)

                self.present_scorer1 = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
                self.present_scorer2 = nn.Linear(self.decoder_hidden_size, 1)

                self.present_rep_linear = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
                self.absent_rep_linear = nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)



    def forward_seg(self, state, first_mask=None):
        encoder_output = state.encoder_output
        batch_size = encoder_output.size(0)
        device = encoder_output.device

        if self.config["contextualized_control_codes"]:
            N, S, D = encoder_output.size()

            assert first_mask.size() == (N, S)
            first_mask = first_mask.unsqueeze(-1)
            # ones_cc = T.ones(N, self.max_kp_num, 1).float().to(first_mask.device)
            # first_mask = T.cat([ones_cc, first_mask], dim=1)


            present_reps = self.present_rep_linear(encoder_output)
            absent_reps = self.absent_rep_linear(encoder_output)
            present_scores = T.exp(self.present_scorer2(F.relu(self.present_scorer1(present_reps)))) * first_mask
            absent_scores = T.exp(self.absent_scorer2(F.relu(self.absent_scorer1(absent_reps)))) * first_mask


            k = 50 #self.max_kp_num // 2
            present_scores, present_idx = T.topk(present_scores, dim=1, k=k)
            absent_scores, absent_idx = T.topk(absent_scores, dim=1, k=k)
            # assert present_scores.size() == (N, k, 1)
            # assert present_idx.size() == (N, k, 1)

            present_reps = T.gather(present_reps, dim=1,
                                    index=present_idx.repeat(1, 1, self.decoder_hidden_size))
            absent_reps = T.gather(absent_reps, dim=1,
                                   index=absent_idx.repeat(1, 1, self.decoder_hidden_size))

            assert present_reps.size() == (N, k, self.decoder_hidden_size)
            assert absent_reps.size() == (N, k, self.decoder_hidden_size)

            present_scores = present_scores.squeeze(-1)
            absent_scores = absent_scores.squeeze(-1)
            assert present_scores.size() == (N, k)
            assert absent_scores.size() == (N, k)

            k2 = self.max_kp_num // 2

            present_scores, present_permute_matrix = self.sorter(present_scores)
            present_permute_matrix = present_permute_matrix.permute(0, 2, 1).contiguous()
            assert present_permute_matrix.size() == (N, k, k)

            present_reps = T.matmul(present_permute_matrix, present_reps)
            present_reps = present_reps.view(N, k, self.decoder_hidden_size)
            present_reps = present_reps[:, -k2:, ...]

            absent_scores, absent_permute_matrix = self.sorter(absent_scores)
            absent_permute_matrix = absent_permute_matrix.permute(0, 2, 1).contiguous()
            assert absent_permute_matrix.size() == (N, k, k)

            absent_reps = T.matmul(absent_permute_matrix, absent_reps)
            absent_reps = absent_reps.view(N, k, self.decoder_hidden_size)
            absent_reps = absent_reps[:, -k2:, ...]


            # assert present_reps.size() == (N, self.max_kp_num // 2, self.decoder_hidden_size)
            # assert absent_reps.size() == (N, self.max_kp_num // 2, self.decoder_hidden_size)

            control_embed = T.cat([present_reps, absent_reps], dim=1)
            scores = None #T.cat([present_scores, absent_scores], dim=1)

            assert control_embed.size() == (N, self.max_kp_num, self.decoder_hidden_size)
        else:
            control_idx = T.arange(0, self.max_kp_num).long().to(device).reshape(1, -1).repeat(batch_size, 1)
            control_embed = self.control_code(control_idx)
            scores = None

        return control_embed, scores

    def forward(self, tokens, state, src_oov, max_num_oov, control_embed=None):
        """
        :param torch.LongTensor tokens: batch x tgt_len，decode的词
        :param TransformerState state: 用于记录encoder的输出以及decode状态的对象，可以通过init_state()获取
        :return: bsz x max_len x vocab_size; 如果return_attention=True, 还会返回bsz x max_len x encode_length
        """


        encoder_output = state.encoder_output
        encoder_mask = state.encoder_mask
        device = tokens.device

        batch_size = encoder_output.size(0)
        S = encoder_output.size(1)
        beam_batch_size = tokens.size(0)
        if beam_batch_size > batch_size:
            beam_size = beam_batch_size // batch_size
            N, S, D = encoder_output.size()
            encoder_output = encoder_output.unsqueeze(1).expand(N, beam_size, S, D)
            encoder_output = encoder_output.reshape(beam_batch_size, S, D)
            assert encoder_mask.size() == (N, S)
            encoder_mask = encoder_mask.unsqueeze(1).expand(N, beam_size, S).reshape(beam_batch_size, S)
            assert src_oov.size() == (N, S)
            src_oov = src_oov.unsqueeze(1).expand(N, beam_size, S).reshape(beam_batch_size, S)

        if self.fix_kp_num_len:
            decode_length = 0 #state.decode_length // self.max_kp_num
            assert decode_length < tokens.size(2), "The decoded tokens in State should be less than tokens."
            tokens = tokens[:, :, decode_length:]
            batch_size, max_kp_num, kp_len = tokens.size()
            max_tgt_len = max_kp_num * kp_len

            D = control_embed.size(-1)
            control_embed = control_embed[0].unsqueeze(0).expand(beam_batch_size, max_kp_num, D).unsqueeze(-2)

            position = T.arange(decode_length, decode_length + kp_len).long().to(device).reshape(1, 1, -1)
            position_embed = self.pos_embed(position)

            word_embed = self.embed_scale * self.embed(tokens)
            embed = self.input_fc(word_embed) + position_embed + control_embed
            x = F.dropout(embed, p=self.dropout, training=self.training)
            x = x.reshape(batch_size, max_kp_num * kp_len, -1)

            if self.self_attn_mask.device is not tokens.device:
                self.self_attn_mask = self.self_attn_mask.to(tokens.device)

            if not self.config["generate"]:  # training
                self_attn_mask = self.self_attn_mask
            else:
                self_attn_mask = self.self_attn_mask.reshape(max_kp_num, self.max_kp_len, max_kp_num, self.max_kp_len)\
                    [:, :kp_len, :, :kp_len] \
                    .reshape(max_kp_num * kp_len, max_kp_num * kp_len)

            for layer in self.layer_stacks:
                x, attn_dist = layer(x=x,
                                     encoder_output=encoder_output,
                                     encoder_mask=encoder_mask,
                                     self_attn_mask=self_attn_mask,
                                     state=state
                                     )

            if self.config["generate"]:
                x = x.reshape(beam_batch_size, max_kp_num, kp_len, -1)
                x = x[:, :, -1, :]
                max_tgt_len = max_kp_num
                attn_dist = attn_dist.reshape(beam_batch_size, max_kp_num, kp_len, S, -1)
                attn_dist = attn_dist[:, :, -1, :]

        else:
            decode_length = 0
            assert state.decode_length < tokens.size(1), "The decoded tokens in State should be less than tokens."
            tokens = tokens[:, decode_length:]

            position = T.arange(decode_length, decode_length + tokens.size(1)).long().to(device)[
                None]
            position_embed = self.pos_embed(position)

            batch_size, max_tgt_len = tokens.size()
            word_embed = self.embed_scale * self.embed(tokens)
            embed = self.input_fc(word_embed) + position_embed
            x = F.dropout(embed, p=self.dropout, training=self.training)
            if max_tgt_len > 1:
                self_attn_mask = self._get_triangle_mask(tokens)
            else:
                self_attn_mask = None

            for layer in self.layer_stacks:
                x, attn_dist = layer(x=x,
                                     encoder_output=encoder_output,
                                     encoder_mask=encoder_mask,
                                     self_attn_mask=self_attn_mask,
                                     state=state
                                     )
            if self.config["generate"]:
                x = x[:, -1, ...].unsqueeze(1)
                max_tgt_len = 1
                attn_dist = attn_dist[:, -1, ...].unsqueeze(1)

        x = self.layer_norm(x)  # batch, tgt_len, dim
        x = self.output_fc(x)

        vocab_dist = F.softmax(self.output_layer(x), -1)
        attn_dist = attn_dist[:, :, :, 0]

        if self.copy_attn:
            p_gen = self.p_gen_linear(x).sigmoid()

            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if max_num_oov > 0:
                extra_zeros = vocab_dist_.new_zeros((batch_size, max_tgt_len, max_num_oov))
                vocab_dist_ = T.cat((vocab_dist_, extra_zeros), dim=-1)

            final_dist = vocab_dist_.scatter_add(2, src_oov.unsqueeze(1).expand_as(attn_dist_), attn_dist_)
            assert final_dist.size() == T.Size([batch_size, max_tgt_len, self.vocab_size + max_num_oov])
        else:
            final_dist = vocab_dist
            assert final_dist.size() == T.Size([batch_size, max_tgt_len, self.vocab_size])
        return final_dist, attn_dist

    def init_state(self, encoder_output, encoder_mask):
        """
        初始化一个TransformerState用于forward
        :param torch.FloatTensor encoder_output: bsz x max_len x d_model, encoder的输出
        :param torch.ByteTensor encoder_mask: bsz x max_len, 为1的位置需要attend。
        :return: TransformerState
        """
        if isinstance(encoder_output, T.Tensor):
            encoder_output = encoder_output
        elif isinstance(encoder_output, (list, tuple)):
            encoder_output = encoder_output[0]  # 防止是LSTMEncoder的输出结果
        else:
            raise TypeError("Unsupported `encoder_output` for TransformerSeq2SeqDecoder")
        state = TransformerState(encoder_output, encoder_mask, num_decoder_layer=self.num_layers)
        return state

    @staticmethod
    def _get_triangle_mask(tokens):
        tensor = tokens.new_ones(tokens.size(1), tokens.size(1))
        return T.tril(tensor).byte()

    @staticmethod
    def _get_self_attn_mask(max_kp_num, max_kp_len):
        mask = T.ones(max_kp_num * max_kp_len, max_kp_num * max_kp_len)
        mask = T.tril(mask).bool()
        for i in range(1, max_kp_num + 1):
            mask[i * max_kp_len:(i + 1) * max_kp_len, :i * max_kp_len] = 0
        return mask
