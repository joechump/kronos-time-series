import numpy as np
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
import sys

from tqdm import trange

sys.path.append("../")
from model.module import *


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    KronosTokenizer模块：使用混合量化方法对输入数据进行标记化

    该标记器利用编码器和解码器Transformer块的组合，
    以及二进制球面量化（BSQuantizer）来压缩和解压缩输入数据。
    主要用于将连续的时间序列数据转换为离散的token表示，
    支持双向编码-解码架构和分层量化策略。

    参数:
           d_in (int): 输入维度。
           d_model (int): 模型维度。
           n_heads (int): 注意力头数。
           ff_dim (int): 前馈网络维度。
           n_enc_layers (int): 编码器层数。
           n_dec_layers (int): 解码器层数。
           ffn_dropout_p (float): 前馈网络的dropout概率。
           attn_dropout_p (float): 注意力机制的dropout概率。
           resid_dropout_p (float): 残差连接的dropout概率。
           s1_bits (int): BSQuantizer中pre token的比特数。
           s2_bits (int): BSQuantizer中post token的比特数。
           beta (float): BSQuantizer的beta参数。
           gamma0 (float): BSQuantizer的gamma0参数。
           gamma (float): BSQuantizer的gamma参数。
           zeta (float): BSQuantizer的zeta参数。
           group_size (int): BSQuantizer的组大小参数。

    """

    def __init__(self, d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers, ffn_dropout_p, attn_dropout_p, resid_dropout_p, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):

        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits # 量化后码本的总维度
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        # 编码器Transformer块
        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])
        # 解码器Transformer块
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])
        self.quant_embed = nn.Linear(in_features=self.d_model, out_features=self.codebook_dim) # 量化前的线性层
        self.post_quant_embed_pre = nn.Linear(in_features=self.s1_bits, out_features=self.d_model) # 量化后的线性层（pre部分 - s1比特）
        self.post_quant_embed = nn.Linear(in_features=self.codebook_dim, out_features=self.d_model) # 量化后的线性层（完整码本）
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size) # BSQuantizer模块

    def forward(self, x):
        """
        KronosTokenizer的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_in)。

        返回:
            tuple: 包含以下元素的元组:
                - tuple: (z_pre, z) - 解码器使用s1_bits和完整码本的重构输出，
                         形状均为 (batch_size, seq_len, d_in)。
                - torch.Tensor: bsq_loss - BSQuantizer的损失。
                - torch.Tensor: quantized - BSQuantizer的量化表示。
                - torch.Tensor: z_indices - BSQuantizer的索引。
        """
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z) # (B, T, codebook)

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, :self.s1_bits] # 提取量化表示的第一部分（s1比特）
        z_pre = self.post_quant_embed_pre(quantized_pre)

        z = self.post_quant_embed(quantized)

        # 解码器层（用于pre部分 - s1比特）
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        # 解码器层（用于完整码本）
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        """
        将索引转换为比特表示并进行缩放。

        参数:
            x (torch.Tensor): 索引张量。
            half (bool, 可选): 是否只处理码本维度的一半。默认为False。

        返回:
            torch.Tensor: 比特表示张量。
        """
        if half:
            x1 = x[0] # 假设x是索引元组，如果half为True
            x2 = x[1]
            mask = 2 ** torch.arange(self.codebook_dim//2, device=x1.device, dtype=torch.long) # 创建比特提取的掩码
            x1 = (x1.unsqueeze(-1) & mask) != 0 # 提取第一半的比特
            x2 = (x2.unsqueeze(-1) & mask) != 0 # 提取第二半的比特
            x = torch.cat([x1, x2], dim=-1) # 连接比特表示
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long) # 创建比特提取的掩码
            x = (x.unsqueeze(-1) & mask) != 0 # 提取比特

        x = x.float() * 2 - 1 # 将布尔值转换为双极值（-1, 1）
        q_scale = 1. / (self.codebook_dim ** 0.5) # 缩放因子
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        """
        将输入数据编码为量化索引。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_in)。
            half (bool, 可选): 是否在BSQuantizer中使用半量化。默认为False。

        返回:
            torch.Tensor: BSQuantizer的量化索引。
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices = self.tokenizer(z, half)
        return z_indices

    def decode(self, x, half=False):
        """
        将量化索引解码回输入数据空间。

        参数:
            x (torch.Tensor): 量化索引张量。
            half (bool, 可选): 索引是否使用半量化生成。默认为False。

        返回:
            torch.Tensor: 重构的输出张量，形状为 (batch_size, seq_len, d_in)。
        """
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z


class Kronos(nn.Module, PyTorchModelHubMixin):
    """
    Kronos模型：基于Transformer架构的时间序列预测模型

    该模型采用分层token表示和依赖感知机制，专门用于金融时间序列预测。
    支持s1和s2 tokens的联合建模，通过依赖感知层实现条件预测。
    集成了时间嵌入、Transformer编码器和双头输出架构。

    参数:
        s1_bits (int): pre tokens的比特数。
        s2_bits (int): post tokens的比特数。
        n_layers (int): Transformer块的数量。
        d_model (int): 模型嵌入和隐藏状态的维度。
        n_heads (int): MultiheadAttention层中的注意力头数。
        ff_dim (int): Transformer块中前馈网络的维度。
        ffn_dropout_p (float): 前馈网络的dropout概率。
        attn_dropout_p (float): 注意力层的dropout概率。
        resid_dropout_p (float): 残差连接的dropout概率。
        token_dropout_p (float): token嵌入的dropout概率。
        learn_te (bool): 是否使用可学习的时间嵌入。
    """

    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, learn_te):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p

        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        """
        参数:
            s1_ids (torch.Tensor): s1 token ID的输入张量。形状: [batch_size, seq_len]
            s2_ids (torch.Tensor): s2 token ID的输入张量。形状: [batch_size, seq_len]
            stamp (torch.Tensor, 可选): 时间戳张量。形状: [batch_size, seq_len]。默认为None。
            padding_mask (torch.Tensor, 可选): 填充token的掩码。形状: [batch_size, seq_len]。默认为None。
            use_teacher_forcing (bool, 可选): 是否对s1解码使用teacher forcing。默认为False。
            s1_targets (torch.Tensor, 可选): teacher forcing的s1 token ID目标。形状: [batch_size, seq_len]。默认为None。

        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: s1 token预测的logits。形状: [batch_size, seq_len, s1_vocab_size]
                - s2_logits: 基于s1条件的s2 token预测的logits。形状: [batch_size, seq_len, s2_vocab_size]
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask) # 依赖感知层：基于s1嵌入的条件
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """
        仅解码s1 tokens。

        该方法执行前向传播以仅预测s1 tokens。它返回s1 logits和来自Transformer的上下文表示，
        可用于后续的s2解码。

        参数:
            s1_ids (torch.Tensor): s1 token ID的输入张量。形状: [batch_size, seq_len]
            s2_ids (torch.Tensor): s2 token ID的输入张量。形状: [batch_size, seq_len]
            stamp (torch.Tensor, 可选): 时间戳张量。形状: [batch_size, seq_len]。默认为None。
            padding_mask (torch.Tensor, 可选): 填充token的掩码。形状: [batch_size, seq_len]。默认为None。

        返回:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: s1 token预测的logits。形状: [batch_size, seq_len, s1_vocab_size]
                - context: Transformer的上下文表示。形状: [batch_size, seq_len, d_model]
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        """
        解码s2 tokens，基于上下文和s1 tokens进行条件化。

        该方法基于预计算的上下文表示（通常来自`decode_s1`）和s1 token ID解码s2 tokens。
        它使用依赖感知层和条件s2头来预测s2 tokens。

        参数:
            context (torch.Tensor): 来自transformer的上下文表示（decode_s1的输出）。
                                    形状: [batch_size, seq_len, d_model]
            s1_ids (torch.torch.Tensor): s1 token ID的输入张量。形状: [batch_size, seq_len]
            padding_mask (torch.Tensor, 可选): 填充token的掩码。形状: [batch_size, seq_len]。默认为None。

        返回:
            torch.Tensor: s2 logits。形状: [batch_size, seq_len, s2_vocab_size]
        """
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    参数:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # 安全检查
        # 移除所有概率小于top-k中最后一个token的token
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的token（保留概率为0的token）
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # 至少保留min_tokens_to_keep个token（设置为min_tokens_to_keep-1，因为我们在下面添加第一个）
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # 将索引向右移动以保留第一个超过阈值的token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 将排序后的张量分散到原始索引
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = top_k(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x


def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5, verbose=False, progress_callback=None):
    with torch.no_grad():
        batch_size = x.size(0)
        initial_seq_len = x.size(1)
        x = torch.clip(x, -clip, clip)

        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)

        def get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, pred_step):

            if current_seq_len <= max_context - pred_step:
                return torch.cat([x_stamp, y_stamp[:, :pred_step, :]], dim=1)
            else:
                start_idx = max_context - pred_step
                return torch.cat([x_stamp[:, -start_idx:, :], y_stamp[:, :pred_step, :]], dim=1)

        if verbose:
            ran = trange
        else:
            ran = range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i

            if current_seq_len <= max_context:
                input_tokens = x_token
            else:
                input_tokens = [t[:, -max_context:].contiguous() for t in x_token]

            current_stamp = get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, i)

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            x_token[0] = torch.cat([x_token[0], sample_pre], dim=1)
            x_token[1] = torch.cat([x_token[1], sample_post], dim=1)

            # 调用进度回调函数
            if progress_callback:
                progress = (i + 1) / pred_len * 100
                progress_callback(progress, f"正在预测第 {i+1}/{pred_len} 步...")

            torch.cuda.empty_cache()

        input_tokens = [t[:, -max_context:].contiguous() for t in x_token]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(batch_size, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)

        return preds


def calc_time_stamps(x_timestamp):
    time_df = pd.DataFrame()
    time_df['minute'] = x_timestamp.dt.minute
    time_df['hour'] = x_timestamp.dt.hour
    time_df['weekday'] = x_timestamp.dt.weekday
    time_df['day'] = x_timestamp.dt.day
    time_df['month'] = x_timestamp.dt.month
    return time_df


class KronosPredictor:
    """
    KronosPredictor类：Kronos模型的预测接口封装

    该类提供了对Kronos模型的便捷预测接口，支持单序列和批量预测。
    自动处理数据预处理、标准化、时间戳转换和结果后处理。
    专门用于金融时间序列的预测任务，支持OHLCV数据和多种采样策略。

    参数:
        model: 已训练的Kronos模型实例
        tokenizer: 对应的KronosTokenizer实例
        device (str): 计算设备，默认为"cuda:0"
        max_context (int): 最大上下文长度，默认为512
        clip (float): 数据裁剪阈值，默认为5
    """

    def __init__(self, model, tokenizer, device="cuda:0", max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_vol = 'amount'
        self.time_cols = ['minute', 'hour', 'weekday', 'day', 'month']
        self.device = device

        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose, progress_callback=None):

        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        preds = auto_regressive_inference(self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor, self.max_context, pred_len,
                                          self.clip, T, top_k, top_p, sample_count, verbose, progress_callback)
        preds = preds[:, -pred_len:, :]
        return preds

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True, progress_callback=None):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0  # 用零填充缺失的成交量
            df[self.amt_vol] = 0.0  # 用零填充缺失的成交额
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        # 如果有进度回调函数，修改generate方法以支持进度反馈
        if progress_callback:
            preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose, progress_callback)
        else:
            preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)

        preds = preds.squeeze(0)
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(preds, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp)
        return pred_df


    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        """
        对多个时间序列执行并行（批量）预测。所有序列必须具有相同的历史长度和预测长度（pred_len）。

        参数:
            df_list (List[pd.DataFrame]): 输入DataFrame列表，每个包含价格列和可选的成交量/成交额列。
            x_timestamp_list (List[pd.DatetimeIndex or Series]): 对应历史数据的时间戳列表，长度应与每个DataFrame的行数匹配。
            y_timestamp_list (List[pd.DatetimeIndex or Series]): 未来预测时间戳列表，长度应等于pred_len。
            pred_len (int): 预测步数。
            T (float): 采样温度。
            top_k (int): Top-k过滤阈值。
            top_p (float): Top-p（核心采样）阈值。
            sample_count (int): 每个序列的并行样本数，内部自动平均。
            verbose (bool): 是否显示自回归进度。

        返回:
            List[pd.DataFrame]: 预测结果列表，顺序与输入相同，每个DataFrame包含
                                `open, high, low, close, volume, amount`列，索引为对应的`y_timestamp`。
        """
        # 基本验证
        if not isinstance(df_list, (list, tuple)) or not isinstance(x_timestamp_list, (list, tuple)) or not isinstance(y_timestamp_list, (list, tuple)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        num_series = len(df_list)

        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []
        y_lens = []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(f"DataFrame at index {i} is missing price columns {self.price_cols}.")

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
                raise ValueError(f"DataFrame at index {i} contains NaN values in price or volume columns.")

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}.")

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)

            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        # 要求所有序列具有一致的历史长度和预测长度以进行批量处理
        if len(set(seq_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent prediction lengths, got: {y_lens}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)           # (B, seq_len, feat)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32) # (B, seq_len, time_feat)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32) # (B, pred_len, time_feat)

        preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose)
        # preds: (B, pred_len, feat)

        pred_dfs = []
        for i in range(num_series):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(preds_i, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp_list[i])
            pred_dfs.append(pred_df)

        return pred_dfs

