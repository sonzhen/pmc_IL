## from https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


def make_batch(sentences):
    """
    把 3 个句子转换成训练时要用的 3 份张量。

    sentences[0]: 编码器输入，例如 "ich mochte ein bier P"
    sentences[1]: 解码器输入，例如 "S i want a beer"
    sentences[2]: 监督目标，例如 "i want a beer E"

    返回:
        enc_inputs:  [batch_size, src_len]
        dec_inputs:  [batch_size, tgt_len]
        target_batch:[batch_size, tgt_len]

    这里的 batch_size=1，因为示例里只放了一组训练样本。
    """
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


## 10. Decoder 自注意力中的 subsequent mask
def get_attn_subsequent_mask(seq):
    """
    生成上三角 mask，用于屏蔽当前位置之后的词。

    为什么要这样做:
    Decoder 在训练时虽然一次性看到了整句 dec_inputs，但预测第 t 个词时，
    理论上不能偷看第 t+1、t+2 ... 这些未来位置，所以要把未来位置 mask 掉。

    参数:
        seq: [batch_size, tgt_len]

    返回:
        subsequence_mask: [batch_size, tgt_len, tgt_len]
        其中为 1 的位置表示“需要被 mask”
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


## 7. Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        参数形状:
            Q: [batch_size, n_heads, len_q, d_k]
            K: [batch_size, n_heads, len_k, d_k]
            V: [batch_size, n_heads, len_k, d_v]
            attn_mask: [batch_size, n_heads, len_q, len_k]

        核心步骤:
        1. 先计算 Q 和 K^T 的相似度分数
        2. 再除以 sqrt(d_k) 做缩放，避免数值过大导致 softmax 梯度过小
        3. 用 mask 把不该看的位置置成极小值
        4. 对最后一维做 softmax，得到注意力权重
        5. 用注意力权重对 V 加权求和，得到上下文表示
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # mask 为 1 的位置会被填成极小值，softmax 后这些位置的权重近似为 0
        scores.masked_fill_(attn_mask, -1e9)

        # attn 是注意力权重，不是最终输出
        attn = nn.Softmax(dim=-1)(scores)

        # context 是加权求和后的上下文向量
        context = torch.matmul(attn, V)
        return context, attn


## 6. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        # 把输入映射成多头注意力需要的 Q、K、V
        # 每个头看到的是 d_k / d_v 维，拼起来后总维度仍然对应 n_heads * d_k(d_v)
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        # 多头拼接后再映射回 d_model
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        """
        参数:
            Q: [batch_size, len_q, d_model]
            K: [batch_size, len_k, d_model]
            V: [batch_size, len_k, d_model]
            attn_mask: [batch_size, len_q, len_k]

        返回:
            output: [batch_size, len_q, d_model]
            attn:   [batch_size, n_heads, len_q, len_k]

        这里的流程可以记成:
        映射 -> 分头 -> 每个头各自做 attention -> 拼接 -> 线性映射 -> 残差连接 + LayerNorm
        """
        residual, batch_size = Q, Q.size(0)

        # (B, S, D) -> (B, S, H * W) -> (B, S, H, W) -> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 原始 mask 是 [B, len_q, len_k]
        # 多头之后要复制到每个 head，变成 [B, H, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # 把多头结果拼回来
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)

        # 残差连接 + LayerNorm
        return self.layer_norm(output + residual), attn


## 8. Position-wise Feed Forward Network
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()

        # 这里用 1x1 卷积实现逐位置的两层前馈网络。
        # 因为 kernel_size=1，所以不会混合相邻 token，只会在特征维度上做变换。
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        inputs: [batch_size, len_q, d_model]
        """
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


## 4. padding mask
def get_attn_pad_mask(seq_q, seq_k):
    """
    根据 seq_k 中的 PAD 位置生成 attention mask。

    参数:
        seq_q: [batch_size, len_q]
        seq_k: [batch_size, len_k]

    返回:
        pad_attn_mask: [batch_size, len_q, len_k]

    重点理解:
    1. attention 分数矩阵的形状是 [len_q, len_k]
       所以 mask 也要扩展成同样的二维结构。
    2. 我们 mask 的是 K 侧的 PAD 位置，因为 Q 去“看”K 时，
       不应该把注意力分给那些只是补齐出来的无效 token。
    3. 这份代码约定 PAD token 的 id 必须是 0，
       因为下面直接用 eq(0) 来判断哪些位置是 PAD。
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # seq_k 中值为 0 的位置就是 PAD，1 表示这些位置需要被 mask
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


## 3. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        """
        Transformer 本身没有循环结构和卷积结构，
        所以它不知道“第几个 token 在前，第几个 token 在后”，
        需要显式加入位置编码。

        这里使用论文中的固定正弦/余弦位置编码:
            PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

        直觉上:
        1. 不同维度用不同频率的正弦/余弦
        2. 这样每个位置都会对应一个独特的编码
        3. 模型也更容易学习相对位置信息
        """
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数维用 sin，奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 变成 [max_len, 1, d_model]，便于和输入 [seq_len, batch_size, d_model] 相加
        pe = pe.unsqueeze(0).transpose(0, 1)
        #等价： pe = pe.unsqueeze(1)

        # register_buffer 表示这个张量属于模型的一部分，
        # 会随着 model.to(device) 一起移动，但不会被优化器更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


## 5. Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        enc_inputs:         [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]

        Encoder 的 self-attention 中:
            Q = K = V = enc_inputs
        因为它是在“源句子内部”做自注意力。
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


## 2. Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # 词嵌入层: [vocab_size] -> [d_model]
        self.src_emb = nn.Embedding(src_vocab_size, d_model)

        # 位置编码层
        self.pos_emb = PositionalEncoding(d_model)

        # 堆叠多个 EncoderLayer
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]

        返回:
            enc_outputs:    [batch_size, src_len, d_model]
            enc_self_attns: 长度为 n_layers 的列表，
                            每个元素形状为 [batch_size, n_heads, src_len, src_len]
        """
        # token id -> embedding
        enc_outputs = self.src_emb(enc_inputs)

        # 加入位置编码
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        # 编码器自注意力的 padding mask
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


## 10. Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        第一步: Decoder 内部先做 masked self-attention
        第二步: 再拿 decoder 当前表示去和 encoder 输出做 cross-attention
        第三步: 经过逐位置前馈网络
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


## 9. Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        参数:
            dec_inputs:  [batch_size, tgt_len]
            enc_inputs:  [batch_size, src_len]
            enc_outputs: [batch_size, src_len, d_model]

        返回:
            dec_outputs:   [batch_size, tgt_len, d_model]
            dec_self_attns:长度为 n_layers 的列表
            dec_enc_attns: 长度为 n_layers 的列表
        """
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)

        # Decoder 自注意力里的 pad mask
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        # Decoder 自注意力里的未来信息 mask
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        # 两类 mask 合并:
        # 只要某个位置属于 PAD 或者属于未来位置，就不能被看到
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        # 交叉注意力的 mask:
        # Q 来自 decoder，K/V 来自 encoder，所以只需要关心 encoder 端哪些位置是 PAD
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask
            )
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


## 1. Transformer 总体结构
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        # 编码器: 负责把源句子编码成上下文化表示
        self.encoder = Encoder()

        # 解码器: 负责结合已生成目标序列和编码器输出，得到当前步的隐藏表示
        self.decoder = Decoder()

        # 输出层: 把 decoder 每个位置的 d_model 维表示映射到目标词表大小
        # 之后通常会在 loss 内部隐式做 softmax，例如 CrossEntropyLoss
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        """
        参数:
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]

        返回:
            dec_logits.view(-1, dec_logits.size(-1)):
                [batch_size * tgt_len, tgt_vocab_size]
                这样可以和 target_batch.view(-1) 对齐，直接送入 CrossEntropyLoss

            enc_self_attns:
                每一层 encoder 的自注意力权重
            dec_self_attns:
                每一层 decoder 的自注意力权重
            dec_enc_attns:
                每一层 decoder 对 encoder 的交叉注意力权重
        """

        # Encoder 输出源序列的上下文表示
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # Decoder 基于目标端输入和 encoder 输出继续建模
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


if __name__ == '__main__':

    # 这里构造一个极小的德语 -> 英语翻译示例
    # P: Padding，占位补齐
    # S: Start，解码器起始符
    # E: End，目标句结束符
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer Parameters
    # 这份代码约定 PAD 的 token id 必须是 0，因为上面的 get_attn_pad_mask 直接用 eq(0) 判断 PAD
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    # 源序列长度和目标序列长度在这个 toy example 中都固定为 5
    src_len = 5
    tgt_len = 5

    # 模型超参数
    d_model = 512   # token embedding / hidden state 的维度
    d_ff = 2048     # 前馈网络隐藏层维度
    d_k = d_v = 64  # 每个 head 中 Q/K/V 的维度
    n_layers = 6    # EncoderLayer / DecoderLayer 堆叠层数
    n_heads = 8     # 多头注意力头数

    model = Transformer()

    # CrossEntropyLoss 内部会先做 log_softmax，再和标签计算交叉熵
    criterion = nn.CrossEntropyLoss()

    # Adam 是一种自适应优化器，lr=0.001 是教学示例里常用的默认起点
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(20):
        # 清空上一轮累积的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        # target_batch.view(-1): [batch_size * tgt_len]
        loss = criterion(outputs, target_batch.contiguous().view(-1))

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        # 反向传播 + 参数更新
        loss.backward()
        optimizer.step()

