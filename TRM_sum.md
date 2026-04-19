# TRM.py 学习笔记

这份笔记围绕 `TRM.py` 这份 Transformer 教学代码展开，目标不是覆盖所有论文细节，而是把这份小 demo 里最重要的概念讲清楚：

- 这份代码到底在做什么
- 数据和张量是怎么流动的
- `Q / K / V`、自注意力、交叉注意力分别是什么
- `PAD mask`、`subsequent mask` 为什么这样写
- `unsqueeze / squeeze / view / transpose` 这些张量操作到底在干什么
- 这个 demo 能学到什么，不能学到什么

## 1. 这份 demo 在做什么

这是一份极小型的 Transformer 翻译示例。  
它用 1 条德语到英语的样本，演示完整的训练流程：

```python
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
```

其中：

- `P`：padding，占位补齐
- `S`：start，解码器起始符
- `E`：end，目标句结束符

这份代码的主流程是：

1. 源句子进入 `Encoder`
2. 目标端输入进入 `Decoder`
3. `Decoder` 一边看目标端已知上下文，一边看 `Encoder` 输出
4. 最后映射到目标词表大小，计算 loss

它的主要用途是教学，不是训练可用翻译器。

## 2. Transformer 的三部分结构

在这份代码里，`Transformer` 可以看成三部分：

1. `Encoder`
2. `Decoder`
3. `Projection`

对应代码是：

```python
self.encoder = Encoder()
self.decoder = Decoder()
self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
```

含义分别是：

- `Encoder`：把源句子编码成上下文化表示
- `Decoder`：结合目标端输入和编码器输出，生成当前目标位置的隐藏表示
- `Projection`：把隐藏表示映射到目标词表大小

## 3. 这份代码的 batch size 和训练方式

### 3.1 batch size 是多少

这份代码里 `batch_size = 1`。

因为 `make_batch` 里外面只包了一层列表：

```python
input_batch = [[src_vocab[n] for n in sentences[0].split()]]
output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
```

所以生成出来的张量形状是：

- `enc_inputs.shape = [1, 5]`
- `dec_inputs.shape = [1, 5]`
- `target_batch.shape = [1, 5]`

### 3.2 `epoch=20` 表示什么

`epoch=20` 表示把同一份训练数据反复训练 20 轮。

而这份代码里训练数据实际上只有 1 条样本，所以每一轮看到的都是同一句：

- 源句：`ich mochte ein bier P`
- decoder 输入：`S i want a beer`
- 目标输出：`i want a beer E`

也就是说，这个 demo 本质上是在让模型反复“背这一题”。

### 3.3 为什么同一句还要训练 20 次

因为模型参数一开始是随机初始化的，不会一遍就学会。训练需要反复迭代：

1. 前向传播得到预测
2. 计算预测和正确答案的误差
3. 反向传播算梯度
4. 根据梯度更新参数
5. 重复很多轮，模型才会逐渐记住这条样本

### 3.4 这个模型能翻译别的句子吗

基本不能。

原因有三个：

- 它只见过 1 条训练样本，几乎没有泛化能力
- 词表非常小，只包含 `ich / mochte / ein / bier / i / want / a / beer` 等少量词
- 它学到的更像是句子级记忆，而不是完整的翻译规律

所以它最多只能学会：

```text
ich mochte ein bier -> i want a beer
```

像“苹果”这种新词没有见过，也没有在词表里，这个模型就处理不了。

## 4. 关键维度都表示什么

这份代码里最重要的几个维度是：

### `batch_size`

- 一次送进模型多少条样本
- 这里是 `1`

### `src_len`

- 源句长度
- 这里是 `5`

对应句子：

```python
ich mochte ein bier P
```

### `tgt_len`

- 目标端输入长度
- 这里也是 `5`

对应句子：

```python
S i want a beer
```

### `d_model`

- 模型内部每个 token 表示向量的维度
- 这里是 `512`

也就是说，一个 token id 经过 embedding 后，会变成一个 512 维向量。

## 5. 整体数据流和张量形状

### 5.1 输入

源句输入：

```python
enc_inputs: [1, 5]
```

目标端输入：

```python
dec_inputs: [1, 5]
```

### 5.2 embedding 后

编码器：

```python
enc_outputs = self.src_emb(enc_inputs)
```

形状变成：

```python
[1, 5, 512]
```

解码器：

```python
dec_outputs = self.tgt_emb(dec_inputs)
```

形状也变成：

```python
[1, 5, 512]
```

注意：`enc_inputs` 和 `dec_inputs` 并不是一起进入同一个 embedding，而是分别进入各自的 embedding：

- `enc_inputs -> src_emb`
- `dec_inputs -> tgt_emb`

### 5.3 最终输出

decoder 输出经过线性层：

```python
dec_logits = self.projection(dec_outputs)
```

形状变成：

```python
[1, 5, tgt_vocab_size]
```

这份代码里 `tgt_vocab_size = 7`，所以实际是：

```python
[1, 5, 7]
```

然后：

```python
dec_logits.view(-1, dec_logits.size(-1))
```

变成：

```python
[5, 7]
```

这就是为了和：

```python
target_batch.view(-1)   # [5]
```

对齐，送进 `CrossEntropyLoss`。

## 6. Q、K、V 到底是谁

可以先用一句最直白的话记住：

- `Q`：我现在想找什么信息
- `K`：每个位置“我是什么”的标签
- `V`：每个位置真正携带的信息

它们并不是固定代表某一种词，而是要看当前是哪一种 attention。

### 6.1 Encoder 自注意力

在 encoder 里：

- `Q` 来自源句表示
- `K` 来自源句表示
- `V` 来自源句表示

也就是：同一个序列内部自己看自己。

### 6.2 Decoder 自注意力

在 decoder 里：

- `Q` 来自目标端当前表示
- `K` 来自目标端当前表示
- `V` 来自目标端当前表示

也是“自己看自己”，但不能看未来位置。

### 6.3 Decoder 交叉注意力

在 decoder 的 cross-attention 里：

- `Q` 来自 decoder
- `K` 来自 encoder
- `V` 来自 encoder

也就是：目标端去看源句子。

## 7. 自注意力和交叉注意力的区别

### 自注意力

特点是：

- `Q / K / V` 都来自同一个序列

作用是：

- 建模序列内部词和词之间的关系

在这份代码里有两种：

- encoder 自注意力：源句自己看自己
- decoder 自注意力：目标句自己看自己，但不能看未来

### 交叉注意力

特点是：

- `Q` 来自一个序列
- `K / V` 来自另一个序列

作用是：

- 让 decoder 在生成目标词时参考源句信息

一句话区分：

- 自注意力：句内建模
- 交叉注意力：跨序列对齐

## 8. 用 `want` 这个位置理解 decoder 的两次注意力

假设 decoder 当前在处理 `want` 这个位置。

源句是：

```text
ich mochte ein bier P
```

目标端输入是：

```text
S i want a beer
```

### 8.1 Decoder 自注意力在做什么

此时 `want` 这个位置可以参考：

- `S`
- `i`
- `want`

但不能看：

- `a`
- `beer`

因为未来位置会被 `subsequent mask` 屏蔽。

所以它解决的是：

- 当前目标位置在目标句内部应该如何理解自己

### 8.2 Decoder 交叉注意力在做什么

然后 decoder 会拿当前这个位置去看 encoder 输出。

比如 `want` 这个位置，可能会更多关注源句中的：

- `mochte`

因为它和 “want” 的语义更接近。

所以 cross-attention 解决的是：

- 当前目标词应该从源句哪里取信息

## 9. 为什么 PAD 要设成 0

这句注释：

```python
# Padding Should be Zero
```

本质意思是：

- `PAD` token 的 id 要设成 `0`

原因在这句代码：

```python
pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
```

这里直接用 `eq(0)` 判断哪些位置是 PAD。  
所以如果词表里：

```python
src_vocab = {'P': 0, ...}
tgt_vocab = {'P': 0, ...}
```

模型就能正确找到 padding 位置并 mask 掉。

如果 `PAD` 不是 `0`，这句判断就会失效。

要注意：

- 这里的 “zero” 指的是 `PAD token id = 0`
- 不代表它的 embedding 向量一定自动是全 0

如果想让 padding 的 embedding 也固定为 0，更规范的写法通常是：

```python
nn.Embedding(vocab_size, d_model, padding_idx=0)
```

## 10. 两种 mask 分别在做什么

### 10.1 PAD mask

作用是：

- 屏蔽掉序列中只是补齐出来的无效位置

对应代码：

```python
pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
```

### 10.2 Subsequent mask

作用是：

- 屏蔽掉当前位置之后的未来词

对应代码：

```python
subsequence_mask = np.triu(np.ones(attn_shape), k=1)
```

上三角为 1，表示“未来位置不能看”。

### 10.3 为什么 mask 后 softmax 基本等于 0

在算注意力前，代码会做：

```python
scores.masked_fill_(attn_mask, -1e9)
```

意思是：

- mask 为真的位置，分数直接改成一个极小值

softmax 后，这些位置的权重就会非常接近 0，所以基本等于“看不到”。

## 11. `pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)` 怎么拆开理解

这句代码可以拆成三步：

```python
pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
```

### 11.1 `.eq(0)`

`eq` 是 `equal` 的缩写，表示逐元素判断是否等于 0。

例如：

```python
seq_k = tensor([[1, 2, 3, 0, 0]])
seq_k.eq(0)
```

得到：

```python
tensor([[False, False, False, True, True]])
```

也就是把 PAD 位置找出来。

### 11.2 `.unsqueeze(1)`

原来形状是：

```python
[batch_size, len_k]
```

加一维后变成：

```python
[batch_size, 1, len_k]
```

这样后面才能扩展成 attention 需要的：

```python
[batch_size, len_q, len_k]
```

### 11.3 `.data` 为什么不推荐

这份教程代码里用了 `.data`，现代 PyTorch 通常更推荐直接写：

```python
seq_k.eq(0)
```

## 12. 注意力计算的核心公式

注意力的核心公式是：

```python
softmax(QK^T / sqrt(d_k))V
```

它可以拆成四步理解：

1. `QK^T`：先算“query 和每个 key 有多像”
2. `/ sqrt(d_k)`：做缩放，防止数值过大
3. `softmax`：把相似度变成权重分布
4. `@ V`：按这些权重把信息取回来

### 12.1 `QK^T` 为什么会得到 `[1, 8, 5, 5]`

在多头拆分后：

```python
Q: [1, 8, 5, 64]
K: [1, 8, 5, 64]
```

为了做矩阵乘法，要先：

```python
K.transpose(-1, -2)
```

这样最后两维从：

```python
[5, 64]
```

变成：

```python
[64, 5]
```

于是：

```python
[5, 64] @ [64, 5] -> [5, 5]
```

再加上 batch 和 head 两维，总体就是：

```python
[1, 8, 5, 5]
```

这个矩阵表示：

- 每个 query 位置
- 对每个 key 位置
- 有多相关

### 12.2 为什么要除以 `sqrt(d_k)`

因为当 `d_k` 变大时，点积值容易变得很大。  
如果数值太大，softmax 会变得过于尖锐，训练会不稳定。

所以要除以：

```python
sqrt(d_k)
```

在这份代码里：

```python
d_k = 64
sqrt(d_k) = 8
```

### 12.3 softmax 后得到的 `attn` 是什么

softmax 后：

```python
attn: [1, 8, 5, 5]
```

它不再是原始分数，而是每个 query 对所有 key 的权重分布。

也就是说：

- `attn` 决定“看谁”

## 13. 为什么 `attn: [1, 8, 5, 5]` 最后又回到 `[1, 5, 512]`

这是多头注意力里最关键的一步。

先看 `V` 的形状：

```python
V: [1, 8, 5, 64]
```

然后做：

```python
context = torch.matmul(attn, V)
```

形状变化是：

```python
[1, 8, 5, 5] @ [1, 8, 5, 64]
-> [1, 8, 5, 64]
```

这一步表示：

- 对每个 head、每个 query 位置
- 用 5 个注意力权重
- 对 5 个 value 向量做加权求和

接着：

```python
context.transpose(1, 2)
```

把形状从：

```python
[1, 8, 5, 64]
```

变成：

```python
[1, 5, 8, 64]
```

再执行：

```python
.contiguous().view(batch_size, -1, n_heads * d_v)
```

因为：

```python
n_heads = 8
d_v = 64
```

所以：

```python
8 * 64 = 512
```

于是：

```python
[1, 5, 8, 64] -> [1, 5, 512]
```

这一步的本质是：

- 把同一个位置上的 8 个 head 拼接起来

最后再过一层：

```python
nn.Linear(512, 512)
```

形状仍然保持：

```python
[1, 5, 512]
```

一句话总结：

- `attn` 决定看谁
- `V` 提供信息
- `attn @ V` 把信息聚合起来
- 多个 head 聚合完后再拼接回 `d_model`

## 14. 优化器和学习率

### 14.1 Adam 是什么

在代码里：

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

`Adam` 是优化器，用来根据梯度更新模型参数。

训练循环中三句最关键的代码是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

分别表示：

- 清空旧梯度
- 反向传播算新梯度
- 用优化器更新参数

### 14.2 为什么这里常用 `lr=0.001`

因为 `0.001` 是 Adam 的经典默认学习率，通常是一个比较稳妥的起点：

- 不至于太大而发散
- 也不至于太小而完全学不动

但这只是教学 demo 的常见默认值，不是所有 Transformer 训练的标准答案。

### 14.3 SGD / Momentum / RMSProp / Adam 的区别

`SGD`

- 最基础的梯度下降
- 只看当前梯度

`Momentum`

- 在 SGD 上加入惯性
- 更新更平滑，不容易来回抖动

`RMSProp`

- 根据梯度历史给不同参数分配不同步长

`Adam`

- 可以理解成 `Momentum + RMSProp`
- 既考虑惯性，又做自适应步长

### 14.4 这些名字的命名背景

`SGD = Stochastic Gradient Descent`

- `Stochastic`：随机近似
- `Gradient Descent`：梯度下降

`Momentum`

- 来自物理中的“动量”

`RMSProp = Root Mean Square Propagation`

- `Root Mean Square`：均方根

`Adam = Adaptive Moment Estimation`

- `Adaptive`：自适应
- `Moment`：统计学里的矩
- `Estimation`：估计

## 15. 常见张量操作

这些操作在 Transformer 代码里非常常见。

### 15.1 `unsqueeze`

作用：

- 增加一个大小为 `1` 的维度

例如：

```python
x = torch.tensor([1, 2, 3])   # [3]
x.unsqueeze(0)                # [1, 3]
x.unsqueeze(1)                # [3, 1]
```

如果原张量有 `n` 维，那么合法正数范围是：

```python
0 <= dim <= n
```

所以对 1 维张量：

```python
x.unsqueeze(2)
```

会报错，因为维度超界。

负数维度表示从后往前数位置，例如：

```python
x.unsqueeze(-1)
x.unsqueeze(-2)
```

### 15.2 `squeeze`

作用：

- 删除大小为 `1` 的维度

例如：

```python
x = torch.randn(1, 3, 1, 5)
x.squeeze()      # [3, 5]
x.squeeze(0)     # [3, 1, 5]
x.squeeze(2)     # [1, 3, 5]
```

如果指定的维度大小不是 `1`，那么不会删掉。

### 15.3 `view`

作用：

- 改整体形状
- 元素总数不能变

例如：

```python
x = torch.arange(6)
x.view(2, 3)
x.view(3, 2)
```

### 15.4 `transpose`

作用：

- 交换两个维度

例如：

```python
x = torch.randn(2, 3)
x.transpose(0, 1)   # [3, 2]
```

### 15.5 `permute`

作用：

- 按指定顺序重排多个维度

例如：

```python
x = torch.randn(2, 3, 4)
x.permute(2, 0, 1)   # [4, 2, 3]
```

一句话记忆：

- `unsqueeze / squeeze`：加壳 / 脱壳
- `view`：改盒子形状
- `transpose / permute`：换维度顺序

## 16. 这些张量操作在 TRM.py 里是怎么用的

### `unsqueeze(1)`

例如：

```python
seq_k.eq(0).unsqueeze(1)
```

把：

```python
[batch_size, len_k]
```

变成：

```python
[batch_size, 1, len_k]
```

### `view(batch_size, -1, n_heads, d_k)`

例如：

```python
self.W_Q(Q).view(batch_size, -1, n_heads, d_k)
```

把：

```python
[batch_size, len_q, d_model]
```

变成：

```python
[batch_size, len_q, n_heads, d_k]
```

也就是把最后一维拆成多头结构。

### `transpose(1, 2)`

例如：

```python
.transpose(1, 2)
```

把：

```python
[batch_size, len_q, n_heads, d_k]
```

变成：

```python
[batch_size, n_heads, len_q, d_k]
```

### `contiguous().view(...)`

例如：

```python
context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
```

意思是：

1. 先换维度顺序
2. 再把多个 head 拼回去
3. 得到 `[batch_size, len_q, n_heads * d_v]`

其中 `contiguous()` 可以粗略理解为：

- 让内存布局重新连续
- 这样后面的 `view(...)` 才能安全使用

## 17. 最后用几句话总结全篇

这份 `TRM.py` 是一个教学 demo，重点不是训练出可用翻译器，而是帮助理解 Transformer 的基本结构和数据流。

看这份代码时，最值得真正吃透的是这几件事：

- `Encoder / Decoder / Projection` 各做什么
- `Q / K / V` 在不同 attention 里分别来自哪里
- 自注意力和交叉注意力的区别
- `QK^T / sqrt(d_k)`、softmax、`@ V` 是如何串起来的
- mask 为什么能屏蔽 PAD 和未来词
- 多头结果为什么能重新拼回 `[batch_size, seq_len, d_model]`

如果这些已经看顺了，这份 demo 的核心价值你就基本拿到了。

## 18. 模型里哪些是可学习参数，哪些不是

很多初学者会先注意到 embedding 的 512 维向量，然后觉得“模型是不是主要就在学这个”。  
这个理解抓住了一部分重点，但还不完整。

更准确地说：

- embedding 参数确实会学习
- 但 Transformer 优化的不只是 embedding
- 而是整个网络里所有可学习参数

## 18.1 embedding 的 512 维为什么是参数

例如一个 token id 经过 embedding 后，会取出 embedding table 中对应的一行：

```python
id -> embedding_table[id] -> 512维向量
```

这 512 个数就是可学习参数。

训练开始时它们通常会被随机初始化，之后在反向传播和优化器更新中不断调整。

但是要注意：

- 模型优化的不是只有 embedding
- embedding 只是整个模型参数的一部分

## 18.2 可学习参数有哪些

在这份 `TRM.py` 里，主要可学习参数包括下面几类。

### 1. Embedding 层

```python
self.src_emb = nn.Embedding(src_vocab_size, d_model)
self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
```

可学习的是：

- `src_emb.weight`
- `tgt_emb.weight`

形状分别是：

- `[src_vocab_size, d_model]`
- `[tgt_vocab_size, d_model]`

也就是词表里每个 token 都对应一行可学习向量。

### 2. Multi-Head Attention 里的线性层

```python
self.W_Q = nn.Linear(d_model, d_k * n_heads)
self.W_K = nn.Linear(d_model, d_k * n_heads)
self.W_V = nn.Linear(d_model, d_v * n_heads)
self.linear = nn.Linear(n_heads * d_v, d_model)
```

这些层里的参数都会学习，包括：

- `W_Q.weight`
- `W_Q.bias`
- `W_K.weight`
- `W_K.bias`
- `W_V.weight`
- `W_V.bias`
- `linear.weight`
- `linear.bias`

它们负责把输入映射成 Q/K/V，以及把多头拼接后的结果映射回 `d_model`。

### 3. 前馈网络参数

```python
self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
```

可学习的是：

- `conv1.weight`
- `conv1.bias`
- `conv2.weight`
- `conv2.bias`

虽然代码里写的是 `Conv1d`，但因为 `kernel_size=1`，可以把它理解成逐位置的线性变换。

### 4. LayerNorm 参数

```python
self.layer_norm = nn.LayerNorm(d_model)
```

`LayerNorm` 默认也有可学习参数，通常是：

- `layer_norm.weight`
- `layer_norm.bias`

可以粗略理解成缩放和平移参数。

### 5. 最后的输出层参数

```python
self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
```

可学习的是：

- `projection.weight`

它负责把 decoder 输出映射到目标词表大小。

## 18.3 不可学习的内容有哪些

下面这些会参与计算，但不会被优化器更新。

### 1. 固定位置编码

在代码里：

```python
self.register_buffer('pe', pe)
```

这说明 `pe` 是 buffer，不是 parameter。

含义是：

- 它属于模型的一部分
- 会随着模型一起移动到设备上
- 但不会被优化器更新

所以这份代码里的位置编码是固定的。

### 2. 各种 mask

例如：

- `pad_attn_mask`
- `subsequent_mask`
- `dec_self_attn_mask`

这些都是根据输入动态计算出来的辅助张量，不是可学习参数。

### 3. 超参数

例如：

- `d_model`
- `d_ff`
- `d_k`
- `n_heads`
- `n_layers`

这些只是模型配置，不会训练。

### 4. 输入数据本身

例如：

- `enc_inputs`
- `dec_inputs`
- `target_batch`

它们是训练数据，不是参数。

## 18.4 一次训练里哪些参数会被更新

原则上：

- 只要属于 `model.parameters()`
- 并且参与了这次前向传播
- 就会在反向传播后得到梯度，并被优化器更新

在这份代码里：

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

所以优化器接收到的是整个模型所有可学习参数。

## 18.5 为什么 embedding 往往只更新“用到的那些行”

embedding 比较特殊。

比如当前输入句子里只出现了：

- `ich`
- `mochte`
- `ein`
- `bier`
- `P`

那么 `src_emb.weight` 中，主要是这些 token 对应的行会参与这次查表和前向传播，因此它们会更直接地收到梯度更新。

没出现在当前输入里的那些 token 行，通常这一轮不会更新。

但像 `W_Q / W_K / W_V` 这类共享权重矩阵，只要这一层被使用了，通常整层参数都会更新。

## 18.6 初始化一定是“均值 0、方差 1”吗

不一定。

更准确的说法是：

- 参数通常会被随机初始化
- 但具体分布依赖层类型和框架默认实现

例如：

- `Embedding`
- `Linear`
- `Conv1d`

这些层常常使用不同的初始化策略。  
所以不能简单地理解成“所有参数都严格按均值 0、方差 1 初始化”。

## 18.7 一句话总结这一节

embedding 的 512 维向量确实是可学习参数，但模型优化的不只是 embedding，而是整个 Transformer 中所有可学习参数；这些参数先随机初始化，再通过前向传播、反向传播和优化器更新逐步收敛。

## 19. 按模块看：每一层学什么

下面这张表把 `TRM.py` 里的核心模块按“输入 / 输出 / 是否可学习 / 学的是什么”整理在一起。

| 模块 | 输入 | 输出 | 是否可学习 | 学的是什么 |
| --- | --- | --- | --- | --- |
| `src_emb / tgt_emb` | token id | token 向量 | 是 | 每个 token 的连续表示 |
| `PositionalEncoding` | token 向量 | 加位置后的向量 | 否 | 这份代码里是固定正余弦位置编码 |
| `W_Q / W_K / W_V` | hidden state | Q/K/V | 是 | 如何把当前表示映射成注意力查询、键和值 |
| `ScaledDotProductAttention` | Q/K/V + mask | context + attn | 否 | 这是计算规则，不是可学习层 |
| `linear`（多头输出） | 拼接后的多头结果 | 回到 `d_model` | 是 | 如何融合多个 head 的信息 |
| `conv1 / conv2` | 每个位置的 hidden state | 新的 hidden state | 是 | 逐位置前馈变换 |
| `LayerNorm` | hidden state | 归一化后的 hidden state | 是 | 缩放和平移参数 |
| `projection` | decoder 输出 | 词表打分 | 是 | 如何把 hidden state 映射到目标词表 |

## 19.1 一个最实用的理解方式

可以这样记：

- `Embedding` 学“词长什么样”
- `Q/K/V` 线性层学“怎么提问、怎么索引、怎么取值”
- 注意力公式本身不学习，它只是计算规则
- 前馈网络学“每个位置的进一步非线性变换”
- `Projection` 学“怎样从隐藏表示变成词表分数”
