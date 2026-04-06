from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """
    模型结构超参数。

    - embed_dim: token 向量维度（也叫 hidden size）
    - block_size: 单次看到的上下文长度（context window）
    - num_heads: 多头注意力头数
    - num_layers: Transformer Block 堆叠层数
    """

    embed_dim: int
    block_size: int
    num_heads: int
    num_layers: int


class Head(nn.Module):
    """
    单个因果自注意力头（causal self-attention）。

    一个 head 会学到一种“相关性模式”：
    例如有的 head 关注邻近字符，有的 head 关注更远的依赖。
    """

    def __init__(self, model_cfg: ModelConfig, head_size: int):
        super().__init__()
        # 将输入映射为 K/Q/V。这里不用 bias，保持实现简洁且常见。
        self.key = nn.Linear(model_cfg.embed_dim, head_size, bias=False)
        self.query = nn.Linear(model_cfg.embed_dim, head_size, bias=False)
        self.value = nn.Linear(model_cfg.embed_dim, head_size, bias=False)
        self.head_size = head_size
        # 下三角 mask：保证第 t 个位置不能看未来 token（语言模型关键约束）。
        # register_buffer 表示它不是可训练参数，但会随模型一起保存/迁移设备。
        self.register_buffer(
            "tril", torch.tril(torch.ones(model_cfg.block_size, model_cfg.block_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        返回: (B, T, head_size)
        """
        _, t, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        # 注意力分数: (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # 再乘 1/sqrt(head_size) 做缩放，避免 softmax 过于尖锐导致训练不稳定。
        attn = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        # 因果 mask：上三角位置置为 -inf，softmax 后概率为 0。
        attn = attn.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        v = self.value(x)
        # 加权求和得到输出: (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return attn @ v


class MultiHeadAttention(nn.Module):
    """
    多头注意力：
    1) 并行计算多个 head
    2) 在通道维拼接
    3) 线性投影回 embed_dim 做融合
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        # 每个头分到相同维度，要求 embed_dim 可整除 num_heads。
        if model_cfg.embed_dim % model_cfg.num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除。")
        head_size = model_cfg.embed_dim // model_cfg.num_heads
        self.heads = nn.ModuleList(
            [Head(model_cfg, head_size) for _ in range(model_cfg.num_heads)]
        )
        self.proj = nn.Linear(model_cfg.embed_dim, model_cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每个 head 输出 (B, T, head_size)，拼接后是 (B, T, embed_dim)。
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """
    Transformer 中的前馈网络（FFN）。

    经典做法是先升维到 4*embed_dim，再降回 embed_dim：
    - 升维: 提升表达能力
    - 非线性: 提供注意力之外的复杂变换能力
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_cfg.embed_dim, 4 * model_cfg.embed_dim),
            nn.ReLU(),
            nn.Linear(4 * model_cfg.embed_dim, model_cfg.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """
    一个完整 Transformer Block（Pre-LN 变体）：
    x = x + Attention(LN(x))
    x = x + FFN(LN(x))

    残差连接让信息与梯度更容易跨层传播。
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.attn = MultiHeadAttention(model_cfg)
        self.ffn = FeedForward(model_cfg)
        self.ln1 = nn.LayerNorm(model_cfg.embed_dim)
        self.ln2 = nn.LayerNorm(model_cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    """
    极简语言模型主干：
    token embedding + position embedding + N 层 Block + lm head
    """

    def __init__(self, model_cfg: ModelConfig, vocab_size: int, device: str):
        super().__init__()
        self.model_cfg = model_cfg
        self.device = device
        # 词嵌入：把 token id 映射到连续向量空间。
        self.token_embedding = nn.Embedding(vocab_size, model_cfg.embed_dim)
        # 位置嵌入：告诉模型“这个 token 在序列的第几个位置”。
        self.position_embedding = nn.Embedding(model_cfg.block_size, model_cfg.embed_dim)
        self.blocks = nn.Sequential(*[Block(model_cfg) for _ in range(model_cfg.num_layers)])
        self.ln_f = nn.LayerNorm(model_cfg.embed_dim)
        # 语言模型输出头：把隐藏状态投影回词表维度，得到每个 token 的 logits。
        self.head = nn.Linear(model_cfg.embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        x: (B, T) token ids
        targets: (B, T) 下一 token 监督信号（可选）
        返回:
          - logits: 若无 targets，形状 (B, T, vocab_size)
                    若有 targets，会被展平为 (B*T, vocab_size) 以计算 CE
          - loss: targets 存在时返回交叉熵，否则为 None
        """
        _, t = x.shape
        tok_emb = self.token_embedding(x)
        # 位置索引长度等于当前序列长度 t（而不是固定 block_size）。
        pos_emb = self.position_embedding(torch.arange(t, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        自回归生成：
        - 每轮只看最后 block_size 个 token
        - 取最后位置分布采样下一个 token
        - 再拼接回输入，进入下一轮
        """
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.model_cfg.block_size :]
            logits, _ = self(x_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)
        return x
