from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass
class TextDataset:
    """
    字符级数据集容器。

    - text: 清洗后的原始文本
    - data: 编码后的 token 序列（1D LongTensor）
    - vocab_size: 词表大小
    - stoi/encode: 字符 -> id
    - decode: id 列表 -> 字符串
    """

    text: str
    data: torch.Tensor
    vocab_size: int
    stoi: dict[str, int]
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]


def clean_text(text: str) -> str:
    """
    最小清洗策略：
    1) 去掉空行
    2) 去掉每行首尾空白
    3) 用换行符重新拼接

    这样做的目的：
    - 保留基本段落结构
    - 避免大量空白字符污染字符词表
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def load_corpus(corpus_path: str) -> str:
    """
    从磁盘读取语料并清洗，确保返回非空文本。
    """
    raw = Path(corpus_path).read_text(encoding="utf-8")
    text = clean_text(raw)
    if not text:
        raise ValueError(f"语料文件为空: {corpus_path}")
    return text


def build_text_dataset(text: str) -> TextDataset:
    """
    把文本构造成字符级数据集。

    核心步骤：
    - chars: 去重并排序后的字符集合（稳定映射）
    - stoi/itos: 双向映射
    - data: 把整段文本编码成一维 token 序列
    """
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    return TextDataset(
        text=text,
        data=data,
        vocab_size=len(chars),
        stoi=stoi,
        encode=encode,
        decode=decode,
    )


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    随机采样训练 batch（next-token prediction）。

    假设某个起点是 i：
    - x = data[i : i + block_size]
    - y = data[i + 1 : i + block_size + 1]

    所以 y 总是 x 的“右移一位”，这正是语言模型训练目标。
    """
    if len(data) <= block_size:
        raise ValueError(
            f"数据长度({len(data)})必须大于 block_size({block_size})，请减小 block_size 或增大语料。"
        )
    # 每个样本随机选一个起点，提升训练样本多样性。
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    # 放到目标设备，避免训练循环里重复搬运。
    return x.to(device), y.to(device)
