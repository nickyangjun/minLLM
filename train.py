import argparse
import configparser
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import torch

from data import build_text_dataset, get_batch, load_corpus
from model import MiniLLM, ModelConfig


@dataclass
class TrainConfig:
    """
    训练与推理配置。

    这部分字段直接对应 `config.ini`，便于“配置文件 -> Python 对象”一一映射。
    """

    embed_dim: int
    block_size: int
    num_heads: int
    num_layers: int
    batch_size: int
    learning_rate: float
    train_steps: int
    eval_interval: int
    device: str
    seed: int
    corpus_path: str
    prompt: str
    max_new_tokens: int
    checkpoint_path: str


def _default_config_dict() -> dict:
    """
    默认配置：
    - 当配置文件缺某个字段时，用这里的默认值补齐
    - 也作为类型推断依据（见 _coerce_value）
    """
    return {
        "embed_dim": 32,
        "block_size": 16,
        "num_heads": 2,
        "num_layers": 2,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "train_steps": 1000,
        "eval_interval": 200,
        "device": "cpu",
        "seed": 42,
        "corpus_path": "sample_corpus_zh.txt",
        "prompt": "你好",
        "max_new_tokens": 40,
        "checkpoint_path": "checkpoints/minllm.pt",
    }


def _coerce_value(key: str, value: str, defaults: dict) -> object:
    """
    把 INI 读取到的字符串，按默认值类型转为 int/float/str。
    """
    default_value = defaults[key]
    if isinstance(default_value, int):
        return int(value)
    if isinstance(default_value, float):
        return float(value)
    return value


def _load_ini_config(config_path: str) -> dict:
    """
    读取 INI 配置并返回“扁平字典”。

    说明：
    - INI 天然支持注释，适合学习项目频繁改参数
    - parser.items(section) 会把 key 转成小写
    """
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    parser.read(config_path, encoding="utf-8")
    defaults = _default_config_dict()
    allowed_keys = {f.name for f in fields(TrainConfig)}
    user_cfg = {}
    for section in parser.sections():
        for key, value in parser.items(section):
            if key in allowed_keys:
                user_cfg[key] = _coerce_value(key, value, defaults)
    return user_cfg


def load_config(config_path: str) -> TrainConfig:
    """
    配置加载入口：
    1) 校验后缀
    2) 读取用户配置
    3) 合并默认值
    4) 构造强类型 TrainConfig
    """
    if not config_path.endswith(".ini"):
        raise ValueError("仅支持 .ini 配置文件。")
    user_cfg = _load_ini_config(config_path)
    merged = _default_config_dict()
    merged.update(user_cfg)
    return TrainConfig(**merged)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数。

    设计思路：
    - 配置文件负责“常用默认实验”
    - CLI 负责“临时覆盖”少数字段（快速试跑/对照实验）
    """
    parser = argparse.ArgumentParser(description="训练 STLLM（教学版极简 LLM）")
    parser.add_argument(
        "--config",
        default="config.ini",
        help="配置文件路径（.ini），默认 config.ini",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="可选覆盖配置中的 train_steps，便于快速试跑",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="可选覆盖配置中的 prompt",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "chat"],
        default="train",
        help="运行模式：train（训练并保存）或 chat（加载权重聊天）",
    )
    return parser


def _save_checkpoint(
    checkpoint_path: str,
    model: MiniLLM,
    model_cfg: ModelConfig,
    dataset,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_cfg": {
            "embed_dim": model_cfg.embed_dim,
            "block_size": model_cfg.block_size,
            "num_heads": model_cfg.num_heads,
            "num_layers": model_cfg.num_layers,
        },
        "chars": sorted(dataset.stoi.keys()),
    }
    torch.save(payload, checkpoint_path)


def _load_checkpoint(checkpoint_path: str, device: str) -> dict:
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"未找到 checkpoint: {checkpoint_path}。请先运行训练模式生成权重。"
        )
    return torch.load(checkpoint_path, map_location=device)


def _build_codec_from_chars(chars: list[str]):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    return stoi, encode, decode


def run_training(
    config_path: str,
    train_steps_override: Optional[int] = None,
    prompt_override: Optional[str] = None,
) -> None:
    """
    训练主流程：
    - 读配置 -> 准备数据 -> 构建模型 -> 训练 -> 生成示例
    """
    cfg = load_config(config_path)
    # 命令行覆盖优先级高于配置文件，方便快速实验。
    if train_steps_override is not None:
        cfg.train_steps = train_steps_override
    if prompt_override is not None:
        cfg.prompt = prompt_override

    # 固定随机种子，保证实验可复现（同设备/同版本下尽量一致）。
    torch.manual_seed(cfg.seed)

    # 读取真实语料并构建字符级数据集。
    text = load_corpus(cfg.corpus_path)
    dataset = build_text_dataset(text)

    # 模型结构参数与训练参数分离，便于后续扩展（例如 val/eval 配置）。
    model_cfg = ModelConfig(
        embed_dim=cfg.embed_dim,
        block_size=cfg.block_size,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
    )
    model = MiniLLM(model_cfg, dataset.vocab_size, cfg.device).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print("开始训练 STLLM（教学版极简LLM）...\n")
    print(f"语料文件: {cfg.corpus_path}")
    print(f"词表大小: {dataset.vocab_size} | 数据长度: {len(dataset.data)}\n")

    for step in range(cfg.train_steps):
        # 采样一个 batch，目标是“预测下一个字符”。
        xb, yb = get_batch(dataset.data, cfg.block_size, cfg.batch_size, cfg.device)
        # 前向计算 loss。
        _, loss = model(xb, yb)
        # 标准优化步骤：清梯度 -> 反向传播 -> 参数更新。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg.eval_interval == 0:
            print(f"训练步骤 {step:4d} | 损失: {loss.item():.3f}")

    print("\n训练完成，开始生成：\n")
    # prompt 中若出现 OOV（词表外字符），先过滤，避免编码报错。
    prompt = "".join(ch for ch in cfg.prompt if ch in dataset.stoi)
    if not prompt:
        # 若过滤后为空，回退到语料第一个字符，确保可生成。
        prompt = dataset.text[:1]
        print("提示：prompt 中字符不在词表，已自动回退到语料首字符。")

    # 生成阶段输入形状要是 (B, T)，这里 B=1。
    context = torch.tensor([dataset.encode(prompt)], device=cfg.device)
    output = model.generate(context, max_new_tokens=cfg.max_new_tokens)
    print("输入：", prompt)
    print("生成：", dataset.decode(output[0].tolist()))
    _save_checkpoint(cfg.checkpoint_path, model, model_cfg, dataset)
    print(f"\n已保存模型权重：{cfg.checkpoint_path}")


def run_chat(config_path: str) -> None:
    """
    交互式聊天模式：
    - 仅加载 checkpoint，不执行训练
    - 连续读取用户输入并生成回复
    """
    cfg = load_config(config_path)
    # 尝试启用 readline，改善终端中的行编辑体验（退格、方向键、历史记录）。
    # 在少数环境不可用时静默降级，不影响主流程。
    try:
        import readline  # noqa: F401
    except Exception:
        pass

    ckpt = _load_checkpoint(cfg.checkpoint_path, cfg.device)

    model_cfg = ModelConfig(**ckpt["model_cfg"])
    chars = ckpt["chars"]
    stoi, encode, decode = _build_codec_from_chars(chars)

    model = MiniLLM(model_cfg, vocab_size=len(chars), device=cfg.device).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("进入聊天模式（输入 exit 退出）")
    print(f"已加载权重：{cfg.checkpoint_path}\n")

    while True:
        user_input = input("你> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("已退出聊天模式。")
            break
        if not user_input:
            continue

        prompt = "".join(ch for ch in user_input if ch in stoi)
        if not prompt:
            print("模型> 输入字符均不在词表中，请尝试使用语料中出现过的字符。")
            continue

        context = torch.tensor([encode(prompt)], device=cfg.device)
        output = model.generate(context, max_new_tokens=cfg.max_new_tokens)
        text = decode(output[0].tolist())
        # 只展示新生成部分，更像“回复”而非续写原提示。
        reply = text[len(prompt) :]
        print(f"模型> {reply}")
