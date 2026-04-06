# minLLM

一个基于 PyTorch 的极简中文字符级语言模型实现，面向学习与快速实验场景。项目提供从语料读取、词表构建、Transformer 建模到训练与生成的完整闭环，并通过 `INI` 配置文件支持参数管理。

## 项目定位

- **目标**：提供一个结构清晰、可运行、可扩展的最小语言模型工程。
- **范围**：用于理解训练流程与核心机制，不以生产级性能为目标。
- **特点**：模块化拆分、配置驱动、注释完善，适合教学和个人研究。

## 功能特性

- 基于 Transformer 的因果语言模型（Causal LM）
- 字符级词表构建与编码/解码
- 可配置训练参数与运行参数（`config.ini`）
- 训练日志输出与自回归文本生成
- 支持命令行临时覆盖部分配置（如训练步数、提示词）

## 环境要求

- Python `>=3.10`
- 依赖见 `requirements.txt`（当前核心依赖为 `torch`）

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

### 1) 使用默认配置训练

```bash
python main.py
```

### 2) 覆盖训练步数进行快速验证

```bash
python main.py --train-steps 50
```

### 3) 覆盖生成提示词

```bash
python main.py --prompt "人工智能"
```

## 训练与生成流程

1. 从 `config.ini` 读取参数并合并默认值
2. 加载并清洗语料，构建字符级词表
3. 随机采样训练片段，执行 next-token 预测
4. 使用交叉熵损失与 Adam 优化器更新参数
5. 训练完成后进行自回归采样生成文本

## 可扩展方向

- 引入训练/验证集切分与验证损失评估
- 增加 temperature、top-k 等采样策略
- 从字符级升级到子词级（BPE/SentencePiece）
- 增加检查点保存/恢复与实验记录
- 增加基础测试与 CI 流水线

## 注意事项

- 当前默认语料较小，生成质量受限，适合快速迭代验证。
- 若 `prompt` 含词表外字符，代码会自动回退到语料内可用字符。

## License

MIT License
