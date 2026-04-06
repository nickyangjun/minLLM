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
- 训练后自动保存 checkpoint，支持后续免训练加载
- 支持交互式聊天模式（`chat`）
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

## 使用方法

### 1) 训练模式（保存权重）

```bash
python main.py --mode train
```

训练结束后会自动保存权重，默认路径为 `checkpoints/minllm.pt`（可在 `config.ini` 的 `[checkpoint]` 区块修改）。

### 2) 聊天模式（加载权重，不再训练）

```bash
python main.py --mode chat
```

进入后输入文本即可生成回复，输入 `exit` 或 `quit` 退出。

### 3) 快速训练验证（覆盖训练步数）

```bash
python main.py --mode train --train-steps 50
```

### 4) 覆盖提示词（训练模式下的演示生成）

```bash
python main.py --mode train --prompt "人工智能"
```

## 运行流程

### 训练模式（`--mode train`）

1. 从 `config.ini` 读取参数并合并默认值
2. 加载并清洗语料，构建字符级词表
3. 随机采样训练片段，执行 next-token 预测
4. 使用交叉熵损失与 Adam 优化器更新参数
5. 训练完成后生成示例文本，并保存 checkpoint

### 聊天模式（`--mode chat`）

1. 从配置中读取 `checkpoint_path`
2. 加载模型结构与权重
3. 进入命令行交互循环，按输入逐轮生成

## 测试方法（手动 smoke test）

建议每次改动后至少执行以下检查：

1. **训练链路检查**
```bash
python main.py --mode train --train-steps 1
```
预期：能看到 loss 输出，并提示已保存 checkpoint。

2. **聊天链路检查**
```bash
python main.py --mode chat
```
预期：出现 `你>` 输入提示，输入文本后有 `模型>` 回复，`exit` 可退出。

## 可扩展方向

- 引入训练/验证集切分与验证损失评估
- 增加 temperature、top-k 等采样策略
- 从字符级升级到子词级（BPE/SentencePiece）
- 增加检查点保存/恢复与实验记录
- 增加基础测试与 CI 流水线

## 注意事项

- 当前默认语料较小，生成质量受限，适合快速迭代验证。
- 若输入含词表外字符，程序会过滤并在必要时回退到可用字符。
- 首次聊天前需先完成至少一次训练，以生成 checkpoint。

## License

MIT License
