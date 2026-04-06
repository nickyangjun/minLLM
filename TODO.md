# STLLM 学习路线 TODO

> 原则：每次只做一个改动，每次只验证一个问题。

## Milestone 0: 先跑通
- [ ] 能成功运行 `python main.py`
- [ ] loss 在训练过程中明显下降
- [ ] 能生成可读中文片段

## Milestone 1: 理解训练机制
- [ ] 把 `TRAIN_STEPS` 改成 200 / 1000 / 3000 做对照
- [ ] 记录每种设置下的 loss 曲线（可先手工记录）
- [ ] 观察“过短训练 vs 过长训练”的生成差异

## Milestone 2: 学会调参
- [ ] 单独修改 `EMBED_DIM`（16, 32, 64）并比较
- [ ] 单独修改 `BLOCK_SIZE`（8, 16, 32）并比较
- [ ] 单独修改 `LEARNING_RATE`（1e-2, 1e-3, 3e-4）并比较
- [ ] 写下每个参数变化的“现象 -> 解释”

## Milestone 3: 数据工程入门
- [ ] 从 `sample_corpus_zh.txt` 读取文本替换内置 `text`
- [ ] 增加最小清洗逻辑（去除过多空行、统一空白）
- [ ] 打印词表大小与数据长度，确认数据生效

## Milestone 4: 评估与泛化
- [ ] 切分 train/val（例如 9:1）
- [ ] 每 N 步评估一次 val loss
- [ ] 对比 train loss 和 val loss，识别过拟合

## Milestone 5: 生成质量
- [ ] 增加 `temperature` 采样
- [ ] 增加 `top-k` 采样
- [ ] 对比不同采样策略下文本质量与多样性

## Milestone 6: 模型结构理解
- [ ] 给 attention 增加 dropout（先 0.1）
- [ ] 给 FFN 增加 dropout
- [ ] 对比“有/无 dropout”的收敛和生成变化

## Milestone 7: 进阶（可选）
- [ ] 从字符级切换到子词级（BPE）
- [ ] 加命令行参数（argparse）
- [ ] 拆分代码模块（model.py, data.py, train.py）
