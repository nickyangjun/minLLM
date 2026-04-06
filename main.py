from train import build_arg_parser, run_training


def main() -> None:
    """
    程序入口：
    - 只负责解析命令行参数并调用训练主流程
    - 训练细节都放在 train.py，保持入口足够薄
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    run_training(args.config, args.train_steps, args.prompt)


if __name__ == "__main__":
    main()