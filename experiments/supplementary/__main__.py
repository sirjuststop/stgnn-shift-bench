"""CLI: python -m experiments.supplementary --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def needed_checkpoints(config):
    needed = set()
    for table in ("table10", "table11", "table31"):
        spec = config[table]
        for dataset in spec["datasets"]:
            needed.update((dataset, name, seed) for name in spec["models"] for seed in config["protocol"]["seeds"])
    needed.update((config["table21"]["source"], name, seed) for name in config["table21"]["models"]
                  for seed in config["protocol"]["seeds"])
    return sorted(needed)


def validate_config(config):
    p = config["protocol"]
    for key in ("seeds", "mask_seeds"):
        if not p[key] or len(p[key]) != len(set(p[key])) or any(s < 0 for s in p[key]):
            raise ValueError(f"Invalid or duplicate {key}")
    if p["batch_size"] < 1:
        raise ValueError("batch_size must be positive")
    for table in ("table10", "table11", "table31"):
        spec = config[table]
        if not spec["models"] or len(spec["models"]) != len(set(spec["models"])):
            raise ValueError(f"Invalid model list: {table}")
        if any(not 0 < v < 1 for v in spec["ratios"]):
            raise ValueError(f"Invalid dropout ratio: {table}")
    from experiments.supplementary.models import ADAPTIVE
    if set(config["table31"]["models"]) != set(ADAPTIVE):
        raise ValueError("Table 31 requires all six adaptive architectures")
    if config["table31"]["ratios"] != [.3] or config["table31"]["corruptions"] != ["zero"]:
        raise ValueError("Table 31 correlates routing with 30% zero-fill degradation")
    spec = config["table21"]
    if 0 not in spec["budgets"] or any(k < 0 for k in spec["budgets"]):
        raise ValueError("Few-shot budgets must include k=0 and be nonnegative")
    if spec["steps"] < 1 or spec["learning_rate"] <= 0:
        raise ValueError("Few-shot steps and learning rate must be positive")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Supplementary experiment reconstruction, Tables 10/11/21/31")
    parser.add_argument("command", choices=["table10", "table11", "table21", "table31", "check", "train", "train-missing"])
    parser.add_argument("--config", default="configs/supplementary_experiments.json")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", help="New result directory, or a new checkpoint path for train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=0, help="Diagnostic cap; 0 evaluates every test window")
    parser.add_argument("--model")
    parser.add_argument("--dataset", choices=["metr-la", "pems-bay", "pems04"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="Override training epochs; recorded in checkpoint")
    parser.add_argument("--max-batches", type=int, default=0, help="Training smoke-test cap, recorded in checkpoint")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args(argv)
    if min(args.max_windows, args.max_batches) < 0 or args.threads < 1:
        parser.error("Diagnostic limits must be nonnegative and threads must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    import torch
    torch.set_num_threads(args.threads)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    if args.command == "check":
        from experiments.supplementary.data import read_data, adjacency
        missing = []
        for dataset in config["checkpoints"]:
            try:
                values, _, _ = read_data(args.data_root, dataset)
                adjacency(args.data_root, dataset, values.shape[1])
            except (FileNotFoundError, ValueError) as exc:
                missing.append(str(exc))
        for dataset, model, seed in needed_checkpoints(config):
            path = config["checkpoints"][dataset][model].format(seed=seed)
            if not Path(path).is_file():
                missing.append(path)
        print(json.dumps({"ready": not missing, "missing": missing}, indent=2))
        return 1 if missing else 0
    if args.command in ("train", "train-missing"):
        from experiments.supplementary.train import train
        if args.command == "train":
            if not args.model or not args.dataset:
                parser.error("train requires --model and --dataset")
            work = [(args.dataset, args.model, args.seed)]
        else:
            if args.output:
                parser.error("train-missing uses the explicit checkpoint paths in --config")
            work = needed_checkpoints(config)
        for dataset, model, seed in work:
            output = args.output or config["checkpoints"][dataset][model].format(seed=seed)
            if args.command == "train-missing" and Path(output).exists():
                continue
            train(config, args.data_root, model, dataset, seed, output, args.device,
                  args.epochs, args.max_batches, args.hidden, args.seq_len, args.horizon)
        return 0
    output = Path(args.output or ("results/supplementary_regenerated/" + args.command))
    if (output / "results.json").exists():
        parser.error(f"Results already exist in {output}; choose a new --output directory")
    if args.command == "table21":
        from experiments.supplementary.fewshot import run
        run(config, args.config, args.data_root, output, args.device, args.max_windows)
    else:
        from experiments.supplementary.evaluate import run_table
        run_table(args.command, config, args.config, args.data_root, output, args.device, args.max_windows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
