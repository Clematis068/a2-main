import os
import timeit
import torch
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import argparse
from contextlib import nullcontext

_PRESETS: dict[str, dict[str, int]] = {
    "sm": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "md": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "lg": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

_MODE_MAP: dict[str, str] = {
    "forward": "f",
    "forward_backward": "fb",
    "train": "fbl"
}

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    size_gp = p.add_mutually_exclusive_group()
    size_gp.add_argument("--size", choices=_PRESETS.keys())
    size_gp.add_argument("--d_model", type=int)

    p.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="forward")

    p.add_argument("--d_ff", type=int)
    p.add_argument("--num-layers", dest="num_layers", type=int)
    p.add_argument("--num-heads", dest="num_heads", type=int)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=10_000)

    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)

    p.add_argument("--mixed", action="store_true")
    p.add_argument("--compile", dest="compile_model", action="store_true")
    p.add_argument("--memory", dest="profile_memory", action="store_true")

    return p.parse_args()

def run_benchmark(
    model_config: dict[str, any],
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    warmup: int,
    steps: int,
    mixed: bool,
    compile_model: bool,
    mode: str,
    profile_memory: bool,
    size_name: str | None = None,
    device: str = "cuda",
) -> dict[str, float]:

    if not torch.cuda.is_available() and device == 'cuda':
        raise RuntimeError("cuda exclusive")
    
    if "rope_theta" not in model_config:
        model_config["rope_theta"] = 10000.0
    
    model: torch.nn.Module = Transformer(vocab_size, seq_len, **model_config).to(
        device = device
    )

    if compile_model:
        torch.compile(model)

    model.train(mode in ("forward_backward", "train"))
    optimizer = AdamW(model.parameters(), lr = 1e-3) if mode == "train" else None

    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device = device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device = device)

    mixed_or_null_ctx = torch.autocast(device_type = device, dtype = torch.bfloat16) if mixed else nullcontext()
    scaler = torch.amp.GradScaler(enabled=mixed)
    fw_acc, bw_acc, opt_acc = [], [], []

    with mixed_or_null_ctx:
        for i in range(warmup + steps):
            if i == warmup and profile_memory:
                torch.cuda.memory._record_memory_history(max_entries = 1000000)

            torch.cuda.synchronize(device=device)
            d0 = timeit.default_timer()
            logits = model(inputs)
            torch.cuda.synchronize(device=device)
            fw_dt = timeit.default_timer() - d0

            dt_bw = 0.0
            if mode in ("forward_backward", "train"):
                loss = cross_entropy(logits, targets)
                d1 = timeit.default_timer()
                scaler.scale(loss).backward()
                torch.cuda.synchronize(device=device)
                dt_bw = timeit.default_timer() - d1

            dt_opt = 0.0
            if mode == "train":
                t2 = timeit.default_timer()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize(device=device)
                dt_opt = timeit.default_timer() - t2
                optimizer.zero_grad(set_to_none = True)
            else:
                model.zero_grad(set_to_none = True)

            if i >= warmup:
                fw_acc.append(fw_dt)
                bw_acc.append(dt_bw)
                opt_acc.append(dt_opt)

    if profile_memory:
        m = _MODE_MAP[mode]
        d = "bf16" if mixed else "fp32"

        size_str = size_name if size_name else f"custom_size{model_config['d_model']}"
        output_dir = "out/memory"
        os.makedirs(output_dir, exist_ok = True)
        torch.cuda.memory._dump_snapshot(f"{output_dir}/mem_{d}_{size_str}_{m}_{seq_len}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    fw_t = torch.tensor(fw_acc, device=device)
    bw_t = torch.tensor(bw_acc, device=device)
    opt_t = torch.tensor(opt_acc, device=device)

    total_t = fw_t + bw_t + opt_t

    results = {}

    results["fw_mean_ms"] = fw_t.mean().item() * 1e3
    results["fw_std_ms"] = fw_t.std().item() * 1e3 if len(fw_t) > 1 else 0.0
    results["fw_pct"] = (fw_t / total_t).mean().item() * 100 if total_t.mean() > 0 else 0.0

    results["bw_mean_ms"] = bw_t.mean().item() * 1e3 if mode in ("forward_backward", "train") else 0.0
    results["bw_std_ms"] = bw_t.std().item() * 1e3 if len(bw_t) > 1 and mode in ("forward_backward", "train") else 0.0
    results["bw_pct"] = (
        (bw_t / total_t).mean().item() * 100 if total_t.mean() > 0 and mode in ("forward_backward", "train") else 0.0
    )

    results["opt_mean_ms"] = opt_t.mean().item() * 1e3 if mode == "train" else 0.0
    results["opt_std_ms"] = opt_t.std().item() * 1e3 if len(opt_t) > 1 and mode == "train" else 0.0
    results["opt_pct"] = (opt_t / total_t).mean().item() * 100 if total_t.mean() > 0 and mode == "train" else 0.0

    results["total_mean_ms"] = total_t.mean().item() * 1e3
    results["total_std_ms"] = total_t.std().item() * 1e3 if len(total_t) > 1 else 0.0
    results["total_pct"] = 100.0

    return results

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("cuda sshit")
    
    args = _parse_args()

    # 1. 确定模型配置 (Config)
    if args.size:
        # 使用预设配置
        if args.size not in _PRESETS:
            raise ValueError(f"Invalid size preset: {args.size}. Valid options: {list(_PRESETS.keys())}")
        model_config = _PRESETS[args.size]
        size_name = args.size
        print(f"Configuration: Using preset '{size_name}'")
    else:
        # 使用自定义配置
        # 必须确保其他参数存在
        # required = ("", "", "", "")
        # if any(getattr(args, k) is None for k in required):
        if not all([args.d_ff, args.num_layers, args.num_heads]):
            raise ValueError("When using custom --d_model, you must also specify --d_ff, --num-layers, and --num-heads.")
        
        model_config = {
            "d_model": args.d_model,
            "d_ff": args.d_ff,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads
        }
        size_name = None
        print(f"Configuration: Custom (d_model={args.d_model}, L={args.num_layers}, H={args.num_heads})")

    # 2. 运行基准测试
    print("-" * 50)
    print(f"Running Benchmark | Mode: {args.mode} | Batch: {args.batch_size} | SeqLen: {args.seq_len}")
    print(f"Mixed Precision: {args.mixed} | Compile: {args.compile_model}")
    print("-" * 50)

    try:
        results = run_benchmark(
            model_config=model_config,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            warmup=args.warmup,
            steps=args.steps,
            mixed=args.mixed,
            compile_model=args.compile_model,
            mode=args.mode,
            profile_memory=args.profile_memory,
            size_name=size_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 3. 打印结果
        print("\n" + "=" * 30)
        print("       BENCHMARK RESULTS       ")
        print("=" * 30)
        
        # 打印总时间
        print(f"Total Step Time: {results['total_mean_ms']:.2f} ms ± {results['total_std_ms']:.2f} ms")
        print("-" * 30)
        
        # 打印详细分解
        print(f"{'Phase':<10} | {'Time (ms)':<15} | {'Std (ms)':<10} | {'% Total':<10}")
        print("-" * 55)
        
        phases = [
            ("Forward", "fw"),
            ("Backward", "bw"),
            ("Optimizer", "opt")
        ]
        
        for name, key_prefix in phases:
            mean = results[f"{key_prefix}_mean_ms"]
            std = results[f"{key_prefix}_std_ms"]
            pct = results[f"{key_prefix}_pct"]
            if mean > 0:
                print(f"{name:<10} | {mean:<15.2f} | {std:<10.2f} | {pct:<9.1f}%")
        print("=" * 30)

    except RuntimeError as e:
        print(f"\nError during benchmark: {e}")
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")

if __name__ == "__main__":
    main()
  

