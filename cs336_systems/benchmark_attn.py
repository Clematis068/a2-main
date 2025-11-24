import torch
from cs336_basics.model import scaled_dot_product_attention
import argparse
import timeit

BATCH = 8
D_LIST = [16, 32, 64, 128]
SEQ_LIST = [256, 1024, 4096, 8192, 16384]
WARMUP = 5
STEPS = 100
print("d_model\tseq_len\tfwd(ms)\tbwd(ms)\tmem_before(MB)")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()

def main(compile: bool = False):
    attn_fn = torch.compile(scaled_dot_product_attention) if compile else scaled_dot_product_attention

    for d in D_LIST:
        for n in SEQ_LIST:
            try:
                q = torch.rand((BATCH, d, n), device="cuda", requires_grad=True)
                k = torch.rand_like(q)
                v = torch.rand_like(q)

                with torch.no_grad():
                    for _ in range(WARMUP):
                        attn_fn(q, k, v)
                        torch.cuda.synchronize()

                with torch.no_grad():
                    torch.cuda.synchronize()
                    t0 = timeit.default_timer()
                    for _ in range(STEPS):
                        attn_fn(q, k, v)
                        torch.cuda.synchronize()

                    fwd_ms = (timeit.default_timer() - t0) * 1e3 / STEPS

                for _ in range(WARMUP):
                    out = attn_fn(q, k, v)
                    out.sum().backward()
                    torch.cuda.synchronize()
                    q.grad = k.grad = v.grad = None

                # timed backward & mem
                mem_before_total = 0.0
                torch.cuda.synchronize()
                t0 = timeit.default_timer()
                for _ in range(STEPS):
                    out = attn_fn(q, k, v)
                    torch.cuda.synchronize()
                    mem_before_total += torch.cuda.memory_allocated()
                    out.sum().backward()
                    torch.cuda.synchronize()
                    q.grad = k.grad = v.grad = None
                bwd_ms = (timeit.default_timer() - t0) * 1e3 / STEPS
                mem_before_mb = mem_before_total / STEPS / 1e6

                print(f"{d}\t{n}\t{fwd_ms:.2f}\t{bwd_ms:.2f}\t{mem_before_mb:.0f}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"{d}\t{n}\tOOM\tOOM\tOOM")
                else:
                    raise

if __name__ == "__main__":
    args = _parse_args()
    main(args.compile)