"""Average the model weights of several Lightning checkpoints (27/10/2026, Consilium Plebis: the final network is the
average of the five epochs of the tail of F's second cycle, author's decision).
The trainer's state_dict holds the model (keys "model.*") next to the optimizer state ("optimizer_state_dict"), the
lambda scheduler and the lambda jitter buffer. Only the floating-point model tensors are averaged (summed in float64,
cast back to their type); the integer model tensors (the routing table "router") must be identical in all the
checkpoints; every other entry comes from the last checkpoint given.
Usage: python avg_ckpt.py <out.ckpt> <in1.ckpt> ... <inN.ckpt>      (the last input is the base)"""
import sys

import torch


def model_tensors(sd):
    return {k: v for k, v in sd.items() if k.startswith("model.") and isinstance(v, torch.Tensor)}


def main():
    out, ins = sys.argv[1], sys.argv[2:]
    if len(ins) < 2:
        raise SystemExit("at least two checkpoints to average")
    acc, base = None, None
    for i, path in enumerate(ins):
        c = torch.load(path, map_location="cpu", weights_only=False)
        m = model_tensors(c["state_dict"])
        print(f"{path}: epoch label {c['epoch']}, global step {c['global_step']}, {len(m)} model tensors", flush=True)
        if acc is None:
            acc = {k: v.double() if v.is_floating_point() else v.clone() for k, v in m.items()}
        else:
            if set(m) != set(acc):
                raise SystemExit(f"{path}: model keys differ from the first checkpoint")
            for k, v in m.items():
                if v.is_floating_point():
                    acc[k] += v.double()
                elif not torch.equal(acc[k], v):
                    raise SystemExit(f"{path}: integer model tensor {k} differs from the first checkpoint")
        if i == len(ins) - 1:
            base = c
        del c, m
    sd = base["state_dict"]
    n, nfloat, largest, rel = len(ins), 0, 0.0, []
    for k, v in model_tensors(sd).items():
        if v.is_floating_point():
            avg = (acc[k] / n).to(v.dtype)
            d = (avg.double() - v.double()).abs()
            largest = max(largest, d.max().item())
            rel.append((d.norm() / v.double().norm().clamp_min(1e-30)).item())
            sd[k] = avg
            nfloat += 1
    torch.save(base, out)
    print(f"averaged {nfloat} floating model tensors over {n} checkpoints -> {out}")
    print(f"change against the last checkpoint: largest element {largest:.3e}, "
          f"largest relative norm over tensors {max(rel):.3e}")


if __name__ == "__main__":
    main()
