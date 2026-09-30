"""Confronta il backward del FT upstream (atomicAdd) con quello a segmenti, su batch VERI (28/09/2026).

USO (dentro nnue-pytorch, PYTHONPATH=.):
    python verify_ft_backward.py /data/bt4/<file>.binpack --batch 65536 --reps 5
Stampa l'errore relativo massimo su grad_weight e grad_bias e i ms per chiamata dei due backward.
Gate: errore relativo (norma della differenza / norma) < 1e-5 su entrambi.
"""
import argparse
import importlib.util
import os

import numpy as np
import torch

from data_loader.config import DataloaderSkipConfig
from data_loader.dataset import SparseBatchProvider
from model.modules.feature_transformer.fused_ft_kernel import (
    BACKWARD_TILE_SIZE,
    make_fused_double_ft_backward_kernel,
    make_fused_double_ft_forward_kernel,
)

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ftseg", os.path.join(here, "ft_backward_segmented.py"))
ftseg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ftseg)

ap = argparse.ArgumentParser()
ap.add_argument("binpack")
ap.add_argument("--features", default="Full_Threats+HalfKAv2_hm_P4+PP_3Wide+PassedPawns")
ap.add_argument("--num-inputs", type=int, default=162768)
ap.add_argument("--batch", type=int, default=65536)
ap.add_argument("--l1", type=int, default=1024)
ap.add_argument("--psqt", type=int, default=8)
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()

dev = torch.device("cuda")
prov = SparseBatchProvider(args.features, [args.binpack], args.batch, num_workers=16,
                           config=DataloaderSkipConfig(random_fen_skipping=10), device=dev)
us, them, white, black, outcome, score, piece_count = next(iter(prov))[:7]
us = us.contiguous().view(-1)
them = them.contiguous().view(-1)
white = white.contiguous()
black = black.contiguous()
psqt = ((piece_count - 1) // 4).clamp(0, args.psqt - 1).contiguous()
B, MAXA = white.shape
act = (white >= 0).sum().item() / B
print(f"batch {B}, max attivi {MAXA}, attivi medi per prospettiva {act:.1f}, "
      f"indice massimo {int(max(white.max(), black.max()))} (tabella {args.num_inputs})")

out = args.l1 + args.psqt
g = torch.Generator(device=dev).manual_seed(0)
weight = (torch.randn(args.num_inputs, out, device=dev, generator=g) * 0.03).contiguous()
bias = (torch.randn(out, device=dev, generator=g) * 0.1 + 0.5).contiguous()
max_act = 255.0 / 256.0
h = args.l1 // 2

l0 = torch.empty(B, args.l1, device=dev)
wps = torch.empty(B, 1, device=dev)
bps = torch.empty(B, 1, device=dev)
clamped = torch.empty(B, 4, h, device=dev)
make_fused_double_ft_forward_kernel(MAXA, args.l1)(
    grid=(B,), args=(us.data_ptr(), them.data_ptr(), white.data_ptr(), black.data_ptr(), psqt.data_ptr(),
                     weight.data_ptr(), bias.data_ptr(), np.float32(max_act), l0.data_ptr(), wps.data_ptr(),
                     bps.data_ptr(), clamped.data_ptr(), np.int32(out)))
live = ((clamped > 0) & (clamped < max_act)).float().mean().item()
print(f"attivazioni non saturate: {live:.1%}")

grad_l0 = torch.randn(B, args.l1, device=dev, generator=g)
grad_w = torch.randn(B, 1, device=dev, generator=g)
grad_b = torch.randn(B, 1, device=dev, generator=g)


def upstream():
    gw = torch.zeros(args.num_inputs, out, device=dev)
    gb = torch.zeros(out, device=dev)
    k = make_fused_double_ft_backward_kernel(MAXA, args.l1, num_psqt_buckets=args.psqt)
    k(grid=((B + BACKWARD_TILE_SIZE - 1) // BACKWARD_TILE_SIZE,),
      args=(us.data_ptr(), them.data_ptr(), white.data_ptr(), black.data_ptr(), psqt.data_ptr(), weight.data_ptr(),
            bias.data_ptr(), np.float32(max_act), grad_l0.data_ptr(), grad_w.data_ptr(), grad_b.data_ptr(),
            clamped.data_ptr(), gw.data_ptr(), gb.data_ptr(), np.int32(B), np.int32(out)))
    return gw, gb


def segmented():
    return ftseg.segmented_backward(us, them, white, black, psqt, args.num_inputs, grad_l0, grad_w, grad_b,
                                    clamped, max_act, args.l1, out)


def timed(fn):
    fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(args.reps):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        r = fn()
        b.record()
        torch.cuda.synchronize()
        ms.append(a.elapsed_time(b))
    return r, sorted(ms)[len(ms) // 2]


(gw0, gb0), t0 = timed(upstream)
(gw1, gb1), t1 = timed(segmented)


def rel(a, b):
    return ((a - b).norm() / a.norm().clamp_min(1e-30)).item()


ew, eb = rel(gw0, gw1), rel(gb0, gb1)
print(f"errore relativo grad_weight {ew:.2e}, grad_bias {eb:.2e}, max |diff| {(gw0 - gw1).abs().max().item():.2e}")
print(f"upstream {t0:.1f} ms   segmenti {t1:.1f} ms   ->  {t0 / t1:.2f}x")
print("GATE", "OK" if ew < 1e-5 and eb < 1e-5 else "FALLITO")
