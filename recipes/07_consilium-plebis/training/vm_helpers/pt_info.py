# Size of the float tensors in nnue-pytorch .pt files (pickled model), to compare a new export with P_final.pt.
import sys
sys.path.insert(0, "/root/c2/nnue-pytorch")
import torch
def walk(x, pre=""):
    if torch.is_tensor(x):
        yield pre, x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from walk(v, f"{pre}.{k}" if pre else str(k))
for f in sys.argv[1:]:
    m = torch.load(f, weights_only=False, map_location="cpu")
    sd = m.state_dict() if hasattr(m, "state_dict") else m
    ts = list(walk(dict(sd)))
    by = {}
    for k, v in ts:
        by[str(v.dtype)] = by.get(str(v.dtype), 0) + v.numel() * v.element_size()
    big = sorted(ts, key=lambda kv: -kv[1].numel())[:4]
    print(f, type(m).__name__, len(ts), "tensors", {k: f"{b / 1e6:.0f} MB" for k, b in by.items()})
    for k, v in big:
        print("   ", k, tuple(v.shape), v.dtype)
