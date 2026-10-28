# Lossless byte-plane split of a file of 4-byte numbers (float32 checkpoints, .pt): byte k of every 4-byte word goes
# to plane k, so the sign/exponent bytes sit together and compress far better than the interleaved stream
# (28/10/2026, archives of the Consilium Plebis run). Header: magic, original length, then the 4 planes.
# Output is written in 256 MB pieces: a single write of more than 2 GB to a pipe came out truncated (first version,
# caught by the md5 check of the 5.9 GB checkpoints).
# Usage: bytesplit.py split <in> <out>   |   bytesplit.py join <in> <out>     (stdin/stdout with "-")
import sys
import numpy as np

MAGIC = b"BSPLIT4\0"
PIECE = 256 << 20

def rd(p):
    return sys.stdin.buffer.read() if p == "-" else open(p, "rb").read()

def out(p):
    return sys.stdout.buffer if p == "-" else open(p, "wb")

def put(f, arr):
    mv = memoryview(arr).cast("B")
    for i in range(0, len(mv), PIECE):
        f.write(mv[i:i + PIECE])

def split(src, dst):
    b = rd(src)
    n = len(b)
    pad = (-n) % 4
    if pad:
        b += b"\0" * pad
    a = np.frombuffer(b, dtype=np.uint8).reshape(-1, 4)
    f = out(dst)
    f.write(MAGIC + n.to_bytes(8, "little"))
    for k in range(4):
        put(f, np.ascontiguousarray(a[:, k]))
    f.flush()

def join(src, dst):
    b = rd(src)
    assert b[:8] == MAGIC, "not a bytesplit file"
    n = int.from_bytes(b[8:16], "little")
    p = np.frombuffer(b, dtype=np.uint8, offset=16).reshape(4, -1)
    a = np.empty((p.shape[1], 4), dtype=np.uint8)
    for k in range(4):
        a[:, k] = p[k]
    f = out(dst)
    put(f, a.reshape(-1)[:n])
    f.flush()

if __name__ == "__main__":
    mode, src, dst = sys.argv[1:4]
    (split if mode == "split" else join)(src, dst)
