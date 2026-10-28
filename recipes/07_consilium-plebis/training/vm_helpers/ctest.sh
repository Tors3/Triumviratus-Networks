#!/usr/bin/env bash
# Compression test on 48 MB slices of a checkpoint, a float .pt and a .nnue (28/10/2026): xz -9e, zstd -22 --long,
# 7z LZMA2 -mx9 when present, and the byte-plane split (bytesplit.py) followed by xz -9e. Sizes in % of the slice.
set -uo pipefail
V=/root/valfix; W=/root/ctest; mkdir -p "$W"; cd "$W"
take() { dd if="$1" of="$2" bs=1M skip="$3" count=48 status=none; }
take /root/valfix/F_final.ckpt ck.bin 2000
take /root/valfix/F_final.pt pt.bin 400
take /root/c2/run_fr3/nets/FR3_avg3_step240282.nnue nn.bin 100
for f in ck pt nn; do
  ( xz -9e -T1 -k -c $f.bin > $f.xz
    zstd -q --ultra -22 --long=27 -T1 -c $f.bin > $f.zst
    python3 "$V/bytesplit.py" split $f.bin $f.split && xz -9e -T1 -c $f.split > $f.split.xz
    command -v 7z > /dev/null && 7z a -mx9 -bd -bso0 $f.7z $f.bin
    o=$(stat -c %s $f.bin)
    for c in $f.xz $f.zst $f.split.xz $f.7z; do
      [ -f "$c" ] && echo "$c $(( $(stat -c %s $c) * 1000 / o ))"   # per mille of the slice
    done ) &
done
wait
command -v 7z || echo "no 7z"; zstd --version; xz --version | head -1
