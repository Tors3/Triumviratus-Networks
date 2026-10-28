"""Consilium-2 engine gate (21/10/2026): the engine's `evalbatch` against the trainer's integer reference.

ep9_gate.fen / ep9_gate.int were written by `c2_gate_int.py --net c2_P_ep9.nnue --dump` (2,004 game positions; .int =
psqt positional total before the final /16). Run the C2 engine with `setoption name EvalFile value
C:\\rz\\c2\\nets\\c2_P_ep9.nnue`, then `evalbatch C:\\rz\\c2eng\\gate\\ep9_gate.fen <out.eng>`, then:
    python check_gate.py <out.eng>
Both columns must be 2004/2004.
"""
import sys
from pathlib import Path


def tdiv(a, b):
    q = abs(a) // b
    return -q if a < 0 else q


here = Path(__file__).parent
ref = [list(map(int, l.split())) for l in open(here / "ep9_gate.int") if l.strip()]
eng = [list(map(int, l.split())) for l in open(sys.argv[1]) if l.strip()]
assert len(ref) == len(eng), f"{len(ref)} reference lines, {len(eng)} engine lines"
ps = sum(tdiv(r[0], 16) == e[0] for r, e in zip(ref, eng))
po = sum(tdiv(r[1], 16) == e[1] for r, e in zip(ref, eng))
print(f"psqt {ps}/{len(ref)}, positional {po}/{len(ref)} equal -> {'OK' if ps == po == len(ref) else 'FAILED'}")
