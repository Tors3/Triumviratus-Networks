# Trainer

Every network is trained with Stockfish's [`nnue-pytorch`](https://github.com/official-stockfish/nnue-pytorch)
(GPLv3). Our changes are kept as patches against a pinned upstream commit.

| networks | upstream commit | our changes |
|---|---|---|
| `rubicon-v1` | `2db3787` (last L1-2560 SFNNv8 layout) | none |
| `rubicon-alea-v1` | master of June 2026 | none |
| `rubicon-alea-v2`, `v3`, experiments | `89d5725` ("Towards Perfect Cross Eval (#477)") | [`alea-fork/`](alea-fork/) |
| `legio-septima` | `9f72946` (2026-07-26) | [`../recipes/05_legio-septima/triumviratus_passedpawns.patch`](../recipes/05_legio-septima/triumviratus_passedpawns.patch) |
| MoE-1024 | `9f72946` | [`triumviratus_trainer.patch`](triumviratus_trainer.patch) (superset of the one above) |

## `alea-fork/`

- `alea_fork_vs_89d5725.patch` — changes to tracked files: the `PawnPair`, `PassedPawns` and `CandidatePassers`
  indexing in the C++ loader, the `min_passed_pawns` streaming filter, the feature registry, and the per-block
  learning rates (`--pawnpair-lr`, `--passedpawns-lr`, `--candidatepassers-lr`) with a multi-group optimizer.
- `new_files/` — files that do not exist upstream: the three feature blocks and `cross_check_trium.py`.

```bash
git clone https://github.com/official-stockfish/nnue-pytorch && cd nnue-pytorch
git checkout 89d5725
git apply ../trainer/alea-fork/alea_fork_vs_89d5725.patch
cp ../trainer/alea-fork/new_files/{pawn_pair,passed_pawns,candidate_passers}.py model/modules/features/
cp ../trainer/alea-fork/new_files/cross_check_trium.py .
bash compile_data_loader.sh
```

## `triumviratus_trainer.patch`

```bash
git clone https://github.com/official-stockfish/nnue-pytorch && cd nnue-pytorch
git checkout 9f72946529c4187d3679014036cd22c3be419716
git apply ../trainer/triumviratus_trainer.patch
```
Adds `PassedPawns`, the four-expert `HalfKAv2_hm_P4^` feature set and the `NNUE_BACKWARD_TILE` knob.

**Before any run:** `cross_check_eval.py` (upstream) on a random network, engine against trainer. For the MoE, build
the random net with `make_random_net.py --perturb-phases`: with all per-phase deltas at zero a wrong phase index is
invisible.
