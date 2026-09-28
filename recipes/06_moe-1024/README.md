# 06 — MoE-1024 (Triumviratus 8.0, in training since 28 September 2026)

`legio-septima` with its king-relative block split into **four experts by material phase**. The full narrative — why,
the design, every measurement as it happens — is in the engine's
[`NETWORKS.md`](https://github.com/Tors3/Triumviratus/blob/main/NETWORKS.md). This folder holds what is needed to
rerun it.

| | |
|---|---|
| Status | pretraining (P) 450 epochs × 1 G positions, then fine-tune (F) 22 epochs; the file will be added here when it ships |
| Architecture | SFNNv16 with `HalfKAv2_hm_P4^`: `Full_Threats + HalfKAv2_hm_P4^ + PP_3Wide + PassedPawns`, L1 1024 |
| Inputs | 154,576 (162,768 in training, with the factorised features) |
| Trainer | official `nnue-pytorch` at **`9f72946`** + [`trainer/triumviratus_trainer.patch`](../../trainer/triumviratus_trainer.patch) (PassedPawns, `HalfKAv2_hm_P4`, `NNUE_BACKWARD_TILE`) |
| Engine | Triumviratus 8.0 built with `-DTRIUMV_PSQ_PHASES=4` |
| Hardware | 4× RTX 5090 (vast.ai), 192 threads, 251 GB RAM; ≈ 4.97 M positions/s, ≈ 3 $/h |

## Files

| file | what it does |
|---|---|
| [`setup_moe1024.sh`](setup_moe1024.sh) | environment, patched trainer, engine build with 4 phases |
| [`download_bt4.sh`](download_bt4.sh) | the 42 BT4-relabelled binpacks, 701 GB |
| [`make_random_net.py`](make_random_net.py) | random network with `--perturb-phases` for the engine/trainer gate |
| [`eqfreq.py`](eqfreq.py), [`bucket_occupancy.py`](bucket_occupancy.py), [`pp_active.py`](pp_active.py) | measurements behind the phase cut points and feature counts |
| [`bench_ddp.sh`](bench_ddp.sh) | real DDP throughput at a given batch, workers and skip (steady rate, compile excluded) |
| [`verify_ft_backward.py`](verify_ft_backward.py) | feature-transformer backward timed on real batches |
| [`train_moe1024.sh`](train_moe1024.sh) | the whole run: P, conversion, F, periodic exports |
| [`export_nets.sh`](export_nets.sh), [`serialize_last.sh`](serialize_last.sh) | `.nnue` from saved checkpoints / from the live checkpoint, on CPU |
| [`LOG_RUN.md`](LOG_RUN.md) | the run log (in Italian): setup, gate, measurements, launch |

## Launch

```bash
bash setup_moe1024.sh $HOME/moe
bash download_bt4.sh /data/bt4 850 3
# gate first: engine vs trainer on a random net with random per-phase deltas
BATCH=524288 ROOT=/root/moe/run P_EPOCHS=450 F_EPOCHS=22 nohup ./train_moe1024.sh > train_moe1024.log 2>&1 &
```

Key settings, all in `train_moe1024.sh`:
- batch **524,288**, lr **8e-4** (4e-4 at 131,072 × √4), one-cycle, 5 % warmup, final divisor 1000;
- λ 1.0 with a −0.3 cycle (25 % warmup), jitter 0.0035 / 0.0070, decay 0.999;
- `pc-y` −0.20 / 0.45 / 1.0 / 0.95 / 0.75;
- `random-fen-skipping 2`, `soft-early 20`, no hard early skip (the ≥ 24-piece expert needs openings);
- `NNUE_WORKER_THREAD_RATIO=0.1`, 46 workers per GPU (0.05 starves the GPUs: one feature thread per rank);
- binpacks under 5 GB excluded (40 of 42 used);
- F: batch 262,144, lr 5.66e-4, resumed from the P weights.

## Following the run

From a local machine: [`tools/fetch_net.ps1`](../../tools/fetch_net.ps1) downloads the exports (and with `-Last`
serializes the live checkpoint first); [`tools/match_moe.ps1`](../../tools/match_moe.ps1) plays a checkpoint against
`legio-septima` with the same 8.0 search on both sides.
