# Log del run MoE-1024

Date in UTC. Comandi lanciati dall'utente sulla VM su indicazione della sessione Claude: da quella sessione la VM non si
raggiunge, perché il proxy lascia passare solo HTTPS sulla porta 443.

## VM (28/09/2026)
- vast.ai, istanza a noleggio (indirizzo e porta SSH omessi: cambiano a ogni istanza).
- 4× RTX 5090 32 GB, PCIe x16, tutte sullo stesso nodo NUMA (`topo -m`: NODE). Driver 595.91.07, CUDA 13.2.
- 192 thread, 251 GB di RAM, 1,4 TB di disco (1.326 GB liberi prima del download).
- Python del venv `/venv/main` 3.12.14, torch cu128 installato a mano prima del setup.
- Il repo è privato: copiato sulla VM come ZIP da GitHub in `~/tt`, senza credenziali sulla VM.

## Download (28/09)
- 11:08: `download_bt4.sh /data/bt4 850 3`.
- Piano: **701,1 GB**, non i ~790 del README. La differenza è spiegata e verificata via API HF:
  - `linrock_relabel_1`: il mese `test80-2022-10-oct` c'è sia intero sia in `part_0`/`part_1` (69,1 GB), e lo
    script lo scarica una volta sola;
  - `xushawn/test80-bt4-relabel`: su HF sono 17,7 GB, non ~33.

  Nessun dato mancante: README, CLAUDE.md e `download_bt4.sh` sono corretti a 701 GB.

  | repo | file | GB |
  |---|---|---|
  | vondele/master-binpacks_relabel | 5 | 129,4 |
  | vondele/linrock_relabel_1 | 13 | 203,9 |
  | vondele/linrock_relabel_2 | 12 | 141,8 |
  | vondele/from_kaggle_2_relabel | 5 | 110,6 |
  | vondele/from_kaggle_1_relabel | 5 | 97,6 |
  | xushawn/test80-bt4-relabel | 2 | 17,7 |
- 11:15: 120 MB/s con 3 download in parallelo.
- ~11:25: rilanciato con 8 paralleli (`>> ~/download.log`), dopo aver fermato il gruppo di processi del lancio a 3:
  114 MB/s, quindi il limite è il link (~1 Gbit/s), non il parallelismo. Fine stimata ~12:50.
- I file di `xushawn` si chiamano `*.relabel.binpack` (senza `-BT4-tf13tune`): lo script di training deve prendere
  tutti i `*.binpack` di `/data/bt4`, non il glob della legio.

## Setup (28/09)
- 11:1x: `setup_moe1024.sh $HOME/moe` in una finestra tmux, output in `~/setup.log`.
- `pip install -r requirements.txt` è rimasto oltre 12 minuti nel backtracking del resolver (`numpy<2.0`): fermato e
  sostituito con `uv pip install --python /venv/main/bin/python3 -r requirements.txt "cupy-cuda12x<14"`, circa 2 minuti.
  Risultato: torch 2.11.0+cu128, cupy-cuda12x 13.6.0, numpy 1.26.4, numba 0.67.0, scipy 1.17.1.
- Setup rilanciato in background (`nohup ... > ~/setup.log`): torch con CUDA ok su 4 GPU, CuPy con kernel fusi,
  Triton, input legio 86.992.
- Controllo MoE a mano (quello di `setup_vm.sh` cercava la chiave sbagliata e veniva saltato, corretto in `2aa967e`):
  `HalfKAv2_hm_P4^` registrata, **154.576 input reali** (come il README), 162.768 input di training.
- `gpu_ceiling.py` e `loader_ceiling.py` misuravano le feature della legio: aggiunto `--features` (`44c6056`),
  applicato a mano anche nella copia `~/tt` della VM.

## Stato alle ~11:55
- Download in corso; setup in chiusura (build del motore `-DTRIUMV_PSQ_PHASES=4`).
- Gate `cross_check_eval` lanciato sul binpack completo più piccolo, output in `~/moe/cross_check.log`.
- Da fare: esito del gate, misure del batch (gpu_ceiling per GPU e prova DDP vera a 131072 e 262144),
  loader_ceiling, budget in giorni dall'utente, `train_moe1024.sh`, lancio.

## Ripresa dalla sessione locale (12:10–12:40, SSH diretto con una chiave dedicata)
- **Setup: il `git clone` di Tors3/Triumviratus era fermo da 15 minuti.** Il repo è pesante e si contendeva il link con
  il download. Ho fermato il clone e `setup_moe1024.sh`, poi portato con scp i sorgenti del commit `ccddd72` in
  `~/moe/engine/source`.
  - Build: `make avx512 EXE=triumv_moe1024 EXTRACXXFLAGS="-DTRIUMV_L1=1024 -DTRIUMV_PSQ_PHASES=4"`.
  - Il motore senza rete accanto al binario si ferma (FATAL): `random_moe1024p.nnue` è copiata anche come
    `nn-legio-septima.nnue`.
- **Rete casuale:** `make_random_net.py --perturb-phases`, delta per fascia casuali, 231 MB.
- **Gate `cross_check_eval`: SUPERATO.** 4.096 posizioni di `test80-2023-11-nov`, L1 1024, motore
  `-DTRIUMV_PSQ_PHASES=4`:

  | | R² | errore medio | 3σ |
  |---|---|---|---|
  | float | 0,999999 | 1,70 | 5,09 |
  | quantizzata | 1,000000 | 0,64 | 1,84 |

- **`gpu_ceiling.py` misurava su 86.992 input anche con `--features`.** Leggeva `num_inputs` (minuscolo), che non
  esiste, e ricadeva sul default. Corretto con `NUM_INPUTS`, che dà 162.768. Le misure sotto sono quelle corrette.
- **Tetto GPU per scheda** (sintetico, loader escluso, feature MoE):

  | batch per GPU | pos/s | VRAM |
  |---|---|---|
  | 32.768 | 313k | 7,3 GB |
  | 65.536 | 348k | 7,4 GB |
  | 131.072 | 369k | 8,7 GB |

  Con le feature della legio, a 131.072, sono 418k: **la MoE costa il 12% di throughput GPU**.
- **Loader** (`loader_ceiling.py`, una sola istanza):
  - scala quasi linearmente fino a 32 thread (circa 40k pos/s per thread);
  - il picco è **1,6–1,7M pos/s a 64 thread**; oltre crolla (96 thread 1,5M, 128 thread 0,9M, 160 thread 0,5M);
  - skip 3 e skip 10 danno lo stesso risultato.

  In DDP ogni rank ha il suo loader: 4 × 24 thread bastano con margine, **il loader non è il collo**.
- **DDP vero** (`bench_ddp.sh`: torchrun, 4 GPU, 24 worker per rank, `NNUE_WORKER_THREAD_RATIO=0.05`, skip 10, 22
  binpack; velocità a regime, esclusa la compilazione):

  | batch globale | per GPU | it/s | pos/s |
  |---|---|---|---|
  | 131.072 | 32.768 | 10,00 | **1,31M** |
  | 262.144 | 65.536 | circa 5,4 | **circa 1,40M** |
  | 524.288 | 131.072 | 2,86 | **1,50M** |

  - I valori coincidono con 4× il tetto per scheda: l'all-reduce su PCIe 5.0 costa quasi niente, il limite è il calcolo
    nella GPU.
  - Raddoppiare il batch rende circa +7% per volta.
  - ⚠️ La barra di Lightning mostra la media dall'inizio, compilazione compresa: a 262.144 finiva a 3,76 it/s con un
    regime di 5,4. `bench_ddp.sh` calcola il regime dai punti oltre il 20%.
- **`NNUE_BACKWARD_TILE` 4 / 8 / 16 / 32:** 369k / 358k / 352k / 352k. Manopola morta anche sulle 5090: resta 4.
- **Profilo di uno step** (`prof_step.py`, batch 131.072 per GPU, 353 ms):
  - `fused_double_ft_backward` **252 ms (71%)**;
  - `fused_double_ft_forward` 72 ms (20%);
  - tutto il resto circa 9%.

  Le 5090 segnano il 97% di utilizzo ma assorbono **160–200 W su 575**. Il forward legge le righe dei pesi a circa
  1,8 TB/s, cioè al limite di banda. Il backward sposta la stessa quantità di dati ma impiega 3,5 volte tanto: il
  tempo va negli atomicAdd contesi su `grad_weight`.

  **La leva vera per andare oltre 1,5M pos/s è riscrivere quel backward come gather**: indici ordinati per feature e
  riduzione per riga in registri, una sola scrittura per riga. Il limite teorico è il costo del forward, quindi fino
  a circa 1,8× sull'intero step.

  🔴 **SMENTITO poco dopo, 12:50–13:00.** Il profilo usava batch sintetici: 120 feature attive a caso su tutta la
  tabella.
  - Sui batch veri (`verify_ft_backward.py`) le feature attive sono **52 per prospettiva** e concentrate. Il backward
    upstream costa **14,7 ms per 65.536 posizioni**; quello a segmenti (`ft_backward_segmented.py`) è corretto
    (errore relativo 1,4e-6) ma **più lento**, 18,0 ms. Non si usa.
  - **Il vero freno era il loader.** Con `NNUE_WORKER_THREAD_RATIO=0.05`, l'ottimo della legio, ogni rank ha **un solo**
    thread che costruisce le feature, e con threat e MoE un thread ne fa circa 350k pos/s.
  - Una 5090 da sola, sul training vero: 381k pos/s con quota 0.05 (GPU al 28%, 134 W), **1.907k** con 0.25 o 0.5
    (GPU al 95%, 415 W).

## Loader e skip, 4 GPU (13:00–13:25; batch 524.288, 46 worker per rank, `ITERS=1200`)
| quota di thread di feature | skip | pos/s |
|---|---|---|
| 0.25 | 10 | 3,40M |
| 0.25 | 3 | 4,62M |
| 0.15 | 3 | 4,70M |
| **0.1** | 3 | **4,79M** |
| 0.1 | 0 | 4,93M |

A 262.144, skip 10 e quota 0.25: 3,15M.
- Oltre skip 3 il guadagno è minimo: ormai il collo sono le GPU (97–99%, circa 300 W).
- 46 worker per rank sono il massimo sensato: 184 thread su 192.

## Costo
- A 3 $/h e circa 5M pos/s fanno **circa 0,17 $ per G posizioni**, come la macchina della legio (4× 5060 Ti) ma circa 6
  volte più veloce.
- Le 8× 3090 da 1,93 $/h (Xeon E5-2673 v4, 80 thread) sono state scartate: il collo è la CPU del loader.

## Smoke test (13:24–13:27)
`train_moe1024.sh` completo in piccolo: P di 2 epoche da 52M, conversione in `.pt`, F con `--resume-from-model`,
export `.nnue`. Tutto OK. Checkpoint da 3,9 GB, reti leb128 da 161 MB.

## Lancio (28/09, 13:39:48 UTC): configurazione definitiva, scelte dell'utente
```
BATCH=524288 ROOT=/root/moe/run P_EPOCHS=450 F_EPOCHS=22 nohup /root/tt/TrainingMoE1024/train_moe1024.sh > /root/train_moe1024.log 2>&1 &
```
- **P:** 450 epoche da 1 G, batch 524.288, lr 8e-4, one-cycle su 858.306 step (warmup 5%, divisore finale 1000).
- **λ:** 1,0 con ciclo −0,3 (warmup 25%) e jitter (0,0035 per campione, 0,0070 per batch, decadimento 0,999).
- **Distribuzione per numero di pezzi:** `pc-y` −0,20 / 0,45 / 1,0 / 0,95 / 0,75.
- **Skip:** casuale **2**; per le aperture **nessuno skip duro, soft 20** (decisione dell'utente: le aperture servono
  all'esperto ≥ 24 pezzi).
- **F:** 22 epoche a batch 262.144, lr 5,66e-4, riprese da `P_final.pt`.
- **Loader:** quota 0.1, 46 worker per rank, 40 binpack; esclusi i due `test60` da 3 GB (< 5 GB, per il campionamento
  uniforme per file).
- **Export:** `.nnue` ogni 45 epoche in `/root/moe/run/nets/` (`export_nets.sh`, ogni 15 minuti).
- **Tentativi scartati prima del lancio:**
  - `run_b262k_fermato`: batch 262.144 con la ricetta SF piena, 3,28M pos/s, troppo lento per 28 ore;
  - `run_b524k_prova`;
  - `run_noearly_fermato`: senza skip delle aperture, sostituito dal soft 20.
- **A regime:** 201 s per epoca, cioè **4,97M pos/s**, GPU al 96–98%, load average circa 35 su 192. Loss:
  0,0122 all'epoca 1, 0,0104 all'epoca 2.
- **Fine prevista:** P verso le 15:15 UTC del 29/09, F verso le 16:40 UTC del 29/09 (circa 27 ore). Poi si spegne
  la VM, lo fa l'utente.

## Andamento (28/09, 13:40–21:15 UTC)
- **Velocità:** da circa 9,25 a circa 8,95 it/s (da 3:26 a 3:32 per epoca), il 3% in meno.
  - Le GPU sono al 97–99%, a 63–68 °C, con consumo tra 265 e 310 W su 575 di limite.
  - Il calo viene dal boost, sceso di circa 40 MHz, e dal limite di potenza software, che scatta a intermittenza.
  - Nessun problema di temperatura o di loader.
  - **Nuova fine prevista:** P verso le 16:20 UTC del 29/09, F con la conversione verso le 17:50 UTC.
- **Learning rate:** `OneCycleLR` di torch a coseno, picco 8e-4 verso l'epoca 22.
  - Oltre il 90% del picco fino all'epoca circa 110.
  - 84% all'epoca 135, 54% alla 225, 23% alla 315.
- **Loss per epoca:**

  | epoca | loss |
  |---|---|
  | 7 | 0,0081 |
  | 20 | 0,0047 |
  | 27 | 0,0041 |
  | 42 | 0,0038 |
  | 80 | 0,0042 |
  | 110 | 0,0044 |

  Sale perché il ciclo di λ sposta il peso sul risultato della partita.
- **Esportazioni:** ogni 45 epoche in `nets/` (`P_epoch=44`, `P_epoch=89`, …). Più il checkpoint corrente su
  richiesta, con `serialize_last.sh`, che serializza sulla CPU senza toccare le GPU (circa 3 minuti).
- **Forza contro legio-septima:** stessa ricerca 8.0 da entrambe le parti, build MinGW non-PGO, 1 thread, 64 MB, UHO
  2024. Dettaglio in `Tors3/Triumviratus` NETWORKS.md.

  | epoca | TC | partite | Elo |
  |---|---|---|---|
  | 30 | 20+0.2 | 589 | −121 ± 21 |
  | 35 | 20+0.2 | 2.000 | −113 ± 11 |
  | 44 | 20+0.2 | 2.000 | −86 ± 11 |
  | 53 | 30+0.3 | 1.774 | −76 ± 11 |
  | 64 | 30+0.3 | 1.001 | −56 ± 14 |
  | 71 | 30+0.3 | 1.012 | −70 ± 14 |
  | 80 | 30+0.3 | 2.000 | −56 ± 11 |
  | 99 | 30+0.3 | 1.159 | −44 ± 14 |
  | 110 | 30+0.3 | 2.000 | −58 ± 11 |

  La forza sale in fretta fino all'epoca circa 60, poi resta su un altopiano col learning rate al picco. Una
  previsione con la curva logaritmica (−32 all'epoca 71) è fallita: il risultato è stato −70.
- **Criteri fissati in anticipo:**
  - all'epoca 225 almeno −30, altrimenti il run va esaminato;
  - all'epoca 315 vicino alla parità;
  - il verdetto è sulla rete finale.
- **Credito vast.ai:** circa 2,98 $/h. Il 28/09 sera il saldo era di circa 57 $, che bastano fino alle 13:50 UTC circa
  del 29/09. L'utente ricarica la mattina del 29.
