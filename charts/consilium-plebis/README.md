# Consilium Plebis: charts of the training run

Consilium Plebis is the network after Consilium, in training since 21/10/2026 (pretraining P in epochs of one
billion positions, then a finetune F). Its architecture is described in the engine repository
([`docs/networks/NETWORKS.md`](https://github.com/Tors3/Triumviratus/blob/main/docs/networks/NETWORKS.md)). The network
file, the recipe and the scripts will be published here with the engine release that ships it; until then, these
charts follow the run. Updated 27/10/2026: P was stopped after epoch 496 of the 550 planned, while the loss was
still falling, and F started from it; the training-loss charts cover P (epochs 0-496).

## Training loss

| chart | what it shows |
|---|---|
| [`c2_loss_per_epoch.png`](c2_loss_per_epoch.png) | training loss per epoch over the whole run, log scale, with the end of the warmup and the point where lambda stops at 0.75 |
| [`c2_loss_lr_lambda.png`](c2_loss_lr_lambda.png) | the same loss from epoch 25 on a linear scale, with the learning rate (% of its peak) and lambda on the same epoch axis |
| [`c2_loss_vs_lr.png`](c2_loss_vs_lr.png) | the loss against the learning rate after the warmup |
| [`c1_c2_training.png`](c1_c2_training.png) | the pretraining of Consilium and of Consilium Plebis on the same epoch axis, with lambda and learning rate of each: different lambdas make the two training losses not comparable in absolute value |

## Loss on held-out positions

`t91` is a sample of Leela Chess Zero test91 games that neither network trained on. Every network reads the same
positions, so the differences to Consilium are paired.

| chart | what it shows |
|---|---|
| [`c1_c2_holdout.png`](c1_c2_holdout.png) | loss of the exports relative to Consilium (the 8.0 release network) at lambda 1.0 (evaluation), 0.75 (training target) and 0 (game result): Consilium Plebis passed Consilium near epoch 240 and is 3% below at epoch 435 |
| [`c2_t91_by_bucket.png`](c2_t91_by_bucket.png) | epoch 435 against Consilium by number of pieces: far ahead in the endgames, less in the middlegame |
| [`c2_t91_by_expert.png`](c2_t91_by_expert.png) | epochs 376 and 435 against Consilium by expert (the opponent's non-pawn material): every expert still improves at low learning rate, the middlegame experts most |

## Evaluation scale

| chart | what it shows |
|---|---|
| [`scale_trend.png`](scale_trend.png) | spread of the search evaluation of 41 exports against Consilium's on 49,594 positions, and the side-to-move offset |

## Strength

| chart | what it shows |
|---|---|
| [`consilium2_elo_trajectory.png`](consilium2_elo_trajectory.png) | Elo of the exports against the final Consilium at fixed nodes, same search |
| [`consilium_compare_trajectory.png`](consilium_compare_trajectory.png) | Consilium and Consilium Plebis across training, each against the network it replaces |
| [`consilium_transpose_fit.png`](consilium_transpose_fit.png) | the Consilium Plebis points matched to a fit of Consilium's curve |

Every chart is also in SVG.
