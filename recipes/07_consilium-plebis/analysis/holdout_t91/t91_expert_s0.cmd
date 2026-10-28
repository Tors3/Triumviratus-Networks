@echo off
rem t91 per expert (26/10/2026): socket 0, one thread per core (20), Consilium then ep376.
start "t91 expert s0" /NODE 0 /AFFINITY 5555555555 powershell -NoProfile -ExecutionPolicy Bypass -File <work>\c2eng\scale\t91_run.ps1 -Nets "%TRIUMV%\Triumviratus_8.1\nn-consilium.nnue,<work>\c2\nets\c2_P_ep376_perm.nnue" -Threads 20 -Done FINE_S0 -Out <work>\c2eng\scale\t91_2026-10-26_expert -Extra "--npz-dir <work>\c2eng\scale\t91_2026-10-26_expert"
