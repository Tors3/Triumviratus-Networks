@echo off
rem t91 per expert (26/10/2026): socket 1, one thread per core (20), ep435.
start "t91 expert s1" /NODE 1 /AFFINITY 5555555555 powershell -NoProfile -ExecutionPolicy Bypass -File <work>\c2eng\scale\t91_run.ps1 -Nets "<work>\c2\nets\c2_P_ep435.nnue" -Threads 20 -Done FINE_S1 -Out <work>\c2eng\scale\t91_2026-10-26_expert -Extra "--npz-dir <work>\c2eng\scale\t91_2026-10-26_expert"
