@echo off
REM Consilium-2 against Consilium (21/10/2026): inference cost on a fixed tree, two PGO builds of the same source.
REM Run with the Xeon idle (no SPSA/SPRT): engine pinned to node 0, processor 2; 4 rounds, order alternated.
REM Live progress here; everything is also appended to nnspeed_results.txt.
title nnspeed measure, C2 against Consilium
cd /d <work>\c2pgo_misura
python measure_nnspeed.py --rounds 4 --pin 0:2 --depth 4
pause
