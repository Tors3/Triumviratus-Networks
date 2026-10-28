# Scale measurement for a list of Consilium Plebis exports (25/10/2026): evalbatch of all.fen with the match engine
# c2ep19.exe through EvalFile, then scale_c2.py and parts.py against c1.eng. A permuted export (_perm) evaluates
# byte for byte like the plain one, so it stands in for its epoch.
# Repository copy: $env:TRIUMV is the author's project folder (trainer venv). Under `powershell -File` the outputs are
# UTF-16: convert them to UTF-8 before scale_trend.py, which skips files it cannot parse (26/10/2026).
param([string[]]$Nets)
Set-Location <work>\c2eng\scale
$py = "$env:TRIUMV\Training_NNUE\localcpu\venv\Scripts\python.exe"
foreach ($n in $Nets) {
    $net = "<work>\c2\nets\c2_P_$n.nnue"
    $ep = ($n -replace '_perm', '')
    $fwd = $net -replace '\\', '/'
    Set-Content -Path "in_scale_$ep.txt" -Value "setoption name EvalFile value $fwd", "isready",
        "evalbatch <work>/c2eng/scale/all.fen <work>/c2eng/scale/c2_$ep.eng", "quit" -Encoding ascii
    $o = cmd /c "<work>\match_c2ep19\c2ep19.exe < in_scale_$ep.txt 2>&1"
    $line = ($o | Select-String "evalbatch").Line
    & $py scale_c2.py analyze all.fen "c2_$ep.eng" c1.eng $net > "scale_$ep.txt" 2>&1
    & $py parts.py all.fen "c2_$ep.eng" c1.eng > "parts_$ep.txt" 2>&1
    "$n -> $ep : $line"
}
