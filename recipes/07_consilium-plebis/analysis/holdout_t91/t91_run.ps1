# t91 holdout loss, Consilium against Consilium Plebis exports (25/10/2026). One python process per network: the
# Windows loader fails at stream creation in some processes (access violation, every in-process retry fails too) and
# works in others, so a failed process is simply started again. Every process reads the same positions, so the
# per-batch dumps give paired differences (t91_paired.py).
# 26/10: -Out (dump folder) and -Extra (more arguments, e.g. --by-bucket).
param([int]$Batches = 300, [int]$Threads = 16, [string[]]$Nets = @(), [string]$Done = "FINE",
      [string]$Out = "<work>\c2eng\scale\t91_2026-10-25", [string]$Extra = "")
# Repository copy: $env:TRIUMV is the author's project folder (trainer venv, t91 sample, 8.1 engine folder);
# <work>\c2\nnue-pytorch holds the Consilium Plebis trainer (patches/consilium2_trainer_vs_upstream_9f72946.patch).
Set-Location <work>\c2\nnue-pytorch
$py = "$env:TRIUMV\Training_NNUE\localcpu\venv\Scripts\python.exe"
$d = "$env:TRIUMV\Training_NNUE\localcpu\data\t91_sample.binpack"
$out = $Out
$default = @("$env:TRIUMV\Triumviratus_8.1\nn-consilium.nnue",
          "<work>\c2\nets\c2_P_ep376_perm.nnue", "<work>\c2\nets\c2_P_ep210.nnue", "<work>\c2\nets\c2_P_ep246.nnue",
          "<work>\c2\nets\c2_P_ep283_perm.nnue", "<work>\c2\nets\c2_P_ep310_perm.nnue",
          "<work>\c2\nets\c2_P_ep342_perm.nnue")
# A second worker on other networks (same dump folder). PowerShell names are case-insensitive: $Nets is the parameter.
# Two workers on the same list skip each other's running network (its log is locked: 20 instant failures).
# With `powershell -File` a list arrives as ONE string "A,B": split it on the commas.
if ($Nets.Count -eq 1 -and $Nets[0] -match ',') { $Nets = $Nets[0] -split ',' }
$list = if ($Nets.Count -gt 0) { $Nets } else { $default }
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($net in $list) {
    $name = [IO.Path]::GetFileNameWithoutExtension($net)
    for ($try = 1; $try -le 20; $try++) {
        $t = Get-Date
        & cmd /c "$py c2_val_loss_fixed2.py $d $net --batches $Batches --threads $Threads --dump $out\$name.txt $Extra > $out\$name.log 2>&1"
        $ok = Select-String -Path "$out\$name.log" -Pattern "batches x" -Quiet
        $line = "{0} {1} attempt {2}: {3} ({4:N0} s)" -f (Get-Date -Format HH:mm:ss), $name, $try,
                $(if ($ok) { "OK" } else { "stream failed" }), ((Get-Date) - $t).TotalSeconds
        Write-Host $line
        Add-Content "$out\progress.txt" $line
        if ($ok) { break }
    }
}
Write-Host $Done
$Done | Set-Content "$out\$Done"
