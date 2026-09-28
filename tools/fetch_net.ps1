# Scarica dalla VM i .nnue esportati dal training MoE-1024 (/root/moe/run/nets) in .\nets, solo quelli nuovi.
# Con -Last serializza prima il last.ckpt corrente (copia stabile, CPU della VM, GPU non toccate) e scarica anche quello.
# USO:  .\fetch_net.ps1 -VmHost <host> -Port <porta> [-Key <chiave ssh>] [-Last]
param([Parameter(Mandatory = $true)][string]$VmHost,
      [Parameter(Mandatory = $true)][int]$Port,
      [string]$Key = "$env:USERPROFILE\.ssh\id_ed25519",
      [switch]$Last)
$here = $PSScriptRoot
$vm = @("-i", $Key, "-P", "$Port", "-o", "BatchMode=yes")
$ssh = @("-i", $Key, "-p", "$Port", "-o", "BatchMode=yes", "root@$VmHost")
New-Item -ItemType Directory -Force -Path "$here\nets" | Out-Null
if ($Last) {
    # Lo script remoto sta nel repo del training (TrainingMoE1024/serialize_last.sh): PowerShell 5.1 perde le
    # virgolette annidate passando una riga di comando lunga a ssh, quindi niente comandi inline.
    & ssh @ssh 'bash /root/tt/TrainingMoE1024/serialize_last.sh'
}
$list = & ssh @ssh 'ls /root/moe/run/nets/*.nnue 2>/dev/null'
foreach ($r in $list) {
    $name = Split-Path $r -Leaf
    if (Test-Path "$here\nets\$name") { continue }
    Write-Host "scarico $name"
    & scp @vm "root@${VmHost}:$r" "$here\nets\$name"
}
Get-ChildItem "$here\nets\*.nnue" | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
