# Partite rete MoE-1024 (8.0) contro legio-septima, STESSA ricerca 8.0 (28/09/2026).
# Le due build vengono dagli stessi sorgenti Triumviratus_8.0 con lo stesso compilatore (MinGW, non PGO): cambia solo
# lo strato d'ingresso (-DTRIUMV_PSQ_PHASES=4) e la rete. Il confronto isola la rete.
#
# USO:  .\match_moe.ps1 -Net nets\P_epoch=44-step=....nnue            (default 10+0.1, 1000 round = 2000 partite)
#       .\match_moe.ps1 -Net nets\X.nnue -TC 20+0.2 -Rounds 1500 -Concurrency 40
# La rete si scarica dalla VM con .\fetch_net.ps1 (mette i .nnue in .\nets).
param([Parameter(Mandatory = $true)][string]$Net,
      [string]$TC = "10+0.1",
      [int]$Rounds = 1000,
      [int]$Concurrency = 75,
      [int]$Hash = 64)
$here = $PSScriptRoot
$root = Split-Path $here
$legio = "$here\build_legio\t80_legio.exe"
$moe = "$here\build_moe\t80_moe.exe"
$netPath = if ([System.IO.Path]::IsPathRooted($Net)) { $Net } else { "$here\$Net" }
if (-not (Test-Path $netPath)) { throw "rete non trovata: $netPath" }
# il binario MoE carica nn-legio-septima.nnue accanto a se' (nome predefinito del motore): ci si copia la rete MoE
Copy-Item $netPath "$here\build_moe\nn-legio-septima.nnue" -Force
# Via cmd: PowerShell 5.1 che scrive su stdin di un exe premette un BOM e il motore non riconosce "bench".
$b1 = "" + (cmd /c "(echo bench& echo quit) | `"$legio`"" | Select-String "Nodes searched")
if ($b1 -notmatch "273477") { throw "legio: bench diverso da 273477: $b1" }
$b2 = "" + (cmd /c "(echo bench& echo quit) | `"$moe`"" | Select-String "Nodes searched")
if (-not $b2) { throw "il binario MoE non parte con questa rete (formato o hash sbagliati?)" }
$tag = [System.IO.Path]::GetFileNameWithoutExtension($netPath) -replace '[^A-Za-z0-9_-]', '_'
$fcArgs = @("-engine", "cmd=$moe", "name=moe_$tag",
    "-engine", "cmd=$legio", "name=legio",
    "-each", "tc=$TC", "option.Hash=$Hash", "option.Threads=1",
    "-openings", "file=$root\OpeningBooks\uho_2024\UHO_2024_+085_+094\UHO_2024_8mvs_+085_+094.epd", "format=epd", "order=random",
    "-draw", "movenumber=40", "movecount=8", "score=10",
    "-resign", "movecount=3", "score=600", "twosided=true",
    "-rounds", "$Rounds", "-games", "2", "-repeat", "-concurrency", "$Concurrency", "-force-concurrency",
    "-ratinginterval", "20", "-pgnout", "file=$here\moe_$tag.pgn", "-log", "file=$here\moe_$tag.log")
Write-Host "moe: $b2   legio: $b1   TC $TC, $Rounds round, concurrency $Concurrency"
Start-Process "$root\Fastechess_For_SPSA\fastchess.exe" -ArgumentList $fcArgs -WorkingDirectory $here
