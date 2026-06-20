$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Paper = Join-Path $Root "paper"
$DownloadsPdf = Join-Path $HOME "Downloads\66.pdf"

Push-Location $Root
try {
    python scripts\render_submission_assets.py
    Push-Location $Paper
    try {
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        bibtex main
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
    }
    finally {
        Pop-Location
    }
    Copy-Item -LiteralPath (Join-Path $Paper "main.pdf") -Destination $DownloadsPdf -Force
    Write-Output "Wrote $DownloadsPdf"
}
finally {
    Pop-Location
}
