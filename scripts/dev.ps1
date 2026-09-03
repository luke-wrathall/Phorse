# Run the phorse API and web dev server together (Windows / PowerShell).
#   pwsh scripts\dev.ps1
# Ctrl-C stops the web server; the API is stopped on the way out.

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
    Write-Host 'starting API on http://127.0.0.1:8000 ...' -ForegroundColor Cyan
    $api = Start-Process python -ArgumentList '-m', 'server.wsgi' -PassThru -NoNewWindow

    Write-Host 'starting web on http://localhost:5173 ...' -ForegroundColor Green
    # npm.cmd, not npm — the npm.ps1 shim is blocked by execution policy on many setups
    npm.cmd --prefix web run dev
}
finally {
    if ($api -and -not $api.HasExited) {
        Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}
