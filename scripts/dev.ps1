# Run the phorse API and web dev server together (Windows / PowerShell).
#   pwsh scripts\dev.ps1
# Ctrl-C stops the web server; the API is stopped on the way out.

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
    # Local dev: no invite needed -- sign up at /join/<anything>.
    if (-not $env:HORSE_REQUIRE_INVITE) { $env:HORSE_REQUIRE_INVITE = '0' }

    Write-Host 'starting API on http://127.0.0.1:8000 ...' -ForegroundColor Cyan
    $api = Start-Process python -ArgumentList '-m', 'server.wsgi' -PassThru -NoNewWindow

    $interval = if ($env:HORSE_RETRAIN_INTERVAL) { $env:HORSE_RETRAIN_INTERVAL } else { '600' }
    Write-Host "starting retrain scheduler (HORSE_RETRAIN_INTERVAL=${interval}s) ..." -ForegroundColor Cyan
    $scheduler = Start-Process python -ArgumentList '-m', 'server.scheduler' -PassThru -NoNewWindow

    Write-Host 'starting web on http://localhost:5173 ...' -ForegroundColor Green
    # npm.cmd, not npm — the npm.ps1 shim is blocked by execution policy on many setups
    npm.cmd --prefix web run dev
}
finally {
    foreach ($proc in $api, $scheduler) {
        if ($proc -and -not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Pop-Location
}
