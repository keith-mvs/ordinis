# Start Live Paper Trading with Massive WebSocket + Alpaca
# Loads credentials from user environment variables

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "LIVE PAPER TRADING - Massive WebSocket + Alpaca" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Load Massive/Polygon credentials
$env:MASSIVE_API_KEY = [System.Environment]::GetEnvironmentVariable('MASSIVE_API_KEY', 'User')
$env:POLYGON_API_KEY = [System.Environment]::GetEnvironmentVariable('POLYGON_API_KEY', 'User')

# Load Alpaca credentials
$env:APCA_API_KEY_ID = [System.Environment]::GetEnvironmentVariable('APCA_API_KEY_ID', 'User')
$env:APCA_API_SECRET_KEY = [System.Environment]::GetEnvironmentVariable('APCA_API_SECRET_KEY', 'User')

# Validate credentials
$credentialsOk = $true

Write-Host ""
Write-Host "Credential Check:" -ForegroundColor Yellow

if ($env:MASSIVE_API_KEY) {
    Write-Host "  [OK] MASSIVE_API_KEY: $($env:MASSIVE_API_KEY.Substring(0,8))..." -ForegroundColor Green
} else {
    Write-Host "  [MISSING] MASSIVE_API_KEY" -ForegroundColor Red
    $credentialsOk = $false
}

if ($env:APCA_API_KEY_ID) {
    Write-Host "  [OK] APCA_API_KEY_ID: $($env:APCA_API_KEY_ID.Substring(0,8))..." -ForegroundColor Green
} else {
    Write-Host "  [MISSING] APCA_API_KEY_ID" -ForegroundColor Red
    $credentialsOk = $false
}

if ($env:APCA_API_SECRET_KEY) {
    Write-Host "  [OK] APCA_API_SECRET_KEY: [SET]" -ForegroundColor Green
} else {
    Write-Host "  [MISSING] APCA_API_SECRET_KEY" -ForegroundColor Red
    $credentialsOk = $false
}

Write-Host ""

if (-not $credentialsOk) {
    Write-Host "ERROR: Missing required credentials. Cannot start." -ForegroundColor Red
    Write-Host ""
    Write-Host "Set credentials with:" -ForegroundColor Yellow
    Write-Host '  [System.Environment]::SetEnvironmentVariable("MASSIVE_API_KEY", "your_key", "User")' -ForegroundColor Cyan
    Write-Host '  [System.Environment]::SetEnvironmentVariable("APCA_API_KEY_ID", "your_key", "User")' -ForegroundColor Cyan
    Write-Host '  [System.Environment]::SetEnvironmentVariable("APCA_API_SECRET_KEY", "your_secret", "User")' -ForegroundColor Cyan
    exit 1
}

# Run the live paper trading script
Set-Location $PSScriptRoot\..\..
$pythonPath = "$env:USERPROFILE\.conda\envs\ordinis-dev-1\python.exe"

Write-Host "Starting live paper trading..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

& $pythonPath scripts\trading\live_paper_trading.py
