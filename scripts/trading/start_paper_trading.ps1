# Start Paper Trading with Alpaca
# Loads credentials from user environment variables

# Load Alpaca credentials from user environment
$env:APCA_API_KEY_ID = [System.Environment]::GetEnvironmentVariable('APCA_API_KEY_ID', 'User')
$env:APCA_API_SECRET_KEY = [System.Environment]::GetEnvironmentVariable('APCA_API_SECRET_KEY', 'User')

Set-Location $PSScriptRoot\..\..
$pythonPath = "$env:USERPROFILE\.conda\envs\ordinis-dev-1\python.exe"

# Check if both credentials are set for Alpaca
if ($env:APCA_API_KEY_ID -and $env:APCA_API_SECRET_KEY) {
    Write-Host "Alpaca API Key: $($env:APCA_API_KEY_ID.Substring(0,8))..." -ForegroundColor Green
    Write-Host "Starting Alpaca paper trading..." -ForegroundColor Cyan
    & $pythonPath scripts\trading\paper_trading_runner.py --broker alpaca --config configs\strategies\atr_optimized_rsi.yaml
} else {
    Write-Host "Alpaca credentials incomplete. Starting simulated broker..." -ForegroundColor Yellow
    & $pythonPath scripts\trading\paper_trading_runner.py --broker simulated --config configs\strategies\atr_optimized_rsi.yaml
}
