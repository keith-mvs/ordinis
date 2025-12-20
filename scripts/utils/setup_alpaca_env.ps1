# Alpaca Paper Trading Setup Script
# Run this before starting paper trading: . .\setup_alpaca_env.ps1

# Load Alpaca credentials from user environment variables
$env:ALPACA_API_KEY = [System.Environment]::GetEnvironmentVariable('ALPACA_API_KEY', 'User')
$env:ALPACA_SECRET_KEY = [System.Environment]::GetEnvironmentVariable('ALPACA_SECRET_KEY', 'User')
$env:ALPACA_BASE_URL = [System.Environment]::GetEnvironmentVariable('ALPACA_BASE_URL', 'User')

# Verify they're set
if ($env:ALPACA_API_KEY -and $env:ALPACA_SECRET_KEY) {
    Write-Host "✅ Alpaca credentials loaded successfully" -ForegroundColor Green
    Write-Host "   API Key: $($env:ALPACA_API_KEY.Substring(0, [Math]::Min(8, $env:ALPACA_API_KEY.Length)))..." -ForegroundColor Gray
    Write-Host "   Secret: $($env:ALPACA_SECRET_KEY.Substring(0, [Math]::Min(8, $env:ALPACA_SECRET_KEY.Length)))..." -ForegroundColor Gray
    Write-Host "   Base URL: $env:ALPACA_BASE_URL" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Ready to run:" -ForegroundColor Cyan
    Write-Host "  python scripts/trading/test_alpaca_connection.py" -ForegroundColor Yellow
    Write-Host "  python scripts/trading/run_live_alpaca.py" -ForegroundColor Yellow
} else {
    Write-Host "❌ Alpaca credentials not found in user environment variables!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Set them permanently using:" -ForegroundColor Yellow
    Write-Host '  [System.Environment]::SetEnvironmentVariable("ALPACA_API_KEY", "your_key", "User")' -ForegroundColor Gray
    Write-Host '  [System.Environment]::SetEnvironmentVariable("ALPACA_SECRET_KEY", "your_secret", "User")' -ForegroundColor Gray
    Write-Host '  [System.Environment]::SetEnvironmentVariable("ALPACA_BASE_URL", "https://paper-api.alpaca.markets", "User")' -ForegroundColor Gray
}
