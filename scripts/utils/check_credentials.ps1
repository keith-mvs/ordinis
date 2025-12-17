# Update Alpaca credentials
# Replace YOUR_NEW_API_KEY and YOUR_NEW_SECRET_KEY with your actual credentials

Write-Host "Current credentials in user environment:" -ForegroundColor Cyan
$currentKey = [System.Environment]::GetEnvironmentVariable('ALPACA_API_KEY', 'User')
Write-Host "  API Key: $($currentKey.Substring(0, [Math]::Min(10, $currentKey.Length)))..."

Write-Host "`nTo update with NEW credentials, run these commands:" -ForegroundColor Yellow
Write-Host '  [System.Environment]::SetEnvironmentVariable("ALPACA_API_KEY", "YOUR_NEW_KEY", "User")' -ForegroundColor Gray
Write-Host '  [System.Environment]::SetEnvironmentVariable("ALPACA_SECRET_KEY", "YOUR_NEW_SECRET", "User")' -ForegroundColor Gray

Write-Host "`nThen reload in current session:" -ForegroundColor Yellow
Write-Host '  $env:ALPACA_API_KEY = [System.Environment]::GetEnvironmentVariable("ALPACA_API_KEY", "User")' -ForegroundColor Gray
Write-Host '  $env:ALPACA_SECRET_KEY = [System.Environment]::GetEnvironmentVariable("ALPACA_SECRET_KEY", "User")' -ForegroundColor Gray
