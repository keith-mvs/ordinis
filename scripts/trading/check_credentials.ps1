# Check Alpaca credentials
$apiKey = [System.Environment]::GetEnvironmentVariable('APCA_API_KEY_ID', 'User')
$secretKey = [System.Environment]::GetEnvironmentVariable('APCA_API_SECRET_KEY', 'User')

Write-Host "Alpaca Credentials Check:"
if ($apiKey) {
    Write-Host "  API Key: $($apiKey.Substring(0,8))..." -ForegroundColor Green
} else {
    Write-Host "  API Key: NOT SET" -ForegroundColor Red
}

if ($secretKey) {
    Write-Host "  Secret Key: [SET - hidden]" -ForegroundColor Green
} else {
    Write-Host "  Secret Key: NOT SET" -ForegroundColor Red
    Write-Host ""
    Write-Host "To set your Alpaca credentials, run:" -ForegroundColor Yellow
    Write-Host '  [System.Environment]::SetEnvironmentVariable("APCA_API_KEY_ID", "your_key", "User")' -ForegroundColor Cyan
    Write-Host '  [System.Environment]::SetEnvironmentVariable("APCA_API_SECRET_KEY", "your_secret", "User")' -ForegroundColor Cyan
}
