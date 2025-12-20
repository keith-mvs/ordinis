# Force reset Alpaca paper trading account
$key = [System.Environment]::GetEnvironmentVariable('APCA_API_KEY_ID', 'User')
$secret = [System.Environment]::GetEnvironmentVariable('APCA_API_SECRET_KEY', 'User')

$headers = @{
    'APCA-API-KEY-ID' = $key
    'APCA-API-SECRET-KEY' = $secret
    'Content-Type' = 'application/json'
}

Write-Host "Force resetting paper account..."

# Try the account reset endpoint (POST to /account/activities with reset action)
try {
    $result = Invoke-WebRequest -Uri 'https://paper-api.alpaca.markets/v2/account' -Headers $headers -Method Delete
    Write-Host "Reset response: $($result.StatusCode)"
} catch {
    Write-Host "Reset endpoint error: $($_.Exception.Message)"
}

Start-Sleep -Seconds 3

# Check account
$account = Invoke-RestMethod -Uri 'https://paper-api.alpaca.markets/v2/account' -Headers $headers
Write-Host ""
Write-Host "=== ACCOUNT STATUS ==="
Write-Host "Equity: $($account.equity)"
Write-Host "Buying Power: $($account.buying_power)"
Write-Host "Cash: $($account.cash)"
Write-Host "Status: $($account.status)"

# Check positions
$positions = Invoke-RestMethod -Uri 'https://paper-api.alpaca.markets/v2/positions' -Headers $headers
Write-Host "Positions: $($positions.Count)"
