# Check Alpaca account status
$key = [System.Environment]::GetEnvironmentVariable('APCA_API_KEY_ID', 'User')
$secret = [System.Environment]::GetEnvironmentVariable('APCA_API_SECRET_KEY', 'User')

$headers = @{
    'APCA-API-KEY-ID' = $key
    'APCA-API-SECRET-KEY' = $secret
}

# Get account
$account = Invoke-RestMethod -Uri 'https://paper-api.alpaca.markets/v2/account' -Headers $headers
Write-Host "Equity: $($account.equity)"
Write-Host "Buying Power: $($account.buying_power)"
Write-Host "Cash: $($account.cash)"

# Get open orders
$orders = Invoke-RestMethod -Uri 'https://paper-api.alpaca.markets/v2/orders?status=open' -Headers $headers
Write-Host "Open Orders: $($orders.Count)"

# Cancel all open orders if any
if ($orders.Count -gt 0) {
    Write-Host "Cancelling all open orders..."
    Invoke-RestMethod -Uri 'https://paper-api.alpaca.markets/v2/orders' -Headers $headers -Method Delete
    Write-Host "Done"
}
