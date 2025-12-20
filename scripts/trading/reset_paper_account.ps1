# Reset Alpaca paper trading account
$key = [System.Environment]::GetEnvironmentVariable('APCA_API_KEY_ID', 'User')
$secret = [System.Environment]::GetEnvironmentVariable('APCA_API_SECRET_KEY', 'User')

$headers = @{
    'APCA-API-KEY-ID' = $key
    'APCA-API-SECRET-KEY' = $secret
}

Write-Host "Resetting Alpaca paper trading account..."

# Close all positions first
Write-Host "Closing all positions..."
try {
    Invoke-RestMethod -Uri 'https://paper-api.alpaca.markets/v2/positions?cancel_orders=true' -Headers $headers -Method Delete
    Write-Host "All positions closed."
} catch {
    Write-Host "Error closing positions: $_"
}

Start-Sleep -Seconds 2

# Reset account to default
Write-Host "Resetting account to $100,000..."
try {
    $body = @{ reset_to_amount = "100000" } | ConvertTo-Json
    $result = Invoke-RestMethod -Uri 'https://paper-api.alpaca.markets/v2/account/configurations' -Headers $headers -Method Patch -Body $body -ContentType 'application/json'
} catch {
    # If that doesn't work, try the account reset endpoint
    try {
        Invoke-RestMethod -Uri 'https://paper-api.alpaca.markets/v2/account' -Headers $headers -Method Delete
        Write-Host "Account reset initiated."
    } catch {
        Write-Host "Note: Full reset may require dashboard. Positions have been closed."
    }
}

Start-Sleep -Seconds 2

# Check final account status
$account = Invoke-RestMethod -Uri 'https://paper-api.alpaca.markets/v2/account' -Headers $headers
Write-Host ""
Write-Host "=== ACCOUNT STATUS ==="
Write-Host "Equity: $($account.equity)"
Write-Host "Buying Power: $($account.buying_power)"
Write-Host "Cash: $($account.cash)"
