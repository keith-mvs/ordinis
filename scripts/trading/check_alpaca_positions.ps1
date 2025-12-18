# Check Alpaca positions
$key = [System.Environment]::GetEnvironmentVariable('APCA_API_KEY_ID', 'User')
$secret = [System.Environment]::GetEnvironmentVariable('APCA_API_SECRET_KEY', 'User')

$headers = @{
    'APCA-API-KEY-ID' = $key
    'APCA-API-SECRET-KEY' = $secret
}

# Get positions
$positions = Invoke-RestMethod -Uri 'https://paper-api.alpaca.markets/v2/positions' -Headers $headers

Write-Host "=== OPEN POSITIONS ==="
Write-Host "Count: $($positions.Count)"
Write-Host ""

foreach ($pos in $positions) {
    $pnl = [math]::Round([double]$pos.unrealized_pl, 2)
    $pnlPct = [math]::Round([double]$pos.unrealized_plpc * 100, 2)
    $value = [math]::Round([double]$pos.market_value, 2)
    Write-Host "$($pos.symbol): $($pos.qty) shares @ $($pos.avg_entry_price) | Value: $value | P&L: $pnl ($pnlPct%)"
}

Write-Host ""
Write-Host "Total Market Value: $($positions | ForEach-Object { [double]$_.market_value } | Measure-Object -Sum | Select-Object -ExpandProperty Sum)"
