# Analyze trading logs from today
$logFile = 'C:\Users\kjfle\Workspace\ordinis\logs\live_paper_trading.log'
$signalsFile = 'C:\Users\kjfle\Workspace\ordinis\logs\signals.log'

Write-Host "=============================================="
Write-Host "TRADING DAY ANALYSIS - December 17, 2025"
Write-Host "=============================================="
Write-Host ""

# Count various events
$signals = (Select-String -Path $logFile -Pattern 'SIGNAL \|' -SimpleMatch).Count
$pendingNewErrors = (Select-String -Path $logFile -Pattern 'pending_new').Count
$insuffBuyingPower = (Select-String -Path $logFile -Pattern 'insufficient buying power').Count
$quantity0 = (Select-String -Path $logFile -Pattern 'calculated quantity is 0').Count
$successfulLong = (Select-String -Path $logFile -Pattern 'LONG [A-Z]+: \d+ shares').Count
$authErrors = (Select-String -Path $logFile -Pattern 'Unexpected auth response').Count
$reconnects = (Select-String -Path $logFile -Pattern 'RECONNECTING').Count

Write-Host "=== SIGNAL STATISTICS ==="
Write-Host "Total Signals Generated: $signals"
Write-Host ""

Write-Host "=== EXECUTION RESULTS ==="
Write-Host "Successful Orders: $successfulLong"
Write-Host ""

Write-Host "=== ERRORS ENCOUNTERED ==="
Write-Host "pending_new OrderStatus errors: $pendingNewErrors"
Write-Host "Insufficient buying power: $insuffBuyingPower"
Write-Host "Quantity = 0 (position sizing): $quantity0"
Write-Host "WebSocket auth errors: $authErrors"
Write-Host "Reconnection attempts: $reconnects"
Write-Host ""

# Get unique symbols that generated signals
Write-Host "=== TOP SIGNAL GENERATORS ==="
$signalLines = Get-Content $signalsFile
$symbolCounts = @{}
foreach ($line in $signalLines) {
    if ($line -match 'SIGNAL \| ([A-Z]+) \|') {
        $symbol = $Matches[1]
        if ($symbolCounts.ContainsKey($symbol)) {
            $symbolCounts[$symbol]++
        } else {
            $symbolCounts[$symbol] = 1
        }
    }
}
$sorted = $symbolCounts.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 15
foreach ($item in $sorted) {
    Write-Host "$($item.Key): $($item.Value) signals"
}
Write-Host ""

# Get equity progression from status lines
Write-Host "=== EQUITY PROGRESSION ==="
$statusLines = Select-String -Path $logFile -Pattern 'Status: Equity=\$([0-9.]+)' | Select-Object -First 1, -Last 1
if ($statusLines) {
    $first = $statusLines[0] -match 'Equity=\$([0-9.]+)'
    $firstEquity = if ($Matches) { $Matches[1] } else { "N/A" }
    $last = $statusLines[-1] -match 'Equity=\$([0-9.]+)'
    $lastEquity = if ($Matches) { $Matches[1] } else { "N/A" }
    Write-Host "Starting Equity: $firstEquity"
    Write-Host "Ending Equity: $lastEquity"
}
