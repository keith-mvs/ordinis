# Alpaca Credential Update Script
# Run this to update your Alpaca API credentials

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "ALPACA CREDENTIAL UPDATE" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

Write-Host "INSTRUCTIONS:" -ForegroundColor Yellow
Write-Host "1. Go to: https://app.alpaca.markets/paper/dashboard/overview" -ForegroundColor Gray
Write-Host "2. Navigate to 'Your API Keys' or 'API Keys' section" -ForegroundColor Gray
Write-Host "3. Click 'Regenerate' or 'Generate New Key Pair'" -ForegroundColor Gray
Write-Host "4. Copy BOTH the new API Key and Secret Key" -ForegroundColor Gray
Write-Host ""

Write-Host "Then run these commands (replace with YOUR new keys):" -ForegroundColor Yellow
Write-Host ""
Write-Host '[System.Environment]::SetEnvironmentVariable("ALPACA_API_KEY", "PK_YOUR_NEW_KEY_HERE", "User")' -ForegroundColor Green
Write-Host '[System.Environment]::SetEnvironmentVariable("ALPACA_SECRET_KEY", "YOUR_NEW_SECRET_HERE", "User")' -ForegroundColor Green
Write-Host ""

Write-Host "After updating, reload them:" -ForegroundColor Yellow
Write-Host ""
Write-Host '. .\setup_alpaca_env.ps1' -ForegroundColor Green
Write-Host ""

Write-Host "Then test the connection:" -ForegroundColor Yellow
Write-Host ""
Write-Host 'python scripts/trading/diagnose_alpaca.py' -ForegroundColor Green
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
