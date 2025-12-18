# Check Massive/Polygon credentials
Write-Host "Massive API Credentials Check:" -ForegroundColor Cyan

$envVars = [System.Environment]::GetEnvironmentVariables('User')
$massiveKeys = $envVars.Keys | Where-Object { $_ -like 'MASSIVE*' -or $_ -like 'POLYGON*' }

if ($massiveKeys.Count -eq 0) {
    Write-Host "  No MASSIVE* or POLYGON* env vars found" -ForegroundColor Yellow
} else {
    foreach ($key in $massiveKeys) {
        $value = $envVars[$key]
        if ($value.Length -gt 8) {
            Write-Host "  $key : $($value.Substring(0,8))..." -ForegroundColor Green
        } else {
            Write-Host "  $key : [SET]" -ForegroundColor Green
        }
    }
}
