# start_monitoring.ps1
$WorkDir = "C:\Users\kjfle\Workspace\ordinis"
$BinDir = "$WorkDir\bin"
$ConfigDir = "$WorkDir\configs"
$DataDir = "$WorkDir\data"
$LogDir = "$WorkDir\logs"

# Ensure data directories exist
New-Item -ItemType Directory -Force -Path "$DataDir\prometheus" | Out-Null
New-Item -ItemType Directory -Force -Path "$DataDir\loki" | Out-Null
New-Item -ItemType Directory -Force -Path "$DataDir\grafana\data" | Out-Null
New-Item -ItemType Directory -Force -Path "$DataDir\grafana\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "$DataDir\grafana\plugins" | Out-Null
New-Item -ItemType Directory -Force -Path "$DataDir\promtail" | Out-Null

Write-Host "Starting Monitoring Stack..."

# 1. Prometheus
$PromArgs = @(
    "--config.file=$ConfigDir\prometheus\prometheus.yml",
    "--storage.tsdb.path=$DataDir\prometheus",
    "--web.listen-address=0.0.0.0:3001",
    "--web.enable-lifecycle"
)
Start-Process -FilePath "$BinDir\prometheus\prometheus.exe" -ArgumentList $PromArgs -WindowStyle Minimized
Write-Host "  Started Prometheus (http://localhost:3001)"

# 2. Alertmanager
$AlertArgs = @(
    "--config.file=$ConfigDir\alertmanager\alertmanager.yml",
    "--storage.path=$DataDir\alertmanager",
    "--web.listen-address=0.0.0.0:3002"
)
Start-Process -FilePath "$BinDir\alertmanager\alertmanager.exe" -ArgumentList $AlertArgs -WindowStyle Minimized
Write-Host "  Started Alertmanager (http://localhost:3002)"

# 3. Loki
$LokiArgs = @(
    "-config.file=$ConfigDir\loki\loki-config.yml"
)
Start-Process -FilePath "$BinDir\loki\loki.exe" -ArgumentList $LokiArgs -WindowStyle Minimized
Write-Host "  Started Loki (http://localhost:3003)"

# 4. Promtail
$PromtailArgs = @(
    "-config.file=$ConfigDir\promtail\promtail-windows.yml"
)
Start-Process -FilePath "$BinDir\promtail\promtail.exe" -ArgumentList $PromtailArgs -WindowStyle Minimized
Write-Host "  Started Promtail (http://localhost:3004)"

# 5. Grafana
# Generate Grafana custom.ini to ensure paths are correctly picked up
$GrafanaIniContent = @"
[paths]
data = $DataDir\grafana\data
logs = $DataDir\grafana\logs
plugins = $DataDir\grafana\plugins
provisioning = $ConfigDir\grafana\provisioning

[server]
http_port = 3010

[security]
admin_user = admin
admin_password = admin
"@
Set-Content -Path "$ConfigDir\grafana\custom.ini" -Value $GrafanaIniContent

# We need to set working directory for Grafana or it complains about defaults
$GrafanaBin = "$BinDir\grafana\bin\grafana-server.exe"
$GrafanaArgs = @(
    "-config", "$ConfigDir\grafana\custom.ini"
)
Start-Process -FilePath $GrafanaBin -ArgumentList $GrafanaArgs -WorkingDirectory "$BinDir\grafana\bin" -WindowStyle Minimized
Write-Host "  Started Grafana (http://localhost:3010)"

Write-Host "`nAll services started in background windows."
Write-Host "Run 'python scripts/trading/live_trading_production.py' to start the trading bot."
