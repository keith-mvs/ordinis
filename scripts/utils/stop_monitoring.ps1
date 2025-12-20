# stop_monitoring.ps1
Write-Host "Stopping Monitoring Stack..."

Stop-Process -Name "prometheus" -ErrorAction SilentlyContinue
Write-Host "  Stopped Prometheus"

Stop-Process -Name "alertmanager" -ErrorAction SilentlyContinue
Write-Host "  Stopped Alertmanager"

Stop-Process -Name "loki" -ErrorAction SilentlyContinue
Write-Host "  Stopped Loki"

Stop-Process -Name "promtail" -ErrorAction SilentlyContinue
Write-Host "  Stopped Promtail"

Stop-Process -Name "grafana-server" -ErrorAction SilentlyContinue
Write-Host "  Stopped Grafana"

Write-Host "All services stopped."
