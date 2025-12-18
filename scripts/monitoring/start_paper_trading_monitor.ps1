<#
.SYNOPSIS
    Start the Ordinis Paper Trading Monitor stack.

.DESCRIPTION
    Launches Prometheus, Grafana, and the metrics exporter for
    real-time paper trading workflow monitoring.

.PARAMETER SkipBrowser
    Skip opening the Grafana dashboard in the browser.

.EXAMPLE
    .\start_paper_trading_monitor.ps1
    Starts all monitoring services and opens the dashboard.

.EXAMPLE
    .\start_paper_trading_monitor.ps1 -SkipBrowser
    Starts services without opening the browser.
#>

[CmdletBinding()]
param(
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"

# Configuration
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$BinDir = Join-Path $ProjectRoot "bin"
$ConfigDir = Join-Path $ProjectRoot "configs"
$DataDir = Join-Path $ProjectRoot "data"

$PrometheusExe = Join-Path $BinDir "prometheus\prometheus.exe"
$GrafanaExe = Join-Path $BinDir "grafana\bin\grafana-server.exe"

$PrometheusConfig = Join-Path $ConfigDir "prometheus\prometheus.yml"
$GrafanaConfig = Join-Path $ConfigDir "grafana\custom.ini"

$PrometheusData = Join-Path $DataDir "prometheus"
$GrafanaData = Join-Path $DataDir "grafana\data"

# Ports
$PrometheusPort = 3001
$GrafanaPort = 3010
$MetricsExporterPort = 3005

# Dashboard URL
$DashboardUrl = "http://localhost:$GrafanaPort/d/ordinis-paper-trading/ordinis-paper-trading-monitor"

function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    $color = switch ($Status) {
        "INFO" { "Cyan" }
        "OK" { "Green" }
        "WARN" { "Yellow" }
        "ERROR" { "Red" }
        default { "White" }
    }
    Write-Host "[$Status] " -ForegroundColor $color -NoNewline
    Write-Host $Message
}

function Test-PortInUse {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

function Stop-ProcessOnPort {
    param([int]$Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($connection) {
        $process = Get-Process -Id $connection.OwningProcess -ErrorAction SilentlyContinue
        if ($process) {
            Write-Status "Stopping existing process on port $Port ($($process.ProcessName))" "WARN"
            Stop-Process -Id $process.Id -Force
            Start-Sleep -Seconds 1
        }
    }
}

function Start-Prometheus {
    Write-Status "Starting Prometheus on port $PrometheusPort..."

    # Ensure data directory exists
    if (-not (Test-Path $PrometheusData)) {
        New-Item -ItemType Directory -Path $PrometheusData -Force | Out-Null
    }

    # Check if already running
    if (Test-PortInUse $PrometheusPort) {
        Write-Status "Prometheus already running on port $PrometheusPort" "OK"
        return
    }

    # Verify binary exists
    if (-not (Test-Path $PrometheusExe)) {
        Write-Status "Prometheus binary not found at $PrometheusExe" "ERROR"
        return
    }

    # Start Prometheus
    $prometheusArgs = @(
        "--config.file=$PrometheusConfig",
        "--storage.tsdb.path=$PrometheusData",
        "--web.listen-address=:$PrometheusPort",
        "--web.enable-lifecycle"
    )

    Start-Process -FilePath $PrometheusExe -ArgumentList $prometheusArgs -WindowStyle Hidden
    Start-Sleep -Seconds 2

    if (Test-PortInUse $PrometheusPort) {
        Write-Status "Prometheus started successfully" "OK"
    } else {
        Write-Status "Failed to start Prometheus" "ERROR"
    }
}

function Start-Grafana {
    Write-Status "Starting Grafana on port $GrafanaPort..."

    # Ensure data directory exists
    if (-not (Test-Path $GrafanaData)) {
        New-Item -ItemType Directory -Path $GrafanaData -Force | Out-Null
    }

    # Check if already running
    if (Test-PortInUse $GrafanaPort) {
        Write-Status "Grafana already running on port $GrafanaPort" "OK"
        return
    }

    # Verify binary exists
    if (-not (Test-Path $GrafanaExe)) {
        Write-Status "Grafana binary not found at $GrafanaExe" "ERROR"
        return
    }

    # Start Grafana
    $grafanaDir = Split-Path -Parent (Split-Path -Parent $GrafanaExe)
    $env:GF_PATHS_CONFIG = $GrafanaConfig
    $env:GF_PATHS_DATA = $GrafanaData
    $env:GF_PATHS_PROVISIONING = Join-Path $ConfigDir "grafana\provisioning"
    $env:GF_SERVER_HTTP_PORT = $GrafanaPort

    Start-Process -FilePath $GrafanaExe -WorkingDirectory $grafanaDir -WindowStyle Hidden
    Start-Sleep -Seconds 3

    if (Test-PortInUse $GrafanaPort) {
        Write-Status "Grafana started successfully" "OK"
    } else {
        Write-Status "Failed to start Grafana" "ERROR"
    }
}

function Start-MetricsExporter {
    Write-Status "Starting Metrics Exporter on port $MetricsExporterPort..."

    # Check if already running
    if (Test-PortInUse $MetricsExporterPort) {
        Write-Status "Metrics Exporter already running on port $MetricsExporterPort" "OK"
        return
    }

    # Activate conda environment and start exporter
    $exporterModule = Join-Path $ProjectRoot "src\ordinis\monitoring\metrics_exporter.py"

    if (-not (Test-Path $exporterModule)) {
        Write-Status "Metrics exporter module not found at $exporterModule" "ERROR"
        return
    }

    # Start using Python from the ordinis-env conda environment
    $pythonArgs = @(
        "-m", "ordinis.monitoring.metrics_exporter",
        "--port", $MetricsExporterPort
    )

    # Try to find Python in conda environment
    $condaPython = "C:\ProgramData\anaconda3\envs\ordinis-env\python.exe"
    if (Test-Path $condaPython) {
        Start-Process -FilePath $condaPython -ArgumentList $pythonArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden
    } else {
        # Fallback to system Python
        Start-Process -FilePath "python" -ArgumentList $pythonArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden
    }

    Start-Sleep -Seconds 2

    if (Test-PortInUse $MetricsExporterPort) {
        Write-Status "Metrics Exporter started successfully" "OK"
    } else {
        Write-Status "Failed to start Metrics Exporter (may need to run manually)" "WARN"
    }
}

function Show-Summary {
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host "  Ordinis Paper Trading Monitor" -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Services:" -ForegroundColor White

    $services = @(
        @{ Name = "Prometheus"; Port = $PrometheusPort; Url = "http://localhost:$PrometheusPort" },
        @{ Name = "Grafana"; Port = $GrafanaPort; Url = "http://localhost:$GrafanaPort" },
        @{ Name = "Metrics Exporter"; Port = $MetricsExporterPort; Url = "http://localhost:$MetricsExporterPort/metrics" }
    )

    foreach ($svc in $services) {
        $status = if (Test-PortInUse $svc.Port) { "[OK]" } else { "[--]" }
        $color = if (Test-PortInUse $svc.Port) { "Green" } else { "Red" }
        Write-Host "    $status " -ForegroundColor $color -NoNewline
        Write-Host "$($svc.Name): " -NoNewline
        Write-Host $svc.Url -ForegroundColor Blue
    }

    Write-Host ""
    Write-Host "  Dashboard: " -NoNewline
    Write-Host $DashboardUrl -ForegroundColor Blue
    Write-Host ""
    Write-Host "  Credentials: admin / admin" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "=" * 60 -ForegroundColor Cyan
}

# Main execution
Write-Host ""
Write-Status "Starting Ordinis Paper Trading Monitor..."
Write-Host ""

# Start services
Start-Prometheus
Start-Grafana
Start-MetricsExporter

# Show summary
Show-Summary

# Open browser
if (-not $SkipBrowser) {
    Write-Status "Opening dashboard in browser..."
    Start-Sleep -Seconds 1
    Start-Process $DashboardUrl
}

Write-Host ""
Write-Status "Monitor started. Press Ctrl+C to exit." "OK"
Write-Host ""
