# install_monitoring.ps1
$ErrorActionPreference = "Stop"
$WorkDir = "C:\Users\kjfle\Workspace\ordinis"
$BinDir = "$WorkDir\bin"
$DownloadDir = "$BinDir\downloads"

# Create directories
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null

# Tools to install
$Tools = @(
    @{
        Name = "prometheus"
        Url = "https://github.com/prometheus/prometheus/releases/download/v2.53.0/prometheus-2.53.0.windows-amd64.zip"
        DirName = "prometheus-2.53.0.windows-amd64"
    },
    @{
        Name = "alertmanager"
        Url = "https://github.com/prometheus/alertmanager/releases/download/v0.27.0/alertmanager-0.27.0.windows-amd64.zip"
        DirName = "alertmanager-0.27.0.windows-amd64"
    },
    @{
        Name = "loki"
        Url = "https://github.com/grafana/loki/releases/download/v2.9.8/loki-windows-amd64.exe.zip"
        IsExeZip = $true
        ExeName = "loki-windows-amd64.exe"
        TargetName = "loki.exe"
    },
    @{
        Name = "promtail"
        Url = "https://github.com/grafana/loki/releases/download/v2.9.8/promtail-windows-amd64.exe.zip"
        IsExeZip = $true
        ExeName = "promtail-windows-amd64.exe"
        TargetName = "promtail.exe"
    },
    @{
        Name = "grafana"
        Url = "https://dl.grafana.com/oss/release/grafana-11.0.0.windows-amd64.zip"
        DirName = "grafana-v11.0.0"
    }
)

foreach ($Tool in $Tools) {
    Write-Host "Processing $($Tool.Name)..."
    $ZipPath = "$DownloadDir\$($Tool.Name).zip"

    # Download
    if (-not (Test-Path $ZipPath)) {
        Write-Host "  Downloading..."
        Invoke-WebRequest -Uri $Tool.Url -OutFile $ZipPath
    }

    # Extract
    $ExtractPath = "$BinDir\$($Tool.Name)"
    if (-not (Test-Path $ExtractPath)) {
        Write-Host "  Extracting..."
        if ($Tool.IsExeZip) {
            # For Loki/Promtail which are single EXEs inside ZIPs
            Expand-Archive -Path $ZipPath -DestinationPath $BinDir -Force
            # Rename/Move
            $SourceExe = "$BinDir\$($Tool.ExeName)"
            New-Item -ItemType Directory -Force -Path $ExtractPath | Out-Null
            Move-Item -Path $SourceExe -Destination "$ExtractPath\$($Tool.TargetName)" -Force
        } else {
            # For standard folders (Prometheus, Grafana)
            Expand-Archive -Path $ZipPath -DestinationPath $BinDir -Force
            # Rename folder to simple name
            $OriginalDir = "$BinDir\$($Tool.DirName)"
            if (Test-Path $OriginalDir) {
                Rename-Item -Path $OriginalDir -NewName $Tool.Name -Force
            }
        }
    }
    Write-Host "  $($Tool.Name) installed."
}

Write-Host "All tools installed to $BinDir"
