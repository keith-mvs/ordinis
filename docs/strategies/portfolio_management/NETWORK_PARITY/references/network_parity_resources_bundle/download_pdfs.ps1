param(
  [string]$Manifest = "pdf_manifest.csv",
  [string]$OutZip = "network_parity_pdfs.zip",
  [int]$SleepSeconds = 1
)

$ErrorActionPreference = "Stop"
$Base = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Base

$PdfDir = Join-Path $Base "pdfs"
New-Item -ItemType Directory -Force -Path $PdfDir | Out-Null

$Rows = Import-Csv (Join-Path $Base $Manifest)
$Failures = @()

foreach ($r in $Rows) {
  $fn = $r.file_name
  $url = $r.url
  $out = Join-Path $PdfDir $fn

  if (Test-Path $out) {
    $len = (Get-Item $out).Length
    if ($len -gt 10000) { continue }
  }

  try {
    Invoke-WebRequest -Uri $url -OutFile $out -Headers @{ "User-Agent" = "Mozilla/5.0" } -MaximumRedirection 10
  } catch {
    $Failures += [PSCustomObject]@{ file_name = $fn; url = $url; error = $_.Exception.Message }
  }

  Start-Sleep -Seconds ([Math]::Max($SleepSeconds, 0))
}

# Build zip
if (Test-Path $OutZip) { Remove-Item $OutZip -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem

$zipPath = Join-Path $Base $OutZip
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, 'Create')

try {
  [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, (Join-Path $Base "README.md"), "README.md") | Out-Null
  [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, (Join-Path $Base "pdf_manifest.csv"), "pdf_manifest.csv") | Out-Null
  [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, (Join-Path $Base "pdf_manifest.json"), "pdf_manifest.json") | Out-Null

  Get-ChildItem -Path $PdfDir -Filter *.pdf | Sort-Object Name | ForEach-Object {
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $_.FullName, ("pdfs/" + $_.Name)) | Out-Null
  }

  if ($Failures.Count -gt 0) {
    $failPath = Join-Path $Base "download_failures.json"
    $Failures | ConvertTo-Json -Depth 3 | Out-File -Encoding utf8 $failPath
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $failPath, "download_failures.json") | Out-Null
    Remove-Item $failPath -Force
  }
} finally {
  $zip.Dispose()
}

Write-Host "Created: $zipPath"
if ($Failures.Count -gt 0) {
  Write-Warning ("Some downloads failed (" + $Failures.Count + "). See download_failures.json inside the zip.")
}
