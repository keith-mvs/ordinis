# Final File Copy and Cleanup Script
# Extracts files.zip and copies skill files to correct locations

$ErrorActionPreference = "Stop"

$projectRoot = "C:\Users\kjfle\.projects\intelligent-investor"
$zipFile = "$projectRoot\files.zip"
$tempExtract = "$projectRoot\temp-extract-files"
$skillDir = "$projectRoot\skills\due-diligence"

Write-Host "=== Due Diligence Skill File Installer ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Extract files.zip
Write-Host "Step 1: Extracting files.zip..." -ForegroundColor Yellow
if (-not (Test-Path $zipFile)) {
    Write-Host "ERROR: files.zip not found at $zipFile" -ForegroundColor Red
    exit 1
}

if (Test-Path $tempExtract) {
    Remove-Item $tempExtract -Recurse -Force
}
Expand-Archive -Path $zipFile -DestinationPath $tempExtract -Force
Write-Host "✓ Extracted files.zip" -ForegroundColor Green

# Step 2: Create directories
Write-Host "Step 2: Creating directory structure..." -ForegroundColor Yellow
$refsDir = "$skillDir\references"
$assetsDir = "$skillDir\assets"

if (-not (Test-Path $refsDir)) {
    New-Item -ItemType Directory -Path $refsDir -Force | Out-Null
}
if (-not (Test-Path $assetsDir)) {
    New-Item -ItemType Directory -Path $assetsDir -Force | Out-Null
}
Write-Host "✓ Directories ready" -ForegroundColor Green

# Step 3: Copy files based on their names
Write-Host "`nStep 3: Copying files..." -ForegroundColor Yellow

# Find all MD files in the extracted content
$allFiles = Get-ChildItem -Path $tempExtract -Filter "*.md" -Recurse

# Reference files (these go in references/)
$refFiles = @("compliance-review.md", "financial-dd.md", "market-analysis.md", "technical-dd.md", "vendor-evaluation.md")

# Asset files
$assetFiles = @("report-template.md")

foreach ($file in $allFiles) {
    if ($refFiles -contains $file.Name) {
        Copy-Item -Path $file.FullName -Destination $refsDir -Force
        Write-Host "  ✓ Copied $($file.Name) to references/" -ForegroundColor Green
    }
    elseif ($assetFiles -contains $file.Name) {
        Copy-Item -Path $file.FullName -Destination $assetsDir -Force
        Write-Host "  ✓ Copied $($file.Name) to assets/" -ForegroundColor Green
    }
}

# Step 4: Cleanup
Write-Host "`nStep 4: Cleaning up..." -ForegroundColor Yellow

# Remove temp extraction folder
if (Test-Path $tempExtract) {
    Remove-Item $tempExtract -Recurse -Force
}

# Remove files.zip
if (Test-Path $zipFile) {
    Remove-Item $zipFile -Force
}

# Remove PowerShell scripts from previous attempts
$scriptsToRemove = @("copy-skill-files.ps1", "simple-copy.ps1")
foreach ($script in $scriptsToRemove) {
    $scriptPath = "$skillDir\$script"
    if (Test-Path $scriptPath) {
        Remove-Item $scriptPath -Force
        Write-Host "  ✓ Removed $script" -ForegroundColor Green
    }
}

Write-Host "✓ Cleanup complete" -ForegroundColor Green

# Step 5: Verify
Write-Host "`n=== Installation Complete! ===" -ForegroundColor Cyan
Write-Host "`nInstalled files:" -ForegroundColor White

Write-Host "`nReferences ($refsDir):" -ForegroundColor Yellow
Get-ChildItem $refsDir -File | ForEach-Object {
    Write-Host "  - $($_.Name) ($([math]::Round($_.Length/1KB, 1)) KB)" -ForegroundColor Gray
}

Write-Host "`nAssets ($assetsDir):" -ForegroundColor Yellow
Get-ChildItem $assetsDir -File | ForEach-Object {
    Write-Host "  - $($_.Name) ($([math]::Round($_.Length/1KB, 1)) KB)" -ForegroundColor Gray
}

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Review the files in your repository"
Write-Host "  2. Commit to git: git add skills/due-diligence/"
Write-Host "  3. Commit to git: git commit -m 'Add due diligence skill frameworks'"
Write-Host ""
