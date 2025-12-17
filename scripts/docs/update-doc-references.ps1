# Update documentation cross-references after kebab-case rename
# Created: 2025-12-12

$ErrorActionPreference = 'Stop'

$replacements = @{
    'SIGNALCORE_SYSTEM\.md' = 'signalcore-system.md'
    'EXECUTION_PATH\.md' = 'execution-path.md'
    'SIMULATION_ENGINE\.md' = 'simulation-engine.md'
    'MONITORING\.md' = 'monitoring.md'
    'NVIDIA_INTEGRATION\.md' = 'nvidia-integration.md'
    'RAG_SYSTEM\.md' = 'rag-system.md'
    'PRODUCTION_ARCHITECTURE\.md' = 'production-architecture.md'
    'PHASE1_API_REFERENCE\.md' = 'phase1-api-reference.md'
    'ARCHITECTURE_REVIEW_RESPONSE\.md' = 'architecture-review-response.md'
    'LAYERED_SYSTEM_ARCHITECTURE\.md' = 'layered-system-architecture.md'
}

$filesToUpdate = @(
    'C:\Users\kjfle\Workspace\ordinis\docs\architecture\index.md'
    'C:\Users\kjfle\Workspace\ordinis\docs\architecture\phase1-api-reference.md'
    'C:\Users\kjfle\Workspace\ordinis\docs\architecture\production-architecture.md'
    'C:\Users\kjfle\Workspace\ordinis\docs\architecture\rag-system.md'
    'C:\Users\kjfle\Workspace\ordinis\docs\architecture\signalcore-system.md'
    'C:\Users\kjfle\Workspace\ordinis\docs\project\CURRENT_STATUS_AND_NEXT_STEPS.md'
    'C:\Users\kjfle\Workspace\ordinis\docs\project\PROJECT_STATUS_REPORT.md'
    'C:\Users\kjfle\Workspace\ordinis\docs\guides\CLI_USAGE.md'
    'C:\Users\kjfle\Workspace\ordinis\docs\index.md'
    'C:\Users\kjfle\Workspace\ordinis\docs\DOCUMENTATION_UPDATE_REPORT_20251212.md'
)

foreach ($file in $filesToUpdate) {
    if (Test-Path $file) {
        Write-Host "Updating: $file"
        $content = Get-Content -Path $file -Raw

        foreach ($old in $replacements.Keys) {
            $new = $replacements[$old]
            $content = $content -replace $old, $new
        }

        Set-Content -Path $file -Value $content -NoNewline
        Write-Host "  ✓ Updated"
    } else {
        Write-Warning "File not found: $file"
    }
}

Write-Host "`nComplete! Updated $($filesToUpdate.Count) files."
