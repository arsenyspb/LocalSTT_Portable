$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Build the portable folder first so the ZIP always contains the latest files.
powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build_portable.ps1")

$zipPath = Join-Path $root "dist\LocalSTT-Portable.zip"
if (Test-Path $zipPath) {
    Remove-Item -Path $zipPath -Force
}

for ($attempt = 1; $attempt -le 5; $attempt++) {
    try {
        if (Test-Path $zipPath) {
            Remove-Item -Path $zipPath -Force
        }

        Start-Sleep -Seconds 2
        Compress-Archive -Path "dist\LocalSTT" -DestinationPath $zipPath -Force
        break
    }
    catch {
        if ($attempt -eq 5) {
            throw "Could not create dist\LocalSTT-Portable.zip after multiple attempts. Close LocalSTT and retry. $($_.Exception.Message)"
        }

        Write-Host "ZIP packaging attempt $attempt failed, retrying..."
        Start-Sleep -Seconds 3
    }
}

Write-Host "Release ZIP completed: dist\LocalSTT-Portable.zip"
