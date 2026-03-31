$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Building release ZIP artifact..."
powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build_release_zip.ps1")

Write-Host "Building one-file EXE artifact..."
powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build_onefile.ps1")

Write-Host "Combined release build completed:"
Write-Host "- dist\LocalSTT-Portable.zip"
Write-Host "- dist\LocalSTT-OneFile.exe"