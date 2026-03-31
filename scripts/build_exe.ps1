$ErrorActionPreference = "Stop"

Write-Host "build_exe.ps1 is kept for compatibility. Running portable build..."
powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build_portable.ps1")
