$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Cleaning previous build artifacts..."
if (Test-Path "build") {
    Remove-Item -Path "build" -Recurse -Force
}
if (Test-Path "dist\LocalSTT") {
    Remove-Item -Path "dist\LocalSTT" -Recurse -Force
}
if (Test-Path "LocalSTT.spec") {
    Remove-Item -Path "LocalSTT.spec" -Force
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found: .venv"
}

if (-not (Test-Path "models\faster-whisper-small")) {
    throw "Missing required model folder: models\faster-whisper-small"
}

$python = ".\.venv\Scripts\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install pyinstaller

# Build portable window app with bundled default model and icon.
& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name LocalSTT `
    --add-data "models\faster-whisper-small;models\faster-whisper-small" `
    --add-data "src\icon.png;src" `
    --collect-data faster_whisper `
    src/main.py

Write-Host "Portable build completed: dist\LocalSTT\LocalSTT.exe"
