param(
    [switch]$Installer
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Version = (Get-Content -LiteralPath (Join-Path $Root "VERSION") -Raw).Trim()

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found: $Python"
}

& $Python (Join-Path $Root "scripts\generate_app_icon.py")
& $Python -m pip install --disable-pip-version-check "pyinstaller==6.16.0"
& $Python -m PyInstaller --noconfirm --clean (Join-Path $Root "DesktopHealthAssistant.spec")

$Release = Join-Path $Root "release"
New-Item -ItemType Directory -Force -Path $Release | Out-Null
$PortableZip = Join-Path $Release "DesktopHealthAssistant-Portable-$Version.zip"
Compress-Archive -Path (Join-Path $Root "dist\Desktop Health Assistant\*") -DestinationPath $PortableZip -Force

if ($Installer) {
    $Iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($null -eq $Iscc) {
        throw "Inno Setup is not installed or iscc.exe is not on PATH."
    }
    & $Iscc.Source (Join-Path $Root "packaging\installer.iss")
}

Write-Host "Windows build complete:"
Write-Host "  $Root\dist\Desktop Health Assistant\DesktopHealthAssistant.exe"
Write-Host "  $PortableZip"
