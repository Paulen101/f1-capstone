# Build .venv-tum on 64-bit Windows; see notes/environment.md. Safe to re-run.
#
#   powershell -ExecutionPolicy Bypass -File integration\setup_windows.ps1
#
# Python 3.8 is unpacked into .tools/python, so nothing is installed system-wide.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = Split-Path $PSScriptRoot -Parent
$Tools = Join-Path $Root '.tools'
$BasePython = Join-Path $Tools 'python\python.exe'
$Python = Join-Path $Root '.venv-tum\Scripts\python.exe'
$Release = 'https://github.com/astral-sh/python-build-standalone/releases/download/20231002'
$Archive = 'cpython-3.8.18+20231002-x86_64-pc-windows-msvc-shared-install_only.tar.gz'

if (-not (Test-Path $BasePython)) {
    $Download = Join-Path $env:TEMP ('f1-capstone-' + [guid]::NewGuid())
    New-Item -ItemType Directory $Download | Out-Null
    try {
        Invoke-WebRequest -UseBasicParsing "$Release/$Archive" -OutFile "$Download\python.tar.gz"
        Invoke-WebRequest -UseBasicParsing "$Release/SHA256SUMS" -OutFile "$Download\SHA256SUMS"
        $Line = Get-Content "$Download\SHA256SUMS" | Where-Object { $_ -match [regex]::Escape($Archive) }
        $Expected = ($Line -split '\s+')[0]
        $Actual = (Get-FileHash "$Download\python.tar.gz" -Algorithm SHA256).Hash.ToLower()
        if (-not $Expected -or $Expected -ne $Actual) { throw "Checksum mismatch for $Archive" }
        New-Item -ItemType Directory -Force $Tools | Out-Null
        # Full path: from Git Bash, plain tar is Git's, which reads C:\ as a remote host.
        & "$env:SystemRoot\System32\tar.exe" -xzf "$Download\python.tar.gz" -C $Tools
        if ($LASTEXITCODE -ne 0) { throw 'Could not unpack Python' }
    } finally {
        Remove-Item -Recurse -Force $Download
    }
}

if (-not (Test-Path $Python)) {
    & $BasePython -m venv (Join-Path $Root '.venv-tum')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv-tum' }
}

# Wheels only: the pinned 2020-era packages should not need a C++ compiler here.
& $Python -m pip install --disable-pip-version-check --only-binary ':all:' `
    -r (Join-Path $Root 'race-simulation\requirements.txt') `
    -c (Join-Path $Root 'notes\requirements-tum-windows.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

& $Python -m pip check --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'pip check failed' }

# Importing cvxpy loads osqp, which crashes natively with the wrong wheel; test it here.
$env:TF_CPP_MIN_LOG_LEVEL = '2'
& $Python -c "import cvxpy, tensorflow; print('cvxpy', cvxpy.__version__, 'tensorflow', tensorflow.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Import check failed with exit code $LASTEXITCODE" }

Write-Output 'Setup complete: .venv-tum is ready.'
