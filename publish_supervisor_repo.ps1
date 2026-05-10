param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteUrl,

    [string]$CommitMessage = "Initial supervisor review snapshot",

    [switch]$Yes
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,

        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host $Label
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not available on PATH."
}

if (-not (Test-Path ".git")) {
    Invoke-Step "Initialising Git repository..." { git init }
}

if (Get-Command git-lfs -ErrorAction SilentlyContinue) {
    Invoke-Step "Configuring Git LFS..." { git lfs install --local }
} else {
    Write-Warning "Git LFS is not available. Install Git LFS before pushing large data files."
}

git status --short --ignored

Write-Host ""
Write-Host "Staging repository snapshot..."
Invoke-Step "Running git add..." { git add . }

Write-Host ""
Write-Host "Large tracked files:"
git ls-files | ForEach-Object {
    $item = Get-Item $_ -ErrorAction SilentlyContinue
    if ($item -and $item.Length -gt 50MB) {
        "{0:N1} MB`t{1}" -f ($item.Length / 1MB), $_
    }
}

Write-Host ""
Write-Host "LFS-tracked files:"
git lfs ls-files

Write-Host ""
Write-Host "Staged summary:"
git diff --cached --stat

Write-Host ""
if (-not $Yes) {
    $confirm = Read-Host "Commit and push this snapshot to $RemoteUrl ? Type YES to continue"
    if ($confirm -ne "YES") {
        Write-Host "Cancelled before commit."
        exit 0
    }
}

Invoke-Step "Committing snapshot..." { git commit -m $CommitMessage }
Invoke-Step "Setting branch to main..." { git branch -M main }

if (-not (git remote | Select-String -SimpleMatch "origin")) {
    Invoke-Step "Adding origin remote..." { git remote add origin $RemoteUrl }
} else {
    Invoke-Step "Updating origin remote..." { git remote set-url origin $RemoteUrl }
}

Invoke-Step "Pushing to GitHub..." { git push -u origin main }
