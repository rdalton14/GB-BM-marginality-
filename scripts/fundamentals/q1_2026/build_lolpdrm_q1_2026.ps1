Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Variable = "lolpdrm"
$RawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$Variable"
$ProcessedDir = Join-Path $ProjectRoot "data/processed/q1_2026/fundamentals"
$DiagnosticsCoverageDir = Join-Path $ProjectRoot "data/diagnostics/coverage"
$DiagnosticsMissingnessDir = Join-Path $ProjectRoot "data/diagnostics/missingness"
$ProcessedCsv = Join-Path $ProcessedDir "lolpdrm_q1_2026.csv"
$DailyCompletenessCsv = Join-Path $DiagnosticsCoverageDir "lolpdrm_q1_2026_daily_completeness.csv"
$MissingSpsCsv = Join-Path $DiagnosticsMissingnessDir "lolpdrm_q1_2026_missing_sps.csv"

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null
New-Item -ItemType Directory -Force -Path $DiagnosticsCoverageDir | Out-Null
New-Item -ItemType Directory -Force -Path $DiagnosticsMissingnessDir | Out-Null

$BaseUrl = "https://data.elexon.co.uk/bmrs/api/v1/forecast/system/loss-of-load"
$StartDate = [datetime]::ParseExact("2026-01-01", "yyyy-MM-dd", $null)
$EndDate = [datetime]::ParseExact("2026-03-31", "yyyy-MM-dd", $null)

function Invoke-LolpDrm {
    param(
        [Parameter(Mandatory = $true)]
        [datetime] $Day
    )

    $dateText = $Day.ToString("yyyy-MM-dd")
    $uri = "$BaseUrl" + "?from=$dateText&to=$dateText&settlementPeriodFrom=1&settlementPeriodTo=48"
    $response = Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }
    return @($response.data)
}

$allRows = New-Object System.Collections.Generic.List[object]
$day = $StartDate
while ($day -le $EndDate) {
    $dateText = $day.ToString("yyyy-MM-dd")
    $rawPath = Join-Path $RawDir "$dateText.csv"

    if ((Test-Path $rawPath) -and ((Get-Item $rawPath).Length -gt 0)) {
        $rows = @(Import-Csv $rawPath)
        Write-Host "[RAW] $dateText rows=$($rows.Count) path=$rawPath (existing)"
    }
    else {
        $rows = @(Invoke-LolpDrm -Day $day)
        $rows |
            Select-Object publishTime, publishingPeriodCommencingTime, startTime, settlementDate, settlementPeriod, forecastHorizon, lossOfLoadProbability, deratedMargin |
            Sort-Object settlementDate, settlementPeriod, forecastHorizon, publishTime |
            Export-Csv -NoTypeInformation -Path $rawPath

        Write-Host "[RAW] $dateText rows=$($rows.Count) path=$rawPath"
    }

    foreach ($row in $rows) {
        $allRows.Add([pscustomobject]@{
            publishTime = $row.publishTime
            publishingPeriodCommencingTime = $row.publishingPeriodCommencingTime
            startTime = $row.startTime
            settlementDate = $row.settlementDate
            settlementPeriod = [int]$row.settlementPeriod
            forecastHorizon = [int]$row.forecastHorizon
            lossOfLoadProbability = [decimal]$row.lossOfLoadProbability
            deratedMargin = [decimal]$row.deratedMargin
        })
    }

    $day = $day.AddDays(1)
}

$processed = $allRows |
    Where-Object { $_.forecastHorizon -eq 2 } |
    Group-Object settlementDate, settlementPeriod |
    ForEach-Object {
        $_.Group |
            Sort-Object @{ Expression = { [datetime]$_.publishTime }; Descending = $true } |
            Select-Object -First 1
    } |
    Sort-Object settlementDate, settlementPeriod |
    Select-Object settlementDate, settlementPeriod, lossOfLoadProbability, deratedMargin, forecastHorizon, publishTime, publishingPeriodCommencingTime, startTime

$processed | Export-Csv -NoTypeInformation -Path $ProcessedCsv
Write-Host "[PROCESSED] rows=$(@($processed).Count) path=$ProcessedCsv"

$expectedRows = (($EndDate - $StartDate).Days + 1) * 48
$actualRows = @($processed).Count
$duplicates = @(
    $processed |
        Group-Object settlementDate, settlementPeriod |
        Where-Object { $_.Count -gt 1 }
)

$expectedKeys = New-Object System.Collections.Generic.List[object]
$day = $StartDate
while ($day -le $EndDate) {
    for ($sp = 1; $sp -le 48; $sp++) {
        $expectedKeys.Add([pscustomobject]@{
            settlementDate = $day.ToString("yyyy-MM-dd")
            settlementPeriod = $sp
        })
    }
    $day = $day.AddDays(1)
}

$actualKeySet = @{}
foreach ($row in $processed) {
    $actualKeySet["$($row.settlementDate)|$($row.settlementPeriod)"] = $true
}

$missing = @(
    $expectedKeys |
        Where-Object { -not $actualKeySet.ContainsKey("$($_.settlementDate)|$($_.settlementPeriod)") }
)

$dailyCompleteness = $processed |
    Group-Object settlementDate |
    ForEach-Object {
        [pscustomobject]@{
            settlementDate = $_.Name
            rows = $_.Count
        }
    } |
    Sort-Object settlementDate

$dailyCompleteness | Export-Csv -NoTypeInformation -Path $DailyCompletenessCsv
$missing | Export-Csv -NoTypeInformation -Path $MissingSpsCsv

Write-Host ""
Write-Host "DIAGNOSTICS"
Write-Host "expected rows = days * 48 = $expectedRows"
Write-Host "actual rows = $actualRows"
Write-Host "duplicates on (settlementDate, settlementPeriod) = $($duplicates.Count)"
Write-Host "missing SPs count = $($missing.Count)"
Write-Host "daily completeness path = $DailyCompletenessCsv"
Write-Host "missing SPs path = $MissingSpsCsv"
if ($missing.Count -gt 0) {
    $missing |
        Select-Object -First 20 |
        Format-Table -AutoSize |
        Out-String -Width 120 |
        Write-Host
}
