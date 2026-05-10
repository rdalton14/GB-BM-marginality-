Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Variable = "day_ahead_demand_forecast_evolution"
$RawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$Variable"
$ProcessedDir = Join-Path $ProjectRoot "data/processed/q1_2026/fundamentals"
$ProcessedCsv = Join-Path $ProcessedDir "day_ahead_national_demand_forecast_q1_2026.csv"
$DiagnosticsCoverageDir = Join-Path $ProjectRoot "data/diagnostics/coverage"
$DiagnosticsMissingnessDir = Join-Path $ProjectRoot "data/diagnostics/missingness"
$DailyCompletenessCsv = Join-Path $DiagnosticsCoverageDir "day_ahead_national_demand_forecast_q1_2026_daily_completeness.csv"
$MissingSpsCsv = Join-Path $DiagnosticsMissingnessDir "day_ahead_national_demand_forecast_q1_2026_missing_sps.csv"

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null
New-Item -ItemType Directory -Force -Path $DiagnosticsCoverageDir | Out-Null
New-Item -ItemType Directory -Force -Path $DiagnosticsMissingnessDir | Out-Null

$BaseUrl = "https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead/evolution"
$StartDate = [datetime]::ParseExact("2026-01-01", "yyyy-MM-dd", $null)
$EndDate = [datetime]::ParseExact("2026-03-31", "yyyy-MM-dd", $null)

function Invoke-DayAheadDemandForecastEvolution {
    param(
        [Parameter(Mandatory = $true)]
        [datetime] $Day,

        [Parameter(Mandatory = $true)]
        [int] $SettlementPeriod
    )

    $dateText = $Day.ToString("yyyy-MM-dd")
    $uri = "$BaseUrl" + "?settlementDate=$dateText&settlementPeriod=$SettlementPeriod"
    $response = Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }
    return @($response.data)
}

$allRows = New-Object System.Collections.Generic.List[object]
$rawCounts = New-Object System.Collections.Generic.List[object]
$day = $StartDate
while ($day -le $EndDate) {
    $dateText = $day.ToString("yyyy-MM-dd")
    $rawPath = Join-Path $RawDir "$dateText.csv"

    if (Test-Path $rawPath) {
        $dayRows = @(Import-Csv $rawPath)
        Write-Host "[SKIP] $dateText rows=$($dayRows.Count) path=$rawPath"
    } else {
        $dayRows = New-Object System.Collections.Generic.List[object]

        for ($sp = 1; $sp -le 48; $sp++) {
            $rows = @(Invoke-DayAheadDemandForecastEvolution -Day $day -SettlementPeriod $sp)
            foreach ($row in $rows) {
                $dayRows.Add([pscustomobject]@{
                    startTime = $row.startTime
                    settlementDate = $row.settlementDate
                    settlementPeriod = [int]$row.settlementPeriod
                    boundary = $row.boundary
                    publishTime = $row.publishTime
                    transmissionSystemDemand = [decimal]$row.transmissionSystemDemand
                    nationalDemand = [decimal]$row.nationalDemand
                })
            }
            Write-Host "  $dateText SP $sp/48 snapshots=$($rows.Count)" -NoNewline
            Write-Host "`r" -NoNewline
        }
        Write-Host ""

        $dayRows |
            Sort-Object settlementDate, settlementPeriod, publishTime |
            Export-Csv -NoTypeInformation -Path $rawPath

        Write-Host "[RAW] $dateText rows=$($dayRows.Count) path=$rawPath"
    }

    foreach ($row in $dayRows) {
        $allRows.Add([pscustomobject]@{
            startTime = $row.startTime
            settlementDate = $row.settlementDate
            settlementPeriod = [int]$row.settlementPeriod
            boundary = $row.boundary
            publishTime = $row.publishTime
            nationalDemand = [decimal]$row.nationalDemand
        })
    }

    $rawCounts.Add([pscustomobject]@{
        settlementDate = $dateText
        rawRows = $dayRows.Count
    })

    $day = $day.AddDays(1)
}

$processed = $allRows |
    Group-Object settlementDate, settlementPeriod |
    ForEach-Object {
        $first = $_.Group | Select-Object -First 1
        $deliveryDate = [datetime]::ParseExact($first.settlementDate, "yyyy-MM-dd", $null)
        $dayAheadDate = $deliveryDate.AddDays(-1).ToString("yyyy-MM-dd")
        $candidates = @(
            $_.Group |
                Where-Object {
                    ([datetime]$_.publishTime).ToUniversalTime().ToString("yyyy-MM-dd") -eq $dayAheadDate -and
                    [decimal]$_.nationalDemand -ne 0
                }
        )
        if ($candidates.Count -gt 0) {
            $selected = $candidates |
                Sort-Object @{ Expression = { [datetime]$_.publishTime }; Descending = $true } |
                Select-Object -First 1

            [pscustomobject]@{
                settlementDate = $selected.settlementDate
                settlementPeriod = $selected.settlementPeriod
                dayAheadNationalDemandForecast = $selected.nationalDemand
                dayAheadNationalDemandForecast_publishTime = $selected.publishTime
                startTime = $selected.startTime
            }
        }
    } |
    Sort-Object settlementDate, settlementPeriod

$processed | Export-Csv -NoTypeInformation -Path $ProcessedCsv
Write-Host "[PROCESSED] rows=$(@($processed).Count) path=$ProcessedCsv"

$days = ($EndDate - $StartDate).Days + 1
$expectedRows48 = $days * 48
$actualRows = @($processed).Count
$duplicates = @(
    $processed |
        Group-Object settlementDate, settlementPeriod |
        Where-Object { $_.Count -gt 1 }
)
$zeroForecastRows = @($processed | Where-Object { [decimal]$_.dayAheadNationalDemandForecast -eq 0 })

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

$daily = New-Object System.Collections.Generic.List[object]
$day = $StartDate
while ($day -le $EndDate) {
    $dateText = $day.ToString("yyyy-MM-dd")
    $rows = @($processed | Where-Object { $_.settlementDate -eq $dateText })
    $sps = @($rows | Select-Object -ExpandProperty settlementPeriod | Sort-Object { [int]$_ } -Unique)
    $missingForDay = @(1..48 | Where-Object { [string]$_ -notin $sps })
    $rawCount = ($rawCounts | Where-Object { $_.settlementDate -eq $dateText }).rawRows
    $daily.Add([pscustomobject]@{
        settlementDate = $dateText
        rawSnapshotRows = $rawCount
        processedRows = $rows.Count
        missingCountVs48 = $missingForDay.Count
        missingSettlementPeriodsVs48 = ($missingForDay -join ";")
    })
    $day = $day.AddDays(1)
}

$daily | Export-Csv -NoTypeInformation -Path $DailyCompletenessCsv
$missing | Export-Csv -NoTypeInformation -Path $MissingSpsCsv

Write-Host ""
Write-Host "DIAGNOSTICS"
Write-Host "date window = 2026-01-01 to 2026-03-31"
Write-Host "days = $days"
Write-Host "expected rows at 48 SP/day = $expectedRows48"
Write-Host "actual rows = $actualRows"
Write-Host "duplicates on (settlementDate, settlementPeriod) = $($duplicates.Count)"
Write-Host "zero dayAheadNationalDemandForecast rows = $($zeroForecastRows.Count)"
Write-Host "missing SPs after D-1 latest-publish selection = $($missing.Count)"
if ($missing.Count -gt 0) {
    $missing |
        Select-Object -First 20 |
        Format-Table -AutoSize |
        Out-String -Width 120 |
        Write-Host
}
Write-Host "daily completeness report = $DailyCompletenessCsv"
Write-Host "missing SP report = $MissingSpsCsv"
