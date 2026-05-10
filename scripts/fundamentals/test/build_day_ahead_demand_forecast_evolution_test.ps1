Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Variable = "day_ahead_demand_forecast_evolution"
$RawDir = Join-Path $ProjectRoot "data/raw_test/fundamentals/$Variable"
$ProcessedDir = Join-Path $ProjectRoot "data/processed_test"
$ProcessedCsv = Join-Path $ProcessedDir "day_ahead_demand_forecast_evolution_test.csv"

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null

$BaseUrl = "https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead/evolution"
$StartDate = [datetime]::ParseExact("2026-01-06", "yyyy-MM-dd", $null)
$EndDate = [datetime]::ParseExact("2026-01-10", "yyyy-MM-dd", $null)

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
$day = $StartDate
while ($day -le $EndDate) {
    $dateText = $day.ToString("yyyy-MM-dd")
    $dayRows = New-Object System.Collections.Generic.List[object]

    for ($sp = 1; $sp -le 48; $sp++) {
        $rows = @(Invoke-DayAheadDemandForecastEvolution -Day $day -SettlementPeriod $sp)
        foreach ($row in $rows) {
            $typedRow = [pscustomobject]@{
                startTime = $row.startTime
                settlementDate = $row.settlementDate
                settlementPeriod = [int]$row.settlementPeriod
                boundary = $row.boundary
                publishTime = $row.publishTime
                transmissionSystemDemand = [decimal]$row.transmissionSystemDemand
                nationalDemand = [decimal]$row.nationalDemand
            }
            $dayRows.Add($typedRow)
            $allRows.Add($typedRow)
        }
        Write-Host "  $dateText SP $sp/48 snapshots=$($rows.Count)" -NoNewline
        Write-Host "`r" -NoNewline
    }
    Write-Host ""

    $rawPath = Join-Path $RawDir "$dateText.csv"
    $dayRows |
        Sort-Object settlementDate, settlementPeriod, publishTime |
        Export-Csv -NoTypeInformation -Path $rawPath

    Write-Host "[RAW] $dateText rows=$($dayRows.Count) path=$rawPath"
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
                Where-Object { ([datetime]$_.publishTime).ToUniversalTime().ToString("yyyy-MM-dd") -eq $dayAheadDate }
        )
        if ($candidates.Count -gt 0) {
            $selected = $candidates |
                Sort-Object @{ Expression = { [datetime]$_.publishTime }; Descending = $true } |
                Select-Object -First 1

            [pscustomobject]@{
                settlementDate = $selected.settlementDate
                settlementPeriod = $selected.settlementPeriod
                dayAheadNationalDemandForecast = $selected.nationalDemand
                dayAheadTransmissionSystemDemandForecast = $selected.transmissionSystemDemand
                dayAheadDemandForecast_publishTime = $selected.publishTime
                startTime = $selected.startTime
            }
        }
    } |
    Sort-Object settlementDate, settlementPeriod

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

Write-Host ""
Write-Host "DIAGNOSTICS"
Write-Host "expected rows = days * 48 = $expectedRows"
Write-Host "actual rows = $actualRows"
Write-Host "duplicates on (settlementDate, settlementPeriod) = $($duplicates.Count)"
Write-Host "missing SPs after D-1 latest-publish selection = $($missing.Count)"
if ($missing.Count -gt 0) {
    $missing |
        Select-Object -First 20 |
        Format-Table -AutoSize |
        Out-String -Width 120 |
        Write-Host
}
