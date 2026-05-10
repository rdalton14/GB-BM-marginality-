Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Variable = "actual_total_load"
$RawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$Variable"
$ProcessedDir = Join-Path $ProjectRoot "data/processed/q1_2026/fundamentals"
$ProcessedCsv = Join-Path $ProcessedDir "actual_total_load_q1_2026.csv"

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null

$BaseUrl = "https://data.elexon.co.uk/bmrs/api/v1/demand/actual/total"
$StartDate = [datetime]::ParseExact("2026-01-01", "yyyy-MM-dd", $null)
$EndDate = [datetime]::ParseExact("2026-03-31", "yyyy-MM-dd", $null)

function Invoke-ActualTotalLoad {
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
$rawCounts = New-Object System.Collections.Generic.List[object]
$day = $StartDate
while ($day -le $EndDate) {
    $dateText = $day.ToString("yyyy-MM-dd")
    $rows = Invoke-ActualTotalLoad -Day $day

    foreach ($row in $rows) {
        $allRows.Add([pscustomobject]@{
            publishTime = $row.publishTime
            startTime = $row.startTime
            settlementDate = $row.settlementDate
            settlementPeriod = [int]$row.settlementPeriod
            quantity = [decimal]$row.quantity
        })
    }

    $rawPath = Join-Path $RawDir "$dateText.csv"
    $rows |
        Select-Object publishTime, startTime, settlementDate, settlementPeriod, quantity |
        Sort-Object settlementDate, settlementPeriod, publishTime |
        Export-Csv -NoTypeInformation -Path $rawPath

    $rawCounts.Add([pscustomobject]@{
        settlementDate = $dateText
        rawRows = $rows.Count
    })
    Write-Host "[RAW] $dateText rows=$($rows.Count) path=$rawPath"
    $day = $day.AddDays(1)
}

$processed = $allRows |
    Group-Object settlementDate, settlementPeriod |
    ForEach-Object {
        $_.Group |
            Sort-Object @{ Expression = "publishTime"; Descending = $true } |
            Select-Object -First 1
    } |
    Sort-Object settlementDate, settlementPeriod |
    Select-Object settlementDate, settlementPeriod, @{ Name = "ActualTotalLoad"; Expression = { $_.quantity } }, publishTime, startTime

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

$rawCountIssues = @($rawCounts | Where-Object { $_.rawRows -ne 48 })

Write-Host ""
Write-Host "DIAGNOSTICS"
Write-Host "date window = 2026-01-01 to 2026-03-31"
Write-Host "days = $days"
Write-Host "expected rows at 48 SP/day = $expectedRows48"
Write-Host "actual rows = $actualRows"
Write-Host "duplicates on (settlementDate, settlementPeriod) = $($duplicates.Count)"
Write-Host "days with raw row count not equal to 48 = $($rawCountIssues.Count)"
if ($rawCountIssues.Count -gt 0) {
    $rawCountIssues |
        Format-Table -AutoSize |
        Out-String -Width 120 |
        Write-Host
}
Write-Host "missing SPs count versus 1..48 calendar expectation = $($missing.Count)"
if ($missing.Count -gt 0) {
    $missing |
        Select-Object -First 20 |
        Format-Table -AutoSize |
        Out-String -Width 120 |
        Write-Host
}
