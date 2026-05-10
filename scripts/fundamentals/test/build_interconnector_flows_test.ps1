Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Variable = "interconnector_flows"
$RawDir = Join-Path $ProjectRoot "data/raw_test/fundamentals/$Variable"
$ProcessedDir = Join-Path $ProjectRoot "data/processed_test"
$ProcessedCsv = Join-Path $ProcessedDir "interconnector_flows_test.csv"

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null

$BaseUrl = "https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/interconnectors"
$StartDate = [datetime]::ParseExact("2026-01-06", "yyyy-MM-dd", $null)
$EndDate = [datetime]::ParseExact("2026-01-10", "yyyy-MM-dd", $null)

function Invoke-InterconnectorFlows {
    param(
        [Parameter(Mandatory = $true)]
        [datetime] $Day
    )

    $dateText = $Day.ToString("yyyy-MM-dd")
    $uri = "$BaseUrl" + "?settlementDateFrom=$dateText&settlementDateTo=$dateText"
    $response = Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }
    return @($response.data)
}

$allRows = New-Object System.Collections.Generic.List[object]
$day = $StartDate
while ($day -le $EndDate) {
    $dateText = $day.ToString("yyyy-MM-dd")
    $rows = @(Invoke-InterconnectorFlows -Day $day)

    foreach ($row in $rows) {
        $allRows.Add([pscustomobject]@{
            dataset = $row.dataset
            publishTime = $row.publishTime
            startTime = $row.startTime
            settlementDate = $row.settlementDate
            settlementDateTimezone = $row.settlementDateTimezone
            settlementPeriod = [int]$row.settlementPeriod
            interconnectorName = $row.interconnectorName
            generation = [decimal]$row.generation
        })
    }

    $rawPath = Join-Path $RawDir "$dateText.csv"
    $rows |
        Select-Object dataset, publishTime, startTime, settlementDate, settlementDateTimezone, settlementPeriod, interconnectorName, generation |
        Sort-Object settlementDate, settlementPeriod, interconnectorName, publishTime |
        Export-Csv -NoTypeInformation -Path $rawPath

    Write-Host "[RAW] $dateText rows=$($rows.Count) path=$rawPath"
    $day = $day.AddDays(1)
}

$latestByInterconnector = $allRows |
    Group-Object settlementDate, settlementPeriod, interconnectorName |
    ForEach-Object {
        $_.Group |
            Sort-Object @{ Expression = { [datetime]$_.publishTime }; Descending = $true } |
            Select-Object -First 1
    }

$processed = $latestByInterconnector |
    Group-Object settlementDate, settlementPeriod |
    ForEach-Object {
        $groupRows = @($_.Group)
        $first = $groupRows | Sort-Object settlementPeriod | Select-Object -First 1
        $flow = ($groupRows | Measure-Object -Property generation -Sum).Sum

        [pscustomobject]@{
            settlementDate = $first.settlementDate
            settlementPeriod = [int]$first.settlementPeriod
            netInterconnectorFlow = [decimal]$flow
            interconnectorCount = @($groupRows | Select-Object -ExpandProperty interconnectorName -Unique).Count
            latestPublishTime = (($groupRows | Sort-Object @{ Expression = { [datetime]$_.publishTime }; Descending = $true } | Select-Object -First 1).publishTime)
            startTime = $first.startTime
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
Write-Host "missing SPs count = $($missing.Count)"
if ($missing.Count -gt 0) {
    $missing |
        Select-Object -First 20 |
        Format-Table -AutoSize |
        Out-String -Width 120 |
        Write-Host
}
