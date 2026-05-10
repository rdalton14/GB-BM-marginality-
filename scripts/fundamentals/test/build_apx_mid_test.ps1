Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Variable = "apx_mid"
$RawDir = Join-Path $ProjectRoot "data/raw_test/fundamentals/$Variable"
$ProcessedDir = Join-Path $ProjectRoot "data/processed_test"
$ProcessedCsv = Join-Path $ProcessedDir "apx_mid_test.csv"

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null

$BaseUrl = "https://data.elexon.co.uk/bmrs/api/v1/datasets/MID"
$StartDate = [datetime]::ParseExact("2026-01-06", "yyyy-MM-dd", $null)
$EndDate = [datetime]::ParseExact("2026-01-10", "yyyy-MM-dd", $null)

function Invoke-ApxMid {
    param(
        [Parameter(Mandatory = $true)]
        [datetime] $Day
    )

    $dateText = $Day.ToString("yyyy-MM-dd")
    $uri = "$BaseUrl" + "?from=$dateText&to=$dateText&settlementPeriodFrom=1&settlementPeriodTo=48&dataProviders=APXMIDP"
    $response = Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }
    return @($response.data)
}

$allRows = New-Object System.Collections.Generic.List[object]
$day = $StartDate
while ($day -le $EndDate) {
    $dateText = $day.ToString("yyyy-MM-dd")
    $rows = @(Invoke-ApxMid -Day $day)

    foreach ($row in $rows) {
        $allRows.Add([pscustomobject]@{
            dataset = $row.dataset
            startTime = $row.startTime
            dataProvider = $row.dataProvider
            settlementDate = $row.settlementDate
            settlementPeriod = [int]$row.settlementPeriod
            price = [decimal]$row.price
            volume = [decimal]$row.volume
        })
    }

    $rawPath = Join-Path $RawDir "$dateText.csv"
    $rows |
        Select-Object dataset, startTime, dataProvider, settlementDate, settlementPeriod, price, volume |
        Sort-Object settlementDate, settlementPeriod |
        Export-Csv -NoTypeInformation -Path $rawPath

    Write-Host "[RAW] $dateText rows=$($rows.Count) path=$rawPath"
    $day = $day.AddDays(1)
}

$processed = $allRows |
    Where-Object { $_.dataProvider -eq "APXMIDP" } |
    Sort-Object settlementDate, settlementPeriod |
    Select-Object settlementDate,
        settlementPeriod,
        @{ Name = "marketIndexPrice"; Expression = { $_.price } },
        @{ Name = "marketIndexVolume"; Expression = { $_.volume } },
        dataProvider,
        startTime

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
