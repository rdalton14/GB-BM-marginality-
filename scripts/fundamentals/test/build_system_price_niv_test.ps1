Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Variable = "system_price_niv"
$RawDir = Join-Path $ProjectRoot "data/raw_test/fundamentals/$Variable"
$ProcessedDir = Join-Path $ProjectRoot "data/processed_test"
$ProcessedCsv = Join-Path $ProcessedDir "system_price_niv_test.csv"

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null

$BaseUrl = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices"
$StartDate = [datetime]::ParseExact("2026-01-06", "yyyy-MM-dd", $null)
$EndDate = [datetime]::ParseExact("2026-01-10", "yyyy-MM-dd", $null)

function Invoke-SystemPriceNiv {
    param(
        [Parameter(Mandatory = $true)]
        [datetime] $Day
    )

    $dateText = $Day.ToString("yyyy-MM-dd")
    $uri = "$BaseUrl/$dateText"
    $response = Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }
    return @($response.data)
}

function Get-SystemLongShort {
    param([decimal] $NetImbalanceVolume)

    if ($NetImbalanceVolume -gt 0) {
        return "short"
    }
    if ($NetImbalanceVolume -lt 0) {
        return "long"
    }
    return "balanced"
}

$allRows = New-Object System.Collections.Generic.List[object]
$day = $StartDate
while ($day -le $EndDate) {
    $dateText = $day.ToString("yyyy-MM-dd")
    $rows = @(Invoke-SystemPriceNiv -Day $day)

    foreach ($row in $rows) {
        $allRows.Add([pscustomobject]@{
            settlementDate = $row.settlementDate
            settlementPeriod = [int]$row.settlementPeriod
            startTime = $row.startTime
            createdDateTime = $row.createdDateTime
            systemSellPrice = [decimal]$row.systemSellPrice
            systemBuyPrice = [decimal]$row.systemBuyPrice
            priceDerivationCode = $row.priceDerivationCode
            reserveScarcityPrice = [decimal]$row.reserveScarcityPrice
            netImbalanceVolume = [decimal]$row.netImbalanceVolume
        })
    }

    $rawPath = Join-Path $RawDir "$dateText.csv"
    $rows |
        Sort-Object settlementDate, settlementPeriod |
        Export-Csv -NoTypeInformation -Path $rawPath

    Write-Host "[RAW] $dateText rows=$($rows.Count) path=$rawPath"
    $day = $day.AddDays(1)
}

$processed = $allRows |
    Sort-Object settlementDate, settlementPeriod |
    ForEach-Object {
        $systemPrice = $null
        if ($_.systemBuyPrice -eq $_.systemSellPrice) {
            $systemPrice = $_.systemBuyPrice
        }

        [pscustomobject]@{
            settlementDate = $_.settlementDate
            settlementPeriod = $_.settlementPeriod
            systemBuyPrice = $_.systemBuyPrice
            systemSellPrice = $_.systemSellPrice
            systemPrice = $systemPrice
            netImbalanceVolume = $_.netImbalanceVolume
            systemLongShort = Get-SystemLongShort -NetImbalanceVolume $_.netImbalanceVolume
            priceDerivationCode = $_.priceDerivationCode
            reserveScarcityPrice = $_.reserveScarcityPrice
            createdDateTime = $_.createdDateTime
            startTime = $_.startTime
        }
    }

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

$nonEqualPrices = @(
    $processed |
        Where-Object { [decimal]$_.systemBuyPrice -ne [decimal]$_.systemSellPrice }
)

Write-Host ""
Write-Host "DIAGNOSTICS"
Write-Host "expected rows = days * 48 = $expectedRows"
Write-Host "actual rows = $actualRows"
Write-Host "duplicates on (settlementDate, settlementPeriod) = $($duplicates.Count)"
Write-Host "missing SPs count = $($missing.Count)"
Write-Host "rows where systemBuyPrice != systemSellPrice = $($nonEqualPrices.Count)"
if ($missing.Count -gt 0) {
    $missing |
        Select-Object -First 20 |
        Format-Table -AutoSize |
        Out-String -Width 120 |
        Write-Host
}
