Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Variable = "day_ahead_generation_wind_solar"
$RawDir = Join-Path $ProjectRoot "data/raw_test/fundamentals/$Variable"
$ProcessedDir = Join-Path $ProjectRoot "data/processed_test"
$ProcessedCsv = Join-Path $ProcessedDir "day_ahead_generation_wind_solar_test.csv"

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null

$BaseUrl = "https://data.elexon.co.uk/bmrs/api/v1/forecast/generation/wind-and-solar/day-ahead"
$StartDate = [datetime]::ParseExact("2026-01-06", "yyyy-MM-dd", $null)
$EndDate = [datetime]::ParseExact("2026-01-10", "yyyy-MM-dd", $null)

function Invoke-DayAheadGenerationWindSolar {
    param(
        [Parameter(Mandatory = $true)]
        [datetime] $Day
    )

    $dateText = $Day.ToString("yyyy-MM-dd")
    $uri = "$BaseUrl" + "?from=$dateText&to=$dateText&settlementPeriodFrom=1&settlementPeriodTo=48&processType=all"
    $response = Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }
    return @($response.data)
}

function Get-QuantityByPsrType {
    param(
        [object[]] $Rows,
        [string] $PsrType
    )

    $match = @($Rows | Where-Object { $_.psrType -eq $PsrType } | Select-Object -First 1)
    if ($match.Count -eq 0) {
        return $null
    }
    return [decimal]$match[0].quantity
}

function Get-PublishTimeByPsrType {
    param(
        [object[]] $Rows,
        [string] $PsrType
    )

    $match = @($Rows | Where-Object { $_.psrType -eq $PsrType } | Select-Object -First 1)
    if ($match.Count -eq 0) {
        return $null
    }
    return $match[0].publishTime
}

$allRows = New-Object System.Collections.Generic.List[object]
$day = $StartDate
while ($day -le $EndDate) {
    $dateText = $day.ToString("yyyy-MM-dd")
    $rows = @(Invoke-DayAheadGenerationWindSolar -Day $day)

    foreach ($row in $rows) {
        $allRows.Add([pscustomobject]@{
            publishTime = $row.publishTime
            processType = $row.processType
            businessType = $row.businessType
            psrType = $row.psrType
            startTime = $row.startTime
            settlementDate = $row.settlementDate
            settlementPeriod = [int]$row.settlementPeriod
            quantity = [decimal]$row.quantity
        })
    }

    $rawPath = Join-Path $RawDir "$dateText.csv"
    $rows |
        Select-Object publishTime, processType, businessType, psrType, startTime, settlementDate, settlementPeriod, quantity |
        Sort-Object settlementDate, settlementPeriod, processType, psrType, publishTime |
        Export-Csv -NoTypeInformation -Path $rawPath

    Write-Host "[RAW] $dateText rows=$($rows.Count) path=$rawPath"
    $day = $day.AddDays(1)
}

$dayAheadRows = @($allRows | Where-Object { $_.processType -eq "Day ahead" })

$latestByPsrType = $dayAheadRows |
    Group-Object settlementDate, settlementPeriod, psrType |
    ForEach-Object {
        $_.Group |
            Sort-Object @{ Expression = { [datetime]$_.publishTime }; Descending = $true } |
            Select-Object -First 1
    }

$processed = $latestByPsrType |
    Group-Object settlementDate, settlementPeriod |
    ForEach-Object {
        $groupRows = @($_.Group)
        $first = $groupRows | Sort-Object settlementPeriod | Select-Object -First 1

        $windOnshore = Get-QuantityByPsrType -Rows $groupRows -PsrType "Wind Onshore"
        $windOffshore = Get-QuantityByPsrType -Rows $groupRows -PsrType "Wind Offshore"
        $solar = Get-QuantityByPsrType -Rows $groupRows -PsrType "Solar"

        $windTotal = $null
        if ($null -ne $windOnshore -or $null -ne $windOffshore) {
            $windOnshoreValue = 0
            $windOffshoreValue = 0
            if ($null -ne $windOnshore) {
                $windOnshoreValue = $windOnshore
            }
            if ($null -ne $windOffshore) {
                $windOffshoreValue = $windOffshore
            }
            $windTotal = [decimal]($windOnshoreValue + $windOffshoreValue)
        }

        $windSolarTotal = $null
        if ($null -ne $windTotal -or $null -ne $solar) {
            $windValue = 0
            $solarValue = 0
            if ($null -ne $windTotal) {
                $windValue = $windTotal
            }
            if ($null -ne $solar) {
                $solarValue = $solar
            }
            $windSolarTotal = [decimal]($windValue + $solarValue)
        }

        [pscustomobject]@{
            settlementDate = $first.settlementDate
            settlementPeriod = [int]$first.settlementPeriod
            dayAheadWindOnshoreForecast = $windOnshore
            dayAheadWindOffshoreForecast = $windOffshore
            dayAheadSolarForecast = $solar
            dayAheadWindTotalForecast = $windTotal
            dayAheadWindSolarTotalForecast = $windSolarTotal
            dayAheadWindOnshoreForecast_publishTime = Get-PublishTimeByPsrType -Rows $groupRows -PsrType "Wind Onshore"
            dayAheadWindOffshoreForecast_publishTime = Get-PublishTimeByPsrType -Rows $groupRows -PsrType "Wind Offshore"
            dayAheadSolarForecast_publishTime = Get-PublishTimeByPsrType -Rows $groupRows -PsrType "Solar"
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
Write-Host "raw rows = $($allRows.Count)"
Write-Host "day-ahead raw rows after processType filter = $($dayAheadRows.Count)"
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
