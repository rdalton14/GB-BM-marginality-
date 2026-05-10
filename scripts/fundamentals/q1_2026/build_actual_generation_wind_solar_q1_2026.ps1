Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$Variable = "actual_generation_wind_solar"
$RawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$Variable"
$ProcessedDir = Join-Path $ProjectRoot "data/processed/q1_2026/fundamentals"
$DiagnosticsCoverageDir = Join-Path $ProjectRoot "data/diagnostics/coverage"
$DiagnosticsMissingnessDir = Join-Path $ProjectRoot "data/diagnostics/missingness"
$ProcessedCsv = Join-Path $ProcessedDir "actual_generation_wind_solar_q1_2026.csv"
$DailyCompletenessCsv = Join-Path $DiagnosticsCoverageDir "actual_generation_wind_solar_q1_2026_daily_completeness.csv"
$MissingSpsCsv = Join-Path $DiagnosticsMissingnessDir "actual_generation_wind_solar_q1_2026_missing_sps.csv"

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null
New-Item -ItemType Directory -Force -Path $DiagnosticsCoverageDir | Out-Null
New-Item -ItemType Directory -Force -Path $DiagnosticsMissingnessDir | Out-Null

$BaseUrl = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type/wind-and-solar"
$StartDate = [datetime]::ParseExact("2026-01-01", "yyyy-MM-dd", $null)
$EndDate = [datetime]::ParseExact("2026-03-31", "yyyy-MM-dd", $null)

function Invoke-ActualGenerationWindSolar {
    param(
        [Parameter(Mandatory = $true)]
        [datetime] $Day
    )

    $dateText = $Day.ToString("yyyy-MM-dd")
    $uri = "$BaseUrl" + "?from=$dateText&to=$dateText&settlementPeriodFrom=1&settlementPeriodTo=48"
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
    $rawPath = Join-Path $RawDir "$dateText.csv"

    if (Test-Path $rawPath) {
        $rows = @(Import-Csv $rawPath)
        Write-Host "[RAW] $dateText rows=$($rows.Count) path=$rawPath (existing)"
    }
    else {
        $rows = @(Invoke-ActualGenerationWindSolar -Day $day)
        $rows |
            Select-Object publishTime, businessType, psrType, quantity, startTime, settlementDate, settlementPeriod |
            Sort-Object settlementDate, settlementPeriod, psrType, publishTime |
            Export-Csv -NoTypeInformation -Path $rawPath

        Write-Host "[RAW] $dateText rows=$($rows.Count) path=$rawPath"
    }

    foreach ($row in $rows) {
        $allRows.Add([pscustomobject]@{
            publishTime = $row.publishTime
            businessType = $row.businessType
            psrType = $row.psrType
            quantity = [decimal]$row.quantity
            startTime = $row.startTime
            settlementDate = $row.settlementDate
            settlementPeriod = [int]$row.settlementPeriod
        })
    }

    $day = $day.AddDays(1)
}

$latestByPsrType = $allRows |
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
            windOnshore = $windOnshore
            windOffshore = $windOffshore
            solar = $solar
            windTotal = $windTotal
            windSolarTotal = $windSolarTotal
            windOnshore_publishTime = Get-PublishTimeByPsrType -Rows $groupRows -PsrType "Wind Onshore"
            windOffshore_publishTime = Get-PublishTimeByPsrType -Rows $groupRows -PsrType "Wind Offshore"
            solar_publishTime = Get-PublishTimeByPsrType -Rows $groupRows -PsrType "Solar"
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
