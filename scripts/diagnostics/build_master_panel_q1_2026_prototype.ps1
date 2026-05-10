Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$FundamentalsDir = Join-Path $ProjectRoot "data/processed/q1_2026/fundamentals"
$ProcessedDir = Join-Path $ProjectRoot "data/processed/q1_2026"
$DiagnosticsDir = Join-Path $ProjectRoot "data/diagnostics/audits"
$PythonExe = Join-Path $ProjectRoot ".venv/Scripts/python.exe"

$OutputCsv = Join-Path $ProcessedDir "master_panel_q1_2026_prototype.csv"
$OutputParquet = Join-Path $ProcessedDir "master_panel_q1_2026_prototype.parquet"
$MissingnessCsv = Join-Path $DiagnosticsDir "master_panel_q1_2026_missingness.csv"
$MergeLogCsv = Join-Path $DiagnosticsDir "master_panel_q1_2026_merge_log.csv"

New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null
New-Item -ItemType Directory -Force -Path $DiagnosticsDir | Out-Null

function Assert-UniqueKeys {
    param(
        [Parameter(Mandatory = $true)]
        [object[]] $Rows,
        [Parameter(Mandatory = $true)]
        [string] $DatasetName
    )

    $dups = @(
        $Rows |
            Group-Object settlementDate, settlementPeriod |
            Where-Object { $_.Count -gt 1 }
    )

    if ($dups.Count -gt 0) {
        throw "$DatasetName has $($dups.Count) duplicate keys on (settlementDate, settlementPeriod)"
    }
}

function Make-Key {
    param(
        [Parameter(Mandatory = $true)]
        [string] $SettlementDate,
        [Parameter(Mandatory = $true)]
        [int] $SettlementPeriod
    )

    return "$SettlementDate|$SettlementPeriod"
}

function Get-MissingSummary {
    param(
        [Parameter(Mandatory = $true)]
        [object[]] $Rows
    )

    $cols = @($Rows[0].PSObject.Properties.Name)
    $out = foreach ($col in $cols) {
        [pscustomobject]@{
            column = $col
            missingValues = @(
                $Rows |
                    Where-Object { [string]::IsNullOrWhiteSpace([string]($_.$col)) }
            ).Count
        }
    }
    return $out
}

function Get-NewMissingCount {
    param(
        [Parameter(Mandatory = $true)]
        [object[]] $Rows,
        [Parameter(Mandatory = $true)]
        [string[]] $Columns
    )

    $count = 0
    foreach ($col in $Columns) {
        $count += @(
            $Rows |
                Where-Object { [string]::IsNullOrWhiteSpace([string]($_.$col)) }
        ).Count
    }
    return $count
}

function Add-MergeLogEntry {
    param(
        [System.Collections.Generic.List[object]] $Log,
        [string] $Dataset,
        [int] $RowsBefore,
        [int] $RowsAfter,
        [int] $DuplicatesAfter,
        [int] $NewMissingValues,
        [string] $Status
    )

    $Log.Add([pscustomobject]@{
        dataset = $Dataset
        rowsBefore = $RowsBefore
        rowsAfter = $RowsAfter
        duplicatesAfter = $DuplicatesAfter
        newMissingValues = $NewMissingValues
        status = $Status
    }) | Out-Null
}

function Left-JoinRows {
    param(
        [Parameter(Mandatory = $true)]
        [object[]] $BaseRows,
        [Parameter(Mandatory = $true)]
        [object[]] $IncomingRows,
        [Parameter(Mandatory = $true)]
        [string] $DatasetName,
        [Parameter(Mandatory = $true)]
        [string[]] $ColumnsToAdd,
        [System.Collections.Generic.List[object]] $MergeLog
    )

    Assert-UniqueKeys -Rows $IncomingRows -DatasetName $DatasetName

    $before = $BaseRows.Count
    $lookup = @{}
    foreach ($row in $IncomingRows) {
        $lookup[(Make-Key -SettlementDate $row.settlementDate -SettlementPeriod ([int]$row.settlementPeriod))] = $row
    }

    $joined = foreach ($row in $BaseRows) {
        $copy = [ordered]@{}
        foreach ($prop in $row.PSObject.Properties.Name) {
            $copy[$prop] = $row.$prop
        }

        $key = Make-Key -SettlementDate $row.settlementDate -SettlementPeriod ([int]$row.settlementPeriod)
        if ($lookup.ContainsKey($key)) {
            $match = $lookup[$key]
            foreach ($col in $ColumnsToAdd) {
                $copy[$col] = $match.$col
            }
        }
        else {
            foreach ($col in $ColumnsToAdd) {
                $copy[$col] = $null
            }
        }

        [pscustomobject]$copy
    }

    $after = @($joined).Count
    $dupAfter = @(
        $joined |
            Group-Object settlementDate, settlementPeriod |
            Where-Object { $_.Count -gt 1 }
    ).Count
    $newMissing = Get-NewMissingCount -Rows $joined -Columns $ColumnsToAdd

    Write-Host ""
    Write-Host "[MERGE] $DatasetName"
    Write-Host "rows before = $before"
    Write-Host "rows after = $after"
    Write-Host "duplicates after = $dupAfter"
    Write-Host "newly introduced missing values = $newMissing"

    if ($after -ne $before) {
        Add-MergeLogEntry -Log $MergeLog -Dataset $DatasetName -RowsBefore $before -RowsAfter $after -DuplicatesAfter $dupAfter -NewMissingValues $newMissing -Status "row_count_changed"
        throw "Merge with $DatasetName changed row count from $before to $after"
    }

    Add-MergeLogEntry -Log $MergeLog -Dataset $DatasetName -RowsBefore $before -RowsAfter $after -DuplicatesAfter $dupAfter -NewMissingValues $newMissing -Status "ok"
    return @($joined)
}

$mergeLog = New-Object 'System.Collections.Generic.List[object]'

$spinePath = Join-Path $FundamentalsDir "system_price_niv_q1_2026.csv"
$spine = @(Import-Csv $spinePath)
Assert-UniqueKeys -Rows $spine -DatasetName "system_price_niv_q1_2026"

$mismatchCount = @(
    $spine |
        Where-Object { [decimal]$_.systemBuyPrice -ne [decimal]$_.systemSellPrice }
).Count

Write-Host "SPINE = system_price_niv_q1_2026.csv"
Write-Host "spine rows = $($spine.Count)"
Write-Host "systemBuyPrice != systemSellPrice rows = $mismatchCount"

$panel = $spine | Sort-Object settlementDate, settlementPeriod | ForEach-Object {
    $systemPrice = $null
    if ([decimal]$_.systemBuyPrice -eq [decimal]$_.systemSellPrice) {
        $systemPrice = [decimal]$_.systemBuyPrice
    }

    [pscustomobject][ordered]@{
        settlementDate = $_.settlementDate
        settlementPeriod = [int]$_.settlementPeriod
        systemBuyPrice = [decimal]$_.systemBuyPrice
        systemSellPrice = [decimal]$_.systemSellPrice
        systemPrice = $systemPrice
        netImbalanceVolume = [decimal]$_.netImbalanceVolume
        systemLongShort = $_.systemLongShort
        priceDerivationCode = $_.priceDerivationCode
        reserveScarcityPrice = [decimal]$_.reserveScarcityPrice
        createdDateTime = $_.createdDateTime
        startTime = $_.startTime
    }
}

$actualLoad = @(Import-Csv (Join-Path $FundamentalsDir "actual_total_load_q1_2026.csv") | ForEach-Object {
    [pscustomobject]@{
        settlementDate = $_.settlementDate
        settlementPeriod = [int]$_.settlementPeriod
        demandOutturn = [decimal]$_.ActualTotalLoad
    }
})
$panel = Left-JoinRows -BaseRows $panel -IncomingRows $actualLoad -DatasetName "actual_total_load" -ColumnsToAdd @("demandOutturn") -MergeLog $mergeLog

$demandForecast = @(Import-Csv (Join-Path $FundamentalsDir "day_ahead_national_demand_forecast_q1_2026.csv") | ForEach-Object {
    [pscustomobject]@{
        settlementDate = $_.settlementDate
        settlementPeriod = [int]$_.settlementPeriod
        dayAheadNationalDemandForecast = [decimal]$_.dayAheadNationalDemandForecast
    }
})
$panel = Left-JoinRows -BaseRows $panel -IncomingRows $demandForecast -DatasetName "day_ahead_national_demand_forecast" -ColumnsToAdd @("dayAheadNationalDemandForecast") -MergeLog $mergeLog

$actualWindSolar = @(Import-Csv (Join-Path $FundamentalsDir "actual_generation_wind_solar_q1_2026.csv") | ForEach-Object {
    [pscustomobject]@{
        settlementDate = $_.settlementDate
        settlementPeriod = [int]$_.settlementPeriod
        windGeneration_actual = [decimal]$_.windTotal
        windOnshoreGeneration_actual = [decimal]$_.windOnshore
        windOffshoreGeneration_actual = [decimal]$_.windOffshore
        solarGeneration_actual = [decimal]$_.solar
    }
})
$panel = Left-JoinRows -BaseRows $panel -IncomingRows $actualWindSolar -DatasetName "actual_generation_wind_solar" -ColumnsToAdd @("windGeneration_actual","windOnshoreGeneration_actual","windOffshoreGeneration_actual","solarGeneration_actual") -MergeLog $mergeLog

$forecastWindSolar = @(Import-Csv (Join-Path $FundamentalsDir "day_ahead_generation_wind_solar_q1_2026.csv") | ForEach-Object {
    [pscustomobject]@{
        settlementDate = $_.settlementDate
        settlementPeriod = [int]$_.settlementPeriod
        windForecast = [decimal]$_.dayAheadWindTotalForecast
        windOnshoreForecast = [decimal]$_.dayAheadWindOnshoreForecast
        windOffshoreForecast = [decimal]$_.dayAheadWindOffshoreForecast
        solarForecast = [decimal]$_.dayAheadSolarForecast
    }
})
$panel = Left-JoinRows -BaseRows $panel -IncomingRows $forecastWindSolar -DatasetName "day_ahead_generation_wind_solar" -ColumnsToAdd @("windForecast","windOnshoreForecast","windOffshoreForecast","solarForecast") -MergeLog $mergeLog

$interconnector = @(Import-Csv (Join-Path $FundamentalsDir "interconnector_flows_q1_2026.csv") | ForEach-Object {
    [pscustomobject]@{
        settlementDate = $_.settlementDate
        settlementPeriod = [int]$_.settlementPeriod
        netInterconnectorFlow = [decimal]$_.netInterconnectorFlow
    }
})
$panel = Left-JoinRows -BaseRows $panel -IncomingRows $interconnector -DatasetName "interconnector_flows" -ColumnsToAdd @("netInterconnectorFlow") -MergeLog $mergeLog

$lolpdrm = @(Import-Csv (Join-Path $FundamentalsDir "lolpdrm_q1_2026.csv") | ForEach-Object {
    [pscustomobject]@{
        settlementDate = $_.settlementDate
        settlementPeriod = [int]$_.settlementPeriod
        deratedMargin = [decimal]$_.deratedMargin
        lossOfLoadProbability = [decimal]$_.lossOfLoadProbability
    }
})
$panel = Left-JoinRows -BaseRows $panel -IncomingRows $lolpdrm -DatasetName "lolpdrm" -ColumnsToAdd @("deratedMargin","lossOfLoadProbability") -MergeLog $mergeLog

$apx = @(Import-Csv (Join-Path $FundamentalsDir "apx_mid_q1_2026.csv") | ForEach-Object {
    [pscustomobject]@{
        settlementDate = $_.settlementDate
        settlementPeriod = [int]$_.settlementPeriod
        dayAheadPrice = [decimal]$_.marketIndexPrice
        dayAheadVolume = [decimal]$_.marketIndexVolume
    }
})
$panel = Left-JoinRows -BaseRows $panel -IncomingRows $apx -DatasetName "apx_mid" -ColumnsToAdd @("dayAheadPrice","dayAheadVolume") -MergeLog $mergeLog

$panel = $panel | Sort-Object settlementDate, settlementPeriod

$previousSystemPrice = $null
$engineered = foreach ($row in $panel) {
    $sp = [int]$row.settlementPeriod
    $hour = [math]::Floor(($sp - 1) / 2)
    $date = [datetime]::ParseExact($row.settlementDate, "yyyy-MM-dd", $null)
    $radians = 2 * [math]::PI * (($sp - 1) / 48.0)

    $demandForecastError = $null
    if ($null -ne $row.dayAheadNationalDemandForecast -and $null -ne $row.demandOutturn -and "$($row.dayAheadNationalDemandForecast)" -ne "" -and "$($row.demandOutturn)" -ne "") {
        $demandForecastError = [decimal]$row.dayAheadNationalDemandForecast - [decimal]$row.demandOutturn
    }

    $windForecastError = $null
    if ($null -ne $row.windForecast -and $null -ne $row.windGeneration_actual -and "$($row.windForecast)" -ne "" -and "$($row.windGeneration_actual)" -ne "") {
        $windForecastError = [decimal]$row.windForecast - [decimal]$row.windGeneration_actual
    }

    $solarForecastError = $null
    if ($null -ne $row.solarForecast -and $null -ne $row.solarGeneration_actual -and "$($row.solarForecast)" -ne "" -and "$($row.solarGeneration_actual)" -ne "") {
        $solarForecastError = [decimal]$row.solarForecast - [decimal]$row.solarGeneration_actual
    }

    $lolpEvent = $null
    if ($null -ne $row.lossOfLoadProbability -and "$($row.lossOfLoadProbability)" -ne "") {
        if ([decimal]$row.lossOfLoadProbability -gt 0) {
            $lolpEvent = 1
        }
        else {
            $lolpEvent = 0
        }
    }

    $lagSystemPrice = $previousSystemPrice
    if ($null -ne $row.systemPrice -and "$($row.systemPrice)" -ne "") {
        $previousSystemPrice = [decimal]$row.systemPrice
    }
    else {
        $previousSystemPrice = $null
    }

    [pscustomobject][ordered]@{
        settlementDate = $row.settlementDate
        settlementPeriod = $sp
        systemBuyPrice = $row.systemBuyPrice
        systemSellPrice = $row.systemSellPrice
        systemPrice = $row.systemPrice
        netImbalanceVolume = $row.netImbalanceVolume
        systemLongShort = $row.systemLongShort
        lag_systemPrice = $lagSystemPrice
        demandOutturn = $row.demandOutturn
        dayAheadNationalDemandForecast = $row.dayAheadNationalDemandForecast
        demandForecastError = $demandForecastError
        windGeneration_actual = $row.windGeneration_actual
        windOnshoreGeneration_actual = $row.windOnshoreGeneration_actual
        windOffshoreGeneration_actual = $row.windOffshoreGeneration_actual
        solarGeneration_actual = $row.solarGeneration_actual
        windForecast = $row.windForecast
        windOnshoreForecast = $row.windOnshoreForecast
        windOffshoreForecast = $row.windOffshoreForecast
        solarForecast = $row.solarForecast
        windForecastError = $windForecastError
        solarForecastError = $solarForecastError
        netInterconnectorFlow = $row.netInterconnectorFlow
        deratedMargin = $row.deratedMargin
        lossOfLoadProbability = $row.lossOfLoadProbability
        lolp_event = $lolpEvent
        dayAheadPrice = $row.dayAheadPrice
        dayAheadVolume = $row.dayAheadVolume
        hour = $hour
        dayOfWeek = [int]$date.DayOfWeek
        isWeekend = [int]([int]$date.DayOfWeek -in 0,6)
        month = $date.Month
        quarter = [math]::Ceiling($date.Month / 3.0)
        year = $date.Year
        settlementPeriod_sin = [math]::Sin($radians)
        settlementPeriod_cos = [math]::Cos($radians)
    }
}

$engineered = @($engineered)

$missingness = Get-MissingSummary -Rows $engineered
$missingness | Export-Csv -NoTypeInformation -Path $MissingnessCsv
$mergeLog | Export-Csv -NoTypeInformation -Path $MergeLogCsv
$engineered | Export-Csv -NoTypeInformation -Path $OutputCsv

$parquetWritten = $false
if (Test-Path $PythonExe) {
    $parquetCode = @"
import pandas as pd
from pathlib import Path

csv_path = Path(r'$OutputCsv')
parquet_path = Path(r'$OutputParquet')
df = pd.read_csv(csv_path)
df.to_parquet(parquet_path, index=False)
"@

    try {
        & $PythonExe -c $parquetCode
        if ($LASTEXITCODE -eq 0 -and (Test-Path $OutputParquet)) {
            $parquetWritten = $true
        }
    }
    catch {
        $parquetWritten = $false
    }
}

Write-Host ""
Write-Host "PROTOTYPE DIAGNOSTICS"
Write-Host "shape = $($engineered.Count) rows x $(@($engineered[0].PSObject.Properties.Name).Count) columns"
Write-Host "columns:"
$engineered[0].PSObject.Properties.Name | ForEach-Object { Write-Host " - $_" }
Write-Host "min settlementDate = $(($engineered | Measure-Object -Property settlementDate -Minimum).Minimum)"
Write-Host "max settlementDate = $(($engineered | Measure-Object -Property settlementDate -Maximum).Maximum)"
Write-Host "duplicate count on (settlementDate, settlementPeriod) = $(@($engineered | Group-Object settlementDate,settlementPeriod | Where-Object Count -gt 1).Count)"
Write-Host "missing values by column:"
$missingness | Format-Table -AutoSize | Out-String -Width 160 | Write-Host
Write-Host "first 10 rows:"
$engineered | Select-Object -First 10 | Format-Table -AutoSize | Out-String -Width 220 | Write-Host

$rowsWithAnyMissing = @(
    $engineered |
        Where-Object {
            foreach ($prop in $_.PSObject.Properties.Name) {
                if ([string]::IsNullOrWhiteSpace([string]($_.$prop))) {
                    return $true
                }
            }
            return $false
        }
)

Write-Host "rows with any missing values = $($rowsWithAnyMissing.Count)"

$corrVars = @("systemPrice","netImbalanceVolume","dayAheadPrice","demandForecastError","windForecastError","solarForecastError","lossOfLoadProbability","deratedMargin")
$corrRows = foreach ($x in $corrVars) {
    foreach ($y in $corrVars) {
        $pairs = @(
            $engineered |
                Where-Object {
                    -not [string]::IsNullOrWhiteSpace([string]($_.$x)) -and
                    -not [string]::IsNullOrWhiteSpace([string]($_.$y))
                }
        )

        if ($pairs.Count -lt 2) {
            $corr = $null
        }
        else {
            $xs = @($pairs | ForEach-Object { [double]($_.$x) })
            $ys = @($pairs | ForEach-Object { [double]($_.$y) })
            $xMean = ($xs | Measure-Object -Average).Average
            $yMean = ($ys | Measure-Object -Average).Average
            $num = 0.0
            $xDen = 0.0
            $yDen = 0.0
            for ($i = 0; $i -lt $xs.Count; $i++) {
                $dx = $xs[$i] - $xMean
                $dy = $ys[$i] - $yMean
                $num += $dx * $dy
                $xDen += $dx * $dx
                $yDen += $dy * $dy
            }
            if ($xDen -eq 0 -or $yDen -eq 0) {
                $corr = $null
            }
            else {
                $corr = $num / [math]::Sqrt($xDen * $yDen)
            }
        }

        [pscustomobject]@{
            var1 = $x
            var2 = $y
            correlation = $corr
        }
    }
}

Write-Host "correlation preview:"
$corrRows | Format-Table -AutoSize | Out-String -Width 160 | Write-Host

Write-Host "CSV saved to $OutputCsv"
if ($parquetWritten) {
    Write-Host "Parquet saved to $OutputParquet"
}
Write-Host "Missingness saved to $MissingnessCsv"
Write-Host "Merge log saved to $MergeLogCsv"

if (-not $parquetWritten) {
    Write-Warning "Parquet export was not written because the local Python/parquet export step failed."
}
