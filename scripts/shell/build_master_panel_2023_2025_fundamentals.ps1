Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$FundamentalsDir = Join-Path $ProjectRoot "data/processed/full_2023_2025/fundamentals"
$ProcessedDir = Join-Path $ProjectRoot "data/processed/full_2023_2025"
$DiagnosticsDir = Join-Path $ProjectRoot "data/diagnostics/audits"
$PythonExe = Join-Path $ProjectRoot ".venv/Scripts/python.exe"

$OutputCsv = Join-Path $ProcessedDir "master_panel_2023_2025_fundamentals.csv"
$OutputParquet = Join-Path $ProcessedDir "master_panel_2023_2025_fundamentals.parquet"
$MissingnessCsv = Join-Path $DiagnosticsDir "master_panel_2023_2025_fundamentals_missingness.csv"
$MergeLogCsv = Join-Path $DiagnosticsDir "master_panel_2023_2025_fundamentals_merge_log.csv"

New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null
New-Item -ItemType Directory -Force -Path $DiagnosticsDir | Out-Null

function Assert-UniqueKeys {
    param([object[]] $Rows, [string] $DatasetName)
    $dups = @($Rows | Group-Object settlementDate, settlementPeriod | Where-Object { $_.Count -gt 1 })
    if ($dups.Count -gt 0) {
        throw "$DatasetName has $($dups.Count) duplicate keys on (settlementDate, settlementPeriod)"
    }
}

function Make-Key {
    param([string] $SettlementDate, [int] $SettlementPeriod)
    return "$SettlementDate|$SettlementPeriod"
}

function Get-MissingSummary {
    param([object[]] $Rows)
    $cols = @($Rows[0].PSObject.Properties.Name)
    foreach ($col in $cols) {
        [pscustomobject]@{
            column = $col
            missingValues = @($Rows | Where-Object { [string]::IsNullOrWhiteSpace([string]($_.$col)) }).Count
        }
    }
}

function Get-NewMissingCount {
    param([object[]] $Rows, [string[]] $Columns)
    $count = 0
    foreach ($col in $Columns) {
        $count += @($Rows | Where-Object { [string]::IsNullOrWhiteSpace([string]($_.$col)) }).Count
    }
    return $count
}

function Add-MergeLogEntry {
    param([System.Collections.Generic.List[object]] $Log, [string] $Dataset, [int] $RowsBefore, [int] $RowsAfter, [int] $DuplicatesAfter, [int] $NewMissingValues, [string] $Status)
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
    param([object[]] $BaseRows, [object[]] $IncomingRows, [string] $DatasetName, [string[]] $ColumnsToAdd, [System.Collections.Generic.List[object]] $MergeLog)

    Assert-UniqueKeys -Rows $IncomingRows -DatasetName $DatasetName
    $before = $BaseRows.Count
    $lookup = @{}
    foreach ($row in $IncomingRows) {
        $lookup[(Make-Key -SettlementDate $row.settlementDate -SettlementPeriod ([int]$row.settlementPeriod))] = $row
    }

    $joined = foreach ($row in $BaseRows) {
        $copy = [ordered]@{}
        foreach ($prop in $row.PSObject.Properties.Name) { $copy[$prop] = $row.$prop }
        $key = Make-Key -SettlementDate $row.settlementDate -SettlementPeriod ([int]$row.settlementPeriod)
        if ($lookup.ContainsKey($key)) {
            $match = $lookup[$key]
            foreach ($col in $ColumnsToAdd) { $copy[$col] = $match.$col }
        }
        else {
            foreach ($col in $ColumnsToAdd) { $copy[$col] = $null }
        }
        [pscustomobject]$copy
    }

    $after = @($joined).Count
    $dupAfter = @($joined | Group-Object settlementDate, settlementPeriod | Where-Object { $_.Count -gt 1 }).Count
    $newMissing = Get-NewMissingCount -Rows $joined -Columns $ColumnsToAdd
    Write-Host "[MERGE] $DatasetName before=$before after=$after duplicates=$dupAfter newMissing=$newMissing"

    if ($after -ne $before) {
        Add-MergeLogEntry -Log $MergeLog -Dataset $DatasetName -RowsBefore $before -RowsAfter $after -DuplicatesAfter $dupAfter -NewMissingValues $newMissing -Status "row_count_changed"
        throw "Merge with $DatasetName changed row count from $before to $after"
    }

    Add-MergeLogEntry -Log $MergeLog -Dataset $DatasetName -RowsBefore $before -RowsAfter $after -DuplicatesAfter $dupAfter -NewMissingValues $newMissing -Status "ok"
    return @($joined)
}

$mergeLog = New-Object 'System.Collections.Generic.List[object]'

$spine = @(Import-Csv (Join-Path $FundamentalsDir "system_price_niv_2023_2025.csv"))
Assert-UniqueKeys -Rows $spine -DatasetName "system_price_niv_2023_2025"

$panel = $spine |
    Sort-Object settlementDate, settlementPeriod |
    ForEach-Object {
        [pscustomobject][ordered]@{
            settlementDate = $_.settlementDate
            settlementPeriod = [int]$_.settlementPeriod
            systemBuyPrice = [decimal]$_.systemBuyPrice
            systemSellPrice = [decimal]$_.systemSellPrice
            systemPrice = if ([decimal]$_.systemBuyPrice -eq [decimal]$_.systemSellPrice) { [decimal]$_.systemBuyPrice } else { $null }
            netImbalanceVolume = [decimal]$_.netImbalanceVolume
            systemLongShort = $_.systemLongShort
        }
    }

$actualLoad = @(Import-Csv (Join-Path $FundamentalsDir "actual_total_load_2023_2025.csv") | ForEach-Object {
    [pscustomobject]@{ settlementDate = $_.settlementDate; settlementPeriod = [int]$_.settlementPeriod; demandOutturn = [decimal]$_.ActualTotalLoad }
})
$panel = Left-JoinRows -BaseRows $panel -IncomingRows $actualLoad -DatasetName "actual_total_load" -ColumnsToAdd @("demandOutturn") -MergeLog $mergeLog

$demandForecast = @(Import-Csv (Join-Path $FundamentalsDir "day_ahead_national_demand_forecast_2023_2025.csv") | ForEach-Object {
    [pscustomobject]@{ settlementDate = $_.settlementDate; settlementPeriod = [int]$_.settlementPeriod; dayAheadNationalDemandForecast = [decimal]$_.dayAheadNationalDemandForecast }
})
$panel = Left-JoinRows -BaseRows $panel -IncomingRows $demandForecast -DatasetName "day_ahead_national_demand_forecast" -ColumnsToAdd @("dayAheadNationalDemandForecast") -MergeLog $mergeLog

$actualWindSolar = @(Import-Csv (Join-Path $FundamentalsDir "actual_generation_wind_solar_2023_2025.csv") | ForEach-Object {
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

$forecastWindSolar = @(Import-Csv (Join-Path $FundamentalsDir "day_ahead_generation_wind_solar_2023_2025.csv") | ForEach-Object {
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

$interconnector = @(Import-Csv (Join-Path $FundamentalsDir "interconnector_flows_2023_2025.csv") | ForEach-Object {
    [pscustomobject]@{ settlementDate = $_.settlementDate; settlementPeriod = [int]$_.settlementPeriod; netInterconnectorFlow = [decimal]$_.netInterconnectorFlow }
})
$panel = Left-JoinRows -BaseRows $panel -IncomingRows $interconnector -DatasetName "interconnector_flows" -ColumnsToAdd @("netInterconnectorFlow") -MergeLog $mergeLog

$lolpdrm = @(Import-Csv (Join-Path $FundamentalsDir "lolpdrm_2023_2025.csv") | ForEach-Object {
    [pscustomobject]@{
        settlementDate = $_.settlementDate
        settlementPeriod = [int]$_.settlementPeriod
        deratedMargin = [decimal]$_.deratedMargin
        lossOfLoadProbability = [decimal]$_.lossOfLoadProbability
    }
})
$panel = Left-JoinRows -BaseRows $panel -IncomingRows $lolpdrm -DatasetName "lolpdrm" -ColumnsToAdd @("deratedMargin","lossOfLoadProbability") -MergeLog $mergeLog

$apx = @(Import-Csv (Join-Path $FundamentalsDir "apx_mid_2023_2025.csv") | ForEach-Object {
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
    $demandForecastError = if ($null -ne $row.dayAheadNationalDemandForecast -and $null -ne $row.demandOutturn -and "$($row.dayAheadNationalDemandForecast)" -ne "" -and "$($row.demandOutturn)" -ne "") { [decimal]$row.dayAheadNationalDemandForecast - [decimal]$row.demandOutturn } else { $null }
    $windForecastError = if ($null -ne $row.windForecast -and $null -ne $row.windGeneration_actual -and "$($row.windForecast)" -ne "" -and "$($row.windGeneration_actual)" -ne "") { [decimal]$row.windForecast - [decimal]$row.windGeneration_actual } else { $null }
    $solarForecastError = if ($null -ne $row.solarForecast -and $null -ne $row.solarGeneration_actual -and "$($row.solarForecast)" -ne "" -and "$($row.solarGeneration_actual)" -ne "") { [decimal]$row.solarForecast - [decimal]$row.solarGeneration_actual } else { $null }
    $lolpEvent = if ($null -ne $row.lossOfLoadProbability -and "$($row.lossOfLoadProbability)" -ne "") { if ([decimal]$row.lossOfLoadProbability -gt 0) { 1 } else { 0 } } else { $null }
    $lagSystemPrice = $previousSystemPrice
    if ($null -ne $row.systemPrice -and "$($row.systemPrice)" -ne "") { $previousSystemPrice = [decimal]$row.systemPrice } else { $previousSystemPrice = $null }

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
        if ($LASTEXITCODE -eq 0 -and (Test-Path $OutputParquet)) { $parquetWritten = $true }
    }
    catch { $parquetWritten = $false }
}

Write-Host "Built clean fundamentals panel: rows=$($engineered.Count) cols=$(@($engineered[0].PSObject.Properties.Name).Count)"
Write-Host "CSV saved to $OutputCsv"
if ($parquetWritten) { Write-Host "Parquet saved to $OutputParquet" } else { Write-Warning "Parquet export failed." }
