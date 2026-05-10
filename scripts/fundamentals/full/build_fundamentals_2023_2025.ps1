param(
    [string[]] $Only = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$ProcessedDir = Join-Path $ProjectRoot "data/processed/full_2023_2025/fundamentals"
$DiagnosticsCoverageDir = Join-Path $ProjectRoot "data/diagnostics/coverage"
$DiagnosticsMissingnessDir = Join-Path $ProjectRoot "data/diagnostics/missingness"
$StartDate = [datetime]::ParseExact("2023-01-01", "yyyy-MM-dd", $null)
$EndDate = [datetime]::ParseExact("2025-12-31", "yyyy-MM-dd", $null)

New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null
New-Item -ItemType Directory -Force -Path $DiagnosticsCoverageDir | Out-Null
New-Item -ItemType Directory -Force -Path $DiagnosticsMissingnessDir | Out-Null

function Get-ExpectedKeys {
    param(
        [datetime] $WindowStart,
        [datetime] $WindowEnd
    )

    $expectedKeys = New-Object System.Collections.Generic.List[object]
    $day = $WindowStart
    while ($day -le $WindowEnd) {
        for ($sp = 1; $sp -le 48; $sp++) {
            $expectedKeys.Add([pscustomobject]@{
                settlementDate = $day.ToString("yyyy-MM-dd")
                settlementPeriod = $sp
            })
        }
        $day = $day.AddDays(1)
    }
    return $expectedKeys
}

function Get-MissingKeys {
    param(
        [object[]] $ProcessedRows,
        [datetime] $WindowStart,
        [datetime] $WindowEnd
    )

    $actualKeySet = @{}
    foreach ($row in $ProcessedRows) {
        $actualKeySet["$($row.settlementDate)|$($row.settlementPeriod)"] = $true
    }

    return @(
        (Get-ExpectedKeys -WindowStart $WindowStart -WindowEnd $WindowEnd) |
            Where-Object { -not $actualKeySet.ContainsKey("$($_.settlementDate)|$($_.settlementPeriod)") }
    )
}

function Save-DailyCompleteness {
    param(
        [object[]] $ProcessedRows,
        [string] $OutputPath
    )

    $ProcessedRows |
        Group-Object settlementDate |
        ForEach-Object {
            [pscustomobject]@{
                settlementDate = $_.Name
                rows = $_.Count
            }
        } |
        Sort-Object settlementDate |
        Export-Csv -NoTypeInformation -Path $OutputPath
}

function Build-ActualTotalLoad {
    $variable = "actual_total_load"
    $rawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$variable"
    $processedCsv = Join-Path $ProcessedDir "actual_total_load_2023_2025.csv"
    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

    $baseUrl = "https://data.elexon.co.uk/bmrs/api/v1/demand/actual/total"
    $allRows = New-Object System.Collections.Generic.List[object]
    $day = $StartDate
    while ($day -le $EndDate) {
        $dateText = $day.ToString("yyyy-MM-dd")
        $rawPath = Join-Path $rawDir "$dateText.csv"
        if ((Test-Path $rawPath) -and ((Get-Item $rawPath).Length -gt 0)) {
            $rows = @(Import-Csv $rawPath)
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count) (existing)"
        }
        else {
            $uri = "${baseUrl}?from=$dateText&to=$dateText&settlementPeriodFrom=1&settlementPeriodTo=48"
            $rows = @((Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }).data)
            $rows |
                Select-Object publishTime, startTime, settlementDate, settlementPeriod, quantity |
                Sort-Object settlementDate, settlementPeriod, publishTime |
                Export-Csv -NoTypeInformation -Path $rawPath
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count)"
        }

        foreach ($row in $rows) {
            $allRows.Add([pscustomobject]@{
                publishTime = $row.publishTime
                startTime = $row.startTime
                settlementDate = $row.settlementDate
                settlementPeriod = [int]$row.settlementPeriod
                quantity = [decimal]$row.quantity
            })
        }

        $day = $day.AddDays(1)
    }

    $processed = @(
        $allRows |
            Group-Object settlementDate, settlementPeriod |
            ForEach-Object {
                $_.Group |
                    Sort-Object @{ Expression = "publishTime"; Descending = $true } |
                    Select-Object -First 1
            } |
            Sort-Object settlementDate, settlementPeriod |
            Select-Object settlementDate, settlementPeriod, @{ Name = "ActualTotalLoad"; Expression = { $_.quantity } }, publishTime, startTime
    )

    $processed | Export-Csv -NoTypeInformation -Path $processedCsv
    $missing = Get-MissingKeys -ProcessedRows $processed -WindowStart $StartDate -WindowEnd $EndDate
    $missing | Export-Csv -NoTypeInformation -Path (Join-Path $DiagnosticsMissingnessDir "actual_total_load_2023_2025_missing_sps.csv")
    Save-DailyCompleteness -ProcessedRows $processed -OutputPath (Join-Path $DiagnosticsCoverageDir "actual_total_load_2023_2025_daily_completeness.csv")
    Write-Host "[PROCESSED][$variable] rows=$($processed.Count) missingSPs=$($missing.Count)"
}

function Build-InitialDemandOutturn {
    $variable = "initial_demand_outturn"
    $rawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$variable"
    $processedCsv = Join-Path $ProcessedDir "initial_demand_outturn_2023_2025.csv"
    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

    $baseUrl = "https://data.elexon.co.uk/bmrs/api/v1/datasets/INDO"
    $allRows = New-Object System.Collections.Generic.List[object]
    $day = $StartDate
    while ($day -le $EndDate) {
        $dateText = $day.ToString("yyyy-MM-dd")
        $rawPath = Join-Path $rawDir "$dateText.csv"
        $publishFrom = $day.AddHours(-1).ToString("yyyy-MM-ddTHH:mm:ss")
        $publishTo = $day.AddDays(1).AddHours(1).ToString("yyyy-MM-ddTHH:mm:ss")
        $uri = "${baseUrl}?publishDateTimeFrom=$publishFrom&publishDateTimeTo=$publishTo&format=json"
        $rows = @(
            (Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }).data |
                Where-Object { $_.settlementDate -eq $dateText }
        )
        $rows |
            Select-Object dataset, publishTime, startTime, settlementDate, settlementPeriod, demand |
            Sort-Object settlementDate, settlementPeriod, publishTime |
            Export-Csv -NoTypeInformation -Path $rawPath
        Write-Host "[RAW][$variable] $dateText rows=$($rows.Count)"

        foreach ($row in $rows) {
            $allRows.Add([pscustomobject]@{
                dataset = $row.dataset
                publishTime = $row.publishTime
                startTime = $row.startTime
                settlementDate = $row.settlementDate
                settlementPeriod = [int]$row.settlementPeriod
                demand = [decimal]$row.demand
            })
        }

        $day = $day.AddDays(1)
    }

    $processed = @(
        $allRows |
            Sort-Object settlementDate, settlementPeriod |
            Select-Object settlementDate, settlementPeriod, @{ Name = "InitialDemandOutturn"; Expression = { $_.demand } }, publishTime, startTime
    )

    $processed | Export-Csv -NoTypeInformation -Path $processedCsv
    $missing = Get-MissingKeys -ProcessedRows $processed -WindowStart $StartDate -WindowEnd $EndDate
    $missing | Export-Csv -NoTypeInformation -Path (Join-Path $DiagnosticsMissingnessDir "initial_demand_outturn_2023_2025_missing_sps.csv")
    Save-DailyCompleteness -ProcessedRows $processed -OutputPath (Join-Path $DiagnosticsCoverageDir "initial_demand_outturn_2023_2025_daily_completeness.csv")
    Write-Host "[PROCESSED][$variable] rows=$($processed.Count) missingSPs=$($missing.Count)"
}

function Build-DayAheadNationalDemandForecast {
    $variable = "day_ahead_demand_forecast_evolution"
    $rawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$variable"
    $processedCsv = Join-Path $ProcessedDir "day_ahead_national_demand_forecast_2023_2025.csv"
    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

    $baseUrl = "https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/day-ahead/evolution"
    $allRows = New-Object System.Collections.Generic.List[object]
    $day = $StartDate
    while ($day -le $EndDate) {
        $dateText = $day.ToString("yyyy-MM-dd")
        $rawPath = Join-Path $rawDir "$dateText.csv"

        if ((Test-Path $rawPath) -and ((Get-Item $rawPath).Length -gt 0)) {
            $dayRows = @(Import-Csv $rawPath)
            Write-Host "[RAW][$variable] $dateText rows=$($dayRows.Count) (existing)"
        }
        else {
            $dayRows = New-Object System.Collections.Generic.List[object]
            for ($sp = 1; $sp -le 48; $sp++) {
                $uri = "${baseUrl}?settlementDate=$dateText&settlementPeriod=$sp"
                $rows = @((Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }).data)
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
            }

            $dayRows |
                Sort-Object settlementDate, settlementPeriod, publishTime |
                Export-Csv -NoTypeInformation -Path $rawPath
            Write-Host "[RAW][$variable] $dateText rows=$($dayRows.Count)"
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

        $day = $day.AddDays(1)
    }

    $processed = @(
        $allRows |
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
    )

    $processed | Export-Csv -NoTypeInformation -Path $processedCsv
    $missing = Get-MissingKeys -ProcessedRows $processed -WindowStart $StartDate -WindowEnd $EndDate
    $missing | Export-Csv -NoTypeInformation -Path (Join-Path $DiagnosticsMissingnessDir "day_ahead_national_demand_forecast_2023_2025_missing_sps.csv")
    Save-DailyCompleteness -ProcessedRows $processed -OutputPath (Join-Path $DiagnosticsCoverageDir "day_ahead_national_demand_forecast_2023_2025_daily_completeness.csv")
    Write-Host "[PROCESSED][$variable] rows=$($processed.Count) missingSPs=$($missing.Count)"
}

function Build-ActualGenerationWindSolar {
    $variable = "actual_generation_wind_solar"
    $rawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$variable"
    $processedCsv = Join-Path $ProcessedDir "actual_generation_wind_solar_2023_2025.csv"
    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

    $baseUrl = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type/wind-and-solar"
    $allRows = New-Object System.Collections.Generic.List[object]
    $day = $StartDate
    while ($day -le $EndDate) {
        $dateText = $day.ToString("yyyy-MM-dd")
        $rawPath = Join-Path $rawDir "$dateText.csv"
        if ((Test-Path $rawPath) -and ((Get-Item $rawPath).Length -gt 0)) {
            $rows = @(Import-Csv $rawPath)
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count) (existing)"
        }
        else {
            $uri = "${baseUrl}?from=$dateText&to=$dateText&settlementPeriodFrom=1&settlementPeriodTo=48"
            $rows = @((Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }).data)
            $rows |
                Select-Object publishTime, businessType, psrType, quantity, startTime, settlementDate, settlementPeriod |
                Sort-Object settlementDate, settlementPeriod, psrType, publishTime |
                Export-Csv -NoTypeInformation -Path $rawPath
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count)"
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

    $latestByPsrType = @(
        $allRows |
            Group-Object settlementDate, settlementPeriod, psrType |
            ForEach-Object {
                $_.Group |
                    Sort-Object @{ Expression = { [datetime]$_.publishTime }; Descending = $true } |
                    Select-Object -First 1
            }
    )

    $processed = @(
        $latestByPsrType |
            Group-Object settlementDate, settlementPeriod |
            ForEach-Object {
                $groupRows = @($_.Group)
                $first = $groupRows | Select-Object -First 1
                $windOnshoreRow = @($groupRows | Where-Object { $_.psrType -eq "Wind Onshore" } | Select-Object -First 1)
                $windOffshoreRow = @($groupRows | Where-Object { $_.psrType -eq "Wind Offshore" } | Select-Object -First 1)
                $solarRow = @($groupRows | Where-Object { $_.psrType -eq "Solar" } | Select-Object -First 1)

                $windOnshore = if ($windOnshoreRow.Count -gt 0) { [decimal]$windOnshoreRow[0].quantity } else { $null }
                $windOffshore = if ($windOffshoreRow.Count -gt 0) { [decimal]$windOffshoreRow[0].quantity } else { $null }
                $solar = if ($solarRow.Count -gt 0) { [decimal]$solarRow[0].quantity } else { $null }
                $windTotal = $null
                if ($null -ne $windOnshore -or $null -ne $windOffshore) {
                    $windTotal = [decimal](($(if ($null -ne $windOnshore) { $windOnshore } else { 0 })) + ($(if ($null -ne $windOffshore) { $windOffshore } else { 0 })))
                }
                $windSolarTotal = $null
                if ($null -ne $windTotal -or $null -ne $solar) {
                    $windSolarTotal = [decimal](($(if ($null -ne $windTotal) { $windTotal } else { 0 })) + ($(if ($null -ne $solar) { $solar } else { 0 })))
                }

                [pscustomobject]@{
                    settlementDate = $first.settlementDate
                    settlementPeriod = [int]$first.settlementPeriod
                    windOnshore = $windOnshore
                    windOffshore = $windOffshore
                    solar = $solar
                    windTotal = $windTotal
                    windSolarTotal = $windSolarTotal
                    windOnshore_publishTime = if ($windOnshoreRow.Count -gt 0) { $windOnshoreRow[0].publishTime } else { $null }
                    windOffshore_publishTime = if ($windOffshoreRow.Count -gt 0) { $windOffshoreRow[0].publishTime } else { $null }
                    solar_publishTime = if ($solarRow.Count -gt 0) { $solarRow[0].publishTime } else { $null }
                    startTime = $first.startTime
                }
            } |
            Sort-Object settlementDate, settlementPeriod
    )

    $processed | Export-Csv -NoTypeInformation -Path $processedCsv
    $missing = Get-MissingKeys -ProcessedRows $processed -WindowStart $StartDate -WindowEnd $EndDate
    $missing | Export-Csv -NoTypeInformation -Path (Join-Path $DiagnosticsMissingnessDir "actual_generation_wind_solar_2023_2025_missing_sps.csv")
    Save-DailyCompleteness -ProcessedRows $processed -OutputPath (Join-Path $DiagnosticsCoverageDir "actual_generation_wind_solar_2023_2025_daily_completeness.csv")
    Write-Host "[PROCESSED][$variable] rows=$($processed.Count) missingSPs=$($missing.Count)"
}

function Build-DayAheadGenerationWindSolar {
    $variable = "day_ahead_generation_wind_solar"
    $rawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$variable"
    $processedCsv = Join-Path $ProcessedDir "day_ahead_generation_wind_solar_2023_2025.csv"
    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

    $baseUrl = "https://data.elexon.co.uk/bmrs/api/v1/forecast/generation/wind-and-solar/day-ahead"
    $allRows = New-Object System.Collections.Generic.List[object]
    $rawProcessTypeCounts = New-Object System.Collections.Generic.List[object]
    $day = $StartDate
    while ($day -le $EndDate) {
        $dateText = $day.ToString("yyyy-MM-dd")
        $rawPath = Join-Path $rawDir "$dateText.csv"
        if ((Test-Path $rawPath) -and ((Get-Item $rawPath).Length -gt 0)) {
            $rows = @(Import-Csv $rawPath)
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count) (existing)"
        }
        else {
            $uri = "${baseUrl}?from=$dateText&to=$dateText&settlementPeriodFrom=1&settlementPeriodTo=48&processType=all"
            $rows = @((Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }).data)
            $rows |
                Select-Object publishTime, processType, businessType, psrType, startTime, settlementDate, settlementPeriod, quantity |
                Sort-Object settlementDate, settlementPeriod, processType, psrType, publishTime |
                Export-Csv -NoTypeInformation -Path $rawPath
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count)"
        }

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

        @($rows | Select-Object -ExpandProperty processType -Unique) | ForEach-Object {
            $rawProcessTypeCounts.Add([pscustomobject]@{
                settlementDate = $dateText
                processType = $_
                rows = @($rows | Where-Object { $_.processType -eq $PSItem }).Count
            })
        }

        $day = $day.AddDays(1)
    }

    $dayAheadRows = @($allRows | Where-Object { $_.processType -eq "Day ahead" })
    $latestByPsrType = @(
        $dayAheadRows |
            Group-Object settlementDate, settlementPeriod, psrType |
            ForEach-Object {
                $_.Group |
                    Sort-Object @{ Expression = { [datetime]$_.publishTime }; Descending = $true } |
                    Select-Object -First 1
            }
    )

    $processed = @(
        $latestByPsrType |
            Group-Object settlementDate, settlementPeriod |
            ForEach-Object {
                $groupRows = @($_.Group)
                $first = $groupRows | Select-Object -First 1
                $windOnshoreRow = @($groupRows | Where-Object { $_.psrType -eq "Wind Onshore" } | Select-Object -First 1)
                $windOffshoreRow = @($groupRows | Where-Object { $_.psrType -eq "Wind Offshore" } | Select-Object -First 1)
                $solarRow = @($groupRows | Where-Object { $_.psrType -eq "Solar" } | Select-Object -First 1)

                $windOnshore = if ($windOnshoreRow.Count -gt 0) { [decimal]$windOnshoreRow[0].quantity } else { $null }
                $windOffshore = if ($windOffshoreRow.Count -gt 0) { [decimal]$windOffshoreRow[0].quantity } else { $null }
                $solar = if ($solarRow.Count -gt 0) { [decimal]$solarRow[0].quantity } else { $null }
                $windTotal = $null
                if ($null -ne $windOnshore -or $null -ne $windOffshore) {
                    $windTotal = [decimal](($(if ($null -ne $windOnshore) { $windOnshore } else { 0 })) + ($(if ($null -ne $windOffshore) { $windOffshore } else { 0 })))
                }
                $windSolarTotal = $null
                if ($null -ne $windTotal -or $null -ne $solar) {
                    $windSolarTotal = [decimal](($(if ($null -ne $windTotal) { $windTotal } else { 0 })) + ($(if ($null -ne $solar) { $solar } else { 0 })))
                }

                [pscustomobject]@{
                    settlementDate = $first.settlementDate
                    settlementPeriod = [int]$first.settlementPeriod
                    dayAheadWindOnshoreForecast = $windOnshore
                    dayAheadWindOffshoreForecast = $windOffshore
                    dayAheadSolarForecast = $solar
                    dayAheadWindTotalForecast = $windTotal
                    dayAheadWindSolarTotalForecast = $windSolarTotal
                    dayAheadWindOnshoreForecast_publishTime = if ($windOnshoreRow.Count -gt 0) { $windOnshoreRow[0].publishTime } else { $null }
                    dayAheadWindOffshoreForecast_publishTime = if ($windOffshoreRow.Count -gt 0) { $windOffshoreRow[0].publishTime } else { $null }
                    dayAheadSolarForecast_publishTime = if ($solarRow.Count -gt 0) { $solarRow[0].publishTime } else { $null }
                    startTime = $first.startTime
                }
            } |
            Sort-Object settlementDate, settlementPeriod
    )

    $processed | Export-Csv -NoTypeInformation -Path $processedCsv
    $rawProcessTypeCounts | Sort-Object settlementDate, processType | Export-Csv -NoTypeInformation -Path (Join-Path $DiagnosticsCoverageDir "day_ahead_generation_wind_solar_2023_2025_raw_process_type_counts.csv")
    $missing = Get-MissingKeys -ProcessedRows $processed -WindowStart $StartDate -WindowEnd $EndDate
    $missing | Export-Csv -NoTypeInformation -Path (Join-Path $DiagnosticsMissingnessDir "day_ahead_generation_wind_solar_2023_2025_missing_sps.csv")
    Save-DailyCompleteness -ProcessedRows $processed -OutputPath (Join-Path $DiagnosticsCoverageDir "day_ahead_generation_wind_solar_2023_2025_daily_completeness.csv")
    Write-Host "[PROCESSED][$variable] rows=$($processed.Count) missingSPs=$($missing.Count)"
}

function Build-InterconnectorFlows {
    $variable = "interconnector_flows"
    $rawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$variable"
    $processedCsv = Join-Path $ProcessedDir "interconnector_flows_2023_2025.csv"
    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

    $baseUrl = "https://data.elexon.co.uk/bmrs/api/v1/generation/outturn/interconnectors"
    $allRows = New-Object System.Collections.Generic.List[object]
    $day = $StartDate
    while ($day -le $EndDate) {
        $dateText = $day.ToString("yyyy-MM-dd")
        $rawPath = Join-Path $rawDir "$dateText.csv"
        if ((Test-Path $rawPath) -and ((Get-Item $rawPath).Length -gt 0)) {
            $rows = @(Import-Csv $rawPath)
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count) (existing)"
        }
        else {
            $uri = "${baseUrl}?settlementDateFrom=$dateText&settlementDateTo=$dateText"
            $rows = @((Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }).data)
            $rows |
                Sort-Object settlementDate, settlementPeriod, interconnectorName, publishTime |
                Export-Csv -NoTypeInformation -Path $rawPath
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count)"
        }

        foreach ($row in $rows) {
            $allRows.Add([pscustomobject]@{
                publishTime = $row.publishTime
                startTime = $row.startTime
                settlementDate = $row.settlementDate
                settlementPeriod = [int]$row.settlementPeriod
                interconnectorName = $row.interconnectorName
                generation = [decimal]$row.generation
            })
        }

        $day = $day.AddDays(1)
    }

    $processed = @(
        $allRows |
            Group-Object settlementDate, settlementPeriod, interconnectorName |
            ForEach-Object {
                $_.Group |
                    Sort-Object @{ Expression = { [datetime]$_.publishTime }; Descending = $true } |
                    Select-Object -First 1
            } |
            Group-Object settlementDate, settlementPeriod |
            ForEach-Object {
                $rows = @($_.Group)
                $first = $rows | Select-Object -First 1
                [pscustomobject]@{
                    settlementDate = $first.settlementDate
                    settlementPeriod = [int]$first.settlementPeriod
                    netInterconnectorFlow = [decimal](($rows | Measure-Object -Property generation -Sum).Sum)
                    interconnectorCount = $rows.Count
                }
            } |
            Sort-Object settlementDate, settlementPeriod
    )

    $processed | Export-Csv -NoTypeInformation -Path $processedCsv
    $missing = Get-MissingKeys -ProcessedRows $processed -WindowStart $StartDate -WindowEnd $EndDate
    $missing | Export-Csv -NoTypeInformation -Path (Join-Path $DiagnosticsMissingnessDir "interconnector_flows_2023_2025_missing_sps.csv")
    Save-DailyCompleteness -ProcessedRows $processed -OutputPath (Join-Path $DiagnosticsCoverageDir "interconnector_flows_2023_2025_daily_completeness.csv")
    Write-Host "[PROCESSED][$variable] rows=$($processed.Count) missingSPs=$($missing.Count)"
}

function Build-LolpDrm {
    $variable = "lolpdrm"
    $rawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$variable"
    $processedCsv = Join-Path $ProcessedDir "lolpdrm_2023_2025.csv"
    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

    $baseUrl = "https://data.elexon.co.uk/bmrs/api/v1/forecast/system/loss-of-load"
    $allRows = New-Object System.Collections.Generic.List[object]
    $day = $StartDate
    while ($day -le $EndDate) {
        $dateText = $day.ToString("yyyy-MM-dd")
        $rawPath = Join-Path $rawDir "$dateText.csv"
        if ((Test-Path $rawPath) -and ((Get-Item $rawPath).Length -gt 0)) {
            $rows = @(Import-Csv $rawPath)
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count) (existing)"
        }
        else {
            $uri = "${baseUrl}?from=$dateText&to=$dateText&settlementPeriodFrom=1&settlementPeriodTo=48"
            $rows = @((Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }).data)
            $rows |
                Select-Object publishTime, publishingPeriodCommencingTime, startTime, settlementDate, settlementPeriod, forecastHorizon, lossOfLoadProbability, deratedMargin |
                Sort-Object settlementDate, settlementPeriod, forecastHorizon, publishTime |
                Export-Csv -NoTypeInformation -Path $rawPath
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count)"
        }

        foreach ($row in $rows) {
            $allRows.Add([pscustomobject]@{
                publishTime = $row.publishTime
                publishingPeriodCommencingTime = $row.publishingPeriodCommencingTime
                startTime = $row.startTime
                settlementDate = $row.settlementDate
                settlementPeriod = [int]$row.settlementPeriod
                forecastHorizon = [int]$row.forecastHorizon
                lossOfLoadProbability = [decimal]$row.lossOfLoadProbability
                deratedMargin = [decimal]$row.deratedMargin
            })
        }

        $day = $day.AddDays(1)
    }

    $processed = @(
        $allRows |
            Where-Object { $_.forecastHorizon -eq 2 } |
            Group-Object settlementDate, settlementPeriod |
            ForEach-Object {
                $_.Group |
                    Sort-Object @{ Expression = { [datetime]$_.publishTime }; Descending = $true } |
                    Select-Object -First 1
            } |
            Sort-Object settlementDate, settlementPeriod |
            Select-Object settlementDate, settlementPeriod, lossOfLoadProbability, deratedMargin, forecastHorizon, publishTime, publishingPeriodCommencingTime, startTime
    )

    $processed | Export-Csv -NoTypeInformation -Path $processedCsv
    $missing = Get-MissingKeys -ProcessedRows $processed -WindowStart $StartDate -WindowEnd $EndDate
    $missing | Export-Csv -NoTypeInformation -Path (Join-Path $DiagnosticsMissingnessDir "lolpdrm_2023_2025_missing_sps.csv")
    Save-DailyCompleteness -ProcessedRows $processed -OutputPath (Join-Path $DiagnosticsCoverageDir "lolpdrm_2023_2025_daily_completeness.csv")
    Write-Host "[PROCESSED][$variable] rows=$($processed.Count) missingSPs=$($missing.Count)"
}

function Build-ApxMid {
    $variable = "apx_mid"
    $rawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$variable"
    $processedCsv = Join-Path $ProcessedDir "apx_mid_2023_2025.csv"
    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

    $baseUrl = "https://data.elexon.co.uk/bmrs/api/v1/datasets/MID"
    $allRows = New-Object System.Collections.Generic.List[object]
    $day = $StartDate
    while ($day -le $EndDate) {
        $dateText = $day.ToString("yyyy-MM-dd")
        $rawPath = Join-Path $rawDir "$dateText.csv"
        if ((Test-Path $rawPath) -and ((Get-Item $rawPath).Length -gt 0)) {
            $rows = @(Import-Csv $rawPath)
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count) (existing)"
        }
        else {
            $uri = "${baseUrl}?from=$dateText&to=$dateText&settlementPeriodFrom=1&settlementPeriodTo=48&dataProviders=APXMIDP"
            $rows = @((Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }).data)
            $rows |
                Sort-Object settlementDate, settlementPeriod |
                Export-Csv -NoTypeInformation -Path $rawPath
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count)"
        }

        foreach ($row in $rows) {
            $allRows.Add([pscustomobject]@{
                settlementDate = $row.settlementDate
                settlementPeriod = [int]$row.settlementPeriod
                marketIndexPrice = [decimal]$row.price
                marketIndexVolume = [decimal]$row.volume
                dataProvider = $row.dataProvider
            })
        }

        $day = $day.AddDays(1)
    }

    $processed = @(
        $allRows |
            Sort-Object settlementDate, settlementPeriod |
            Select-Object settlementDate, settlementPeriod, marketIndexPrice, marketIndexVolume, dataProvider
    )

    $processed | Export-Csv -NoTypeInformation -Path $processedCsv
    $missing = Get-MissingKeys -ProcessedRows $processed -WindowStart $StartDate -WindowEnd $EndDate
    $missing | Export-Csv -NoTypeInformation -Path (Join-Path $DiagnosticsMissingnessDir "apx_mid_2023_2025_missing_sps.csv")
    Save-DailyCompleteness -ProcessedRows $processed -OutputPath (Join-Path $DiagnosticsCoverageDir "apx_mid_2023_2025_daily_completeness.csv")
    Write-Host "[PROCESSED][$variable] rows=$($processed.Count) missingSPs=$($missing.Count)"
}

function Build-SystemPriceNiv {
    $variable = "system_price_niv"
    $rawDir = Join-Path $ProjectRoot "data/raw/fundamentals/$variable"
    $processedCsv = Join-Path $ProcessedDir "system_price_niv_2023_2025.csv"
    New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

    $baseUrl = "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices"
    $allRows = New-Object System.Collections.Generic.List[object]
    $day = $StartDate
    while ($day -le $EndDate) {
        $dateText = $day.ToString("yyyy-MM-dd")
        $rawPath = Join-Path $rawDir "$dateText.csv"
        if ((Test-Path $rawPath) -and ((Get-Item $rawPath).Length -gt 0)) {
            $rows = @(Import-Csv $rawPath)
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count) (existing)"
        }
        else {
            $uri = "$baseUrl/$dateText"
            $rows = @((Invoke-RestMethod -Uri $uri -Method Get -Headers @{ Accept = "application/json" }).data)
            $rows | Sort-Object settlementDate, settlementPeriod | Export-Csv -NoTypeInformation -Path $rawPath
            Write-Host "[RAW][$variable] $dateText rows=$($rows.Count)"
        }

        foreach ($row in $rows) {
            $allRows.Add([pscustomobject]@{
                settlementDate = $row.settlementDate
                settlementPeriod = [int]$row.settlementPeriod
                systemSellPrice = [decimal]$row.systemSellPrice
                systemBuyPrice = [decimal]$row.systemBuyPrice
                netImbalanceVolume = [decimal]$row.netImbalanceVolume
            })
        }

        $day = $day.AddDays(1)
    }

    $processed = @(
        $allRows |
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
                    systemLongShort = if ($_.netImbalanceVolume -gt 0) { "short" } elseif ($_.netImbalanceVolume -lt 0) { "long" } else { "balanced" }
                }
            }
    )

    $processed | Export-Csv -NoTypeInformation -Path $processedCsv
    $missing = Get-MissingKeys -ProcessedRows $processed -WindowStart $StartDate -WindowEnd $EndDate
    $missing | Export-Csv -NoTypeInformation -Path (Join-Path $DiagnosticsMissingnessDir "system_price_niv_2023_2025_missing_sps.csv")
    Save-DailyCompleteness -ProcessedRows $processed -OutputPath (Join-Path $DiagnosticsCoverageDir "system_price_niv_2023_2025_daily_completeness.csv")
    Write-Host "[PROCESSED][$variable] rows=$($processed.Count) missingSPs=$($missing.Count)"
}

function Should-Run {
    param([string] $Name)
    if ($Only.Count -eq 0) { return $true }
    return $Only -contains $Name
}

Write-Host "Building system fundamentals for 2023-01-01 to 2025-12-31"
if (Should-Run "system_price_niv") { Build-SystemPriceNiv }
if (Should-Run "actual_total_load") { Build-ActualTotalLoad }
if (Should-Run "initial_demand_outturn") { Build-InitialDemandOutturn }
if (Should-Run "day_ahead_national_demand_forecast") { Build-DayAheadNationalDemandForecast }
if (Should-Run "actual_generation_wind_solar") { Build-ActualGenerationWindSolar }
if (Should-Run "day_ahead_generation_wind_solar") { Build-DayAheadGenerationWindSolar }
if (Should-Run "interconnector_flows") { Build-InterconnectorFlows }
if (Should-Run "lolpdrm") { Build-LolpDrm }
if (Should-Run "apx_mid") { Build-ApxMid }
Write-Host "Full fundamentals build complete."
