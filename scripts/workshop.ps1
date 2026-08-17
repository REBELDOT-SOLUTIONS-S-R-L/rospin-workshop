param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "logs", "check", "train", "deploy", "preview", "generate")]
    [string]$Command = "start",

    [Parameter(Position = 1)]
    [string]$Dataset = "",

    [switch]$Gpu,
    [int]$Steps = 100000,
    [int]$Episodes = 10,
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$hostDataRoot = Join-Path $repoRoot "data"
New-Item -ItemType Directory -Force -Path $hostDataRoot | Out-Null
$env:ROSPIN_HOST_DATA_DIR = $hostDataRoot

$compose = @(
    "compose",
    "--project-directory", $repoRoot,
    "-f", (Join-Path $repoRoot "compose.yaml")
)
if ($Gpu) {
    $compose += @("-f", (Join-Path $repoRoot "compose.gpu.yaml"))
}

function Assert-DataMount {
    $probeName = ".rospin-mount-probe-$([Guid]::NewGuid().ToString('N'))"
    $probeValue = "rospin-mount-ok-$([Guid]::NewGuid().ToString('N'))"
    $probePath = Join-Path $hostDataRoot $probeName

    docker @compose exec -T workshop sh -c "printf '%s' '$probeValue' > '/workspace/data/$probeName'"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not write the data-mount verification file inside the container."
    }
    $deadline = [DateTime]::UtcNow.AddSeconds(5)
    $actualValue = $null
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-Path -LiteralPath $probePath) {
            $actualValue = [System.IO.File]::ReadAllText($probePath)
            if ($actualValue -eq $probeValue) { break }
        }
        Start-Sleep -Milliseconds 100
    }
    if (-not (Test-Path -LiteralPath $probePath)) {
        docker @compose exec -T workshop rm -f "/workspace/data/$probeName" | Out-Null
        throw "Docker cannot write through to the host data directory: $hostDataRoot. Refusing to run because datasets would remain inside Docker."
    }
    if ($actualValue -ne $probeValue) {
        Remove-Item -LiteralPath $probePath -Force
        throw "Docker returned stale data for the host directory: $hostDataRoot."
    }
    Remove-Item -LiteralPath $probePath -Force
}

switch ($Command) {
    "start" {
        docker @compose up --build -d --force-recreate --wait workshop
        if ($LASTEXITCODE -ne 0) { throw "Docker Compose failed to start the workshop." }
        Assert-DataMount
        Write-Host "SO-101 workshop is running at http://localhost:8000"
        docker @compose logs -f workshop
    }
    "stop" {
        docker @compose down
    }
    "logs" {
        docker @compose logs -f workshop
    }
    "check" {
        if (-not $Dataset) { throw "Pass a dataset directory name from data/datasets." }
        docker @compose run --rm --entrypoint rospin-check-dataset workshop "/workspace/data/datasets/$Dataset"
    }
    "train" {
        if (-not $Dataset) { throw "Pass a dataset directory name from data/datasets." }
        $device = if ($Gpu) { "cuda" } else { "cpu" }
        docker @compose run --rm --entrypoint rospin-train-act workshop `
            "/workspace/data/datasets/$Dataset" --device $device --steps $Steps
    }
    "deploy" {
        if (-not $Dataset) { throw "Pass a checkpoint path relative to data/outputs." }
        $device = if ($Gpu) { "cuda" } else { "cpu" }
        docker @compose up -d --build --wait workshop
        if ($LASTEXITCODE -ne 0) { throw "Docker Compose failed to start the workshop." }
        Assert-DataMount
        docker @compose exec -T workshop rospin-deploy $Dataset `
            --device $device --episodes $Episodes --seed $Seed
    }
    "preview" {
        if (-not $Dataset) { throw "Pass a Python filename from trajectories/." }
        docker @compose exec -T workshop rospin-generate $Dataset --preview --seed $Seed
    }
    "generate" {
        if (-not $Dataset) { throw "Pass a Python filename from trajectories/." }
        docker @compose exec -T workshop rospin-generate $Dataset `
            --episodes $Episodes --seed $Seed
    }
}
