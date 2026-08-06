param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "logs", "check", "train", "preview", "generate")]
    [string]$Command = "start",

    [Parameter(Position = 1)]
    [string]$Dataset = "",

    [switch]$Gpu,
    [int]$Steps = 100000,
    [int]$Episodes = 10,
    [int]$Seed = 0
)

$ErrorActionPreference = "Stop"
$compose = @("compose")
if ($Gpu) {
    $compose += @("-f", "compose.yaml", "-f", "compose.gpu.yaml")
}

switch ($Command) {
    "start" {
        docker @compose up --build -d
        Write-Host "SO-101 workshop is starting at http://localhost:8000"
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
