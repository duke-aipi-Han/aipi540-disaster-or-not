param(
    [string]$SpaceRepo = "hw391/AIPI540-disaster-or-not",
    [string]$ModelRepo = "hw391/disaster-or-not-tweet-model",
    [string]$WorkDir = "$env:TEMP\AIPI540-disaster-or-not-space",
    [string]$CommitMessage = "Deploy Streamlit app to Hugging Face Space"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$spaceUrl = "https://huggingface.co/spaces/$SpaceRepo"

function Copy-RequiredFile {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Required deploy file is missing: $Source"
    }

    $destinationDir = Split-Path -Parent $Destination
    if ($destinationDir -and -not (Test-Path -LiteralPath $destinationDir)) {
        New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
    }

    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required to deploy the Space."
}

if (-not (Test-Path -LiteralPath $WorkDir)) {
    git clone $spaceUrl $WorkDir
} else {
    git -C $WorkDir fetch origin
    git -C $WorkDir checkout main
    git -C $WorkDir pull --ff-only origin main
}

$spaceSrc = Join-Path $WorkDir "src"
$spaceModels = Join-Path $WorkDir "models"
New-Item -ItemType Directory -Path $spaceSrc -Force | Out-Null
New-Item -ItemType Directory -Path $spaceModels -Force | Out-Null

Copy-RequiredFile (Join-Path $repoRoot "Dockerfile") (Join-Path $WorkDir "Dockerfile")
Copy-RequiredFile (Join-Path $repoRoot "app.py") (Join-Path $WorkDir "app.py")
Copy-RequiredFile (Join-Path $repoRoot "requirements.txt") (Join-Path $WorkDir "requirements.txt")
Copy-RequiredFile (Join-Path $repoRoot "src\__init__.py") (Join-Path $spaceSrc "__init__.py")
Copy-RequiredFile (Join-Path $repoRoot "src\disasters.py") (Join-Path $spaceSrc "disasters.py")

Copy-RequiredFile (Join-Path $repoRoot "models\tfidf_logreg.joblib") (Join-Path $spaceModels "tfidf_logreg.joblib")
Copy-RequiredFile (Join-Path $repoRoot "models\baseline_metrics.json") (Join-Path $spaceModels "baseline_metrics.json")
Copy-RequiredFile (Join-Path $repoRoot "models\transformer_metrics_cardiffnlp-twitter-roberta-base.json") `
    (Join-Path $spaceModels "transformer_metrics_cardiffnlp-twitter-roberta-base.json")
Copy-RequiredFile (Join-Path $repoRoot "models\transformer_metrics_distilbert-base-uncased.json") `
    (Join-Path $spaceModels "transformer_metrics_distilbert-base-uncased.json")

$templateApp = Join-Path $spaceSrc "streamlit_app.py"
if (Test-Path -LiteralPath $templateApp) {
    Remove-Item -LiteralPath $templateApp -Force
}

$env:HF_MODEL_REPO = $ModelRepo
git -C $WorkDir add Dockerfile app.py requirements.txt src models

git -C $WorkDir diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "No Space changes to deploy."
    exit 0
}

git -C $WorkDir commit -m $CommitMessage
git -C $WorkDir push

Write-Host "Deployed $SpaceRepo from $repoRoot"
