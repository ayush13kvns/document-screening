$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
py -3.13 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Install Python 3.13 or follow README for Python 3.11/3.12" }
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -c constraints.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
& .\.venv\Scripts\python.exe -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
& .\.venv\Scripts\python.exe -m app.train --demo
if ($LASTEXITCODE -ne 0) { throw "Training failed" }
& .\.venv\Scripts\python.exe -m app.batch
if ($LASTEXITCODE -ne 0) { throw "Prediction failed" }
Write-Output "Read outputs/predictions.jsonl and artifacts/evaluation.json"
