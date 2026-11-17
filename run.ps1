# One-command local run on Windows: builds the React app and serves everything from FastAPI on :8000
$ErrorActionPreference = "Stop"
Push-Location "$PSScriptRoot\frontend"; npm install; npm run build; Pop-Location
Push-Location "$PSScriptRoot\backend"; .\.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000; Pop-Location
