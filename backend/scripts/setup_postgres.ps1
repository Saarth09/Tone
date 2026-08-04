# Creates the Tone Postgres role + database, then application tables.
# Usage (PowerShell):
#   $env:PGPASSWORD = "YOUR_ACTUAL_POSTGRES_PASSWORD"
#   .\scripts\setup_postgres.ps1

param(
  [string]$PostgresBin = "C:\Program Files\PostgreSQL\17\bin",
  [string]$SuperUser = "postgres",
  [string]$HostAddress = "127.0.0.1",
  [int]$Port = 5432
)

$ErrorActionPreference = "Stop"
$psql = Join-Path $PostgresBin "psql.exe"
if (-not (Test-Path $psql)) {
  throw "psql not found at $psql - update -PostgresBin"
}
if (-not $env:PGPASSWORD) {
  throw "Set `$env:PGPASSWORD to your postgres superuser password first."
}

Write-Host "Creating role/database..."
& $psql -U $SuperUser -h $HostAddress -p $Port -d postgres -v ON_ERROR_STOP=1 -f (Join-Path $PSScriptRoot "setup_postgres.sql")

Write-Host "Installing Python driver + creating tables..."
Set-Location (Join-Path $PSScriptRoot "..")
.\.venv\Scripts\python.exe -m pip install asyncpg==0.30.0 -q
.\.venv\Scripts\python.exe -m scripts.init_db

Write-Host "Done. DATABASE_URL should be:"
Write-Host "postgresql+asyncpg://tone:tone@127.0.0.1:5432/tone"
