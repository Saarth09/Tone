# Reset the local PostgreSQL 17 "postgres" superuser password.
# Run this in an elevated PowerShell (Right-click PowerShell -> Run as administrator):
#   cd c:\projects\Tone\backend
#   .\scripts\reset_postgres_password.ps1
#
# Default new password: tone-admin
# Override with: .\scripts\reset_postgres_password.ps1 -NewPassword "YourNewPassword"

param(
  [string]$NewPassword = "tone-admin",
  [string]$PostgresRoot = "C:\Program Files\PostgreSQL\17",
  [string]$ServiceName = "postgresql-x64-17"
)

$ErrorActionPreference = "Stop"
$psql = Join-Path $PostgresRoot "bin\psql.exe"
$hba = Join-Path $PostgresRoot "data\pg_hba.conf"
$backup = "$hba.bak-tone-reset"

if (-not (Test-Path $psql)) { throw "psql not found at $psql" }
if (-not (Test-Path $hba)) { throw "pg_hba.conf not found at $hba" }

Write-Host "Backing up pg_hba.conf..."
Copy-Item $hba $backup -Force

Write-Host "Switching local auth to trust..."
$lines = Get-Content $hba
$updated = foreach ($line in $lines) {
  if ($line -match '^\s*host\s+all\s+all\s+127\.0\.0\.1/32\s+') {
    "host    all             all             127.0.0.1/32            trust"
  } elseif ($line -match '^\s*host\s+all\s+all\s+::1/128\s+') {
    "host    all             all             ::1/128                 trust"
  } else {
    $line
  }
}
$updated | Set-Content -Path $hba -Encoding ascii

Write-Host "Restarting $ServiceName..."
Restart-Service $ServiceName -Force
Start-Sleep -Seconds 4

Write-Host "Setting new postgres password..."
& $psql -U postgres -h 127.0.0.1 -d postgres -c "ALTER USER postgres WITH PASSWORD '$NewPassword';"

Write-Host "Restoring pg_hba.conf from backup..."
Copy-Item $backup $hba -Force
Restart-Service $ServiceName -Force
Start-Sleep -Seconds 4

Write-Host ""
Write-Host "Password reset complete."
Write-Host "New postgres password: $NewPassword"
Write-Host ""
Write-Host "Next:"
Write-Host "  cd c:\projects\Tone\backend"
Write-Host "  `$env:PGPASSWORD = `"$NewPassword`""
Write-Host "  .\scripts\setup_postgres.ps1"
