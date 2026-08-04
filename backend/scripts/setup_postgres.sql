-- Create Tone application database + role (run as postgres superuser)
-- Example:
--   & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -f scripts/setup_postgres.sql

DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'tone') THEN
    CREATE ROLE tone LOGIN PASSWORD 'tone';
  END IF;
END
$$;

SELECT 'CREATE DATABASE tone OWNER tone'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'tone')\gexec

GRANT ALL PRIVILEGES ON DATABASE tone TO tone;
