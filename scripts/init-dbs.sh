#!/bin/bash
# Creates the langfuse database alongside the main aris database.
# Mounted into postgres docker-entrypoint-initdb.d/ — runs once on first boot.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE langfuse;
    GRANT ALL PRIVILEGES ON DATABASE langfuse TO "$POSTGRES_USER";
EOSQL
