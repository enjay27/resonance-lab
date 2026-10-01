#!/bin/sh
# Runs once, when the data volume is empty (the postgres image runs /docker-entrypoint-initdb.d/*.sh at first start).
# Creates the `mlflow` role and the two databases deploy/mlflow/ uses. The password comes from the environment and is
# handed to psql as a variable, so it is quoted by psql and never pasted into the SQL by the shell.
set -e

psql -v ON_ERROR_STOP=1 -v mlflow_password="$MLFLOW_DB_PASSWORD" --username "$POSTGRES_USER" --dbname postgres <<'SQL'
CREATE ROLE mlflow LOGIN PASSWORD :'mlflow_password';
CREATE DATABASE mlflow OWNER mlflow;
CREATE DATABASE mlflow_auth OWNER mlflow;
REVOKE ALL ON DATABASE mlflow FROM PUBLIC;
REVOKE ALL ON DATABASE mlflow_auth FROM PUBLIC;
SQL
