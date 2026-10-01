# PostgreSQL on the Synology NAS

The database behind the MLflow server (`deploy/mlflow/`): database `mlflow` for the runs, `mlflow_auth` for its login.
Not published on the LAN: only containers on the `resonance-db` Docker network reach it (MLflow joins that network).
Postgres is the backend MLflow itself is tested against most, and unlike SQLite it does not need a local volume.

## Set up on the NAS
1. **Folder.** In File Station create `docker/postgres` (`/volume1/docker/postgres`). Copy this folder's
   `Dockerfile`, `docker-compose.yml`, `.env.example` and the `initdb/` folder (with `10-mlflow.sh`) into it.
2. **Settings.** Copy `.env.example` to `.env` and fill in two passwords, letters and digits only
   (`openssl rand -hex 24` for each). `MLFLOW_DB_PASSWORD` must be the same value you put into `deploy/mlflow/.env`.
   `.env` stays on the NAS and is gitignored here.
3. **Start** (before the MLflow project). Container Manager -> Project -> Create: path `/volume1/docker/postgres`, use the
   existing `docker-compose.yml`, build and start. Or over SSH: `cd /volume1/docker/postgres && sudo docker compose up -d --build`.
   On the **first** start (empty data volume) `initdb/10-mlflow.sh` creates the `mlflow` role and both databases. Changing
   the password in `.env` later does not change the role: `sudo docker exec -it resonance-postgres psql -U postgres -c "ALTER ROLE mlflow PASSWORD '...'"`.
4. **Check.** `sudo docker exec resonance-postgres pg_isready -U postgres` says "accepting connections", and
   `sudo docker exec resonance-postgres psql -U postgres -c '\l'` lists `mlflow` and `mlflow_auth`.

## What the settings do
| Requirement | Setting |
|---|---|
| nobody on the LAN | no `ports:`; the container is only on the `resonance-db` network. To reach it with a client from the desktop, add `ports: ["<NAS IP>:5432:5432"]` yourself and a firewall rule for one IP |
| passwords | `scram-sha-256` for every network connection (never `trust`); the passwords are only in `.env` |
| unprivileged | the image's `postgres` user (70), read-only root file system, `/tmp` and `/run/postgresql` as tmpfs (the second owned by user 70), the init script copied into the image (not bind-mounted), no capabilities, no new privileges, 1 GB |
| the data | the named volume `resonance-postgres-data` (Docker keeps the image's ownership, so no `chown` and no root start); it lives under Container Manager's volume folder |

## If the first start went wrong
The init script runs only when the data volume is empty, and a start that skipped it leaves a data directory without the
`mlflow` role. Remove the volume and start again (it holds nothing yet): `sudo docker compose down -v && sudo docker compose up -d --build`.
Earlier versions of this folder bind-mounted `initdb/` and printed `ls: can't open '/docker-entrypoint-initdb.d/': Permission denied`
and `chmod: /var/run/postgresql: Operation not permitted`; the Dockerfile and the tmpfs ownership fix both.

## Backup and upgrade
- **Backup:** a dump is a consistent copy and works while the server runs. In DSM Task Scheduler (user-defined script, daily):
  `docker exec resonance-postgres pg_dump -U postgres -Fc mlflow > /volume1/backup/mlflow.dump` and the same for `mlflow_auth`;
  add that folder to Hyper Backup. Restore: `docker exec -i resonance-postgres pg_restore -U postgres -d mlflow --clean < mlflow.dump`.
- **Version:** `Dockerfile` starts `FROM postgres:18-alpine`, the newest 18.x (PostgreSQL has no LTS; each major is supported for 5 years). To pin an
  exact minor write `postgres:18.<N>-alpine` there (check the tag exists on Docker Hub first). A minor update is
  `docker compose build --pull && docker compose up -d`. To move to another major version (`postgres:19-alpine`) dump both databases, remove
  the volume, change the tag, start, restore: a major version cannot open the older one's data directory.
- The 18+ image keeps its data in `/var/lib/postgresql/18/docker`, so the volume is mounted on `/var/lib/postgresql`.

## What was and was not verified
Verified (a real PostgreSQL **16** server, a throwaway environment; 18 was not available there): `initdb/10-mlflow.sh` run as the image does it creates the role
and both databases, the `mlflow` role logs in with its password and not with a wrong one, and MLflow 3.16.1 runs its schema
migration in it (see `deploy/mlflow/README.md`).
**Not verified:** PostgreSQL 18 itself (the init script is plain `CREATE ROLE`/`CREATE DATABASE`/`REVOKE`; MLflow's migration was not run on 18) and the container on the NAS: the read-only root file system with `user: 70:70`, the image's entrypoint running the
init script (it passes `PGPASSWORD`, which the script relies on), Container Manager's project import, the DSM Task Scheduler backup command.
