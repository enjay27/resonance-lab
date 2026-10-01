# MLflow tracking server on the Synology NAS

Records every training run of resonance-lab (parameters, metrics, tags, small reports) so runs can be compared without
pasting reports into chat. **One user (you), LAN only, runs no code, keeps metadata and small files only.** The plan and
the reasons are in `.memory/roadmap/mlflow-plan.md`.

Target: Synology DSM 7.2 with Container Manager, x86_64 (Intel Celeron J4025), 10 GB RAM. The server container is
limited to 1 GB and one worker.

## Set up on the NAS
Needs the Postgres of `deploy/postgres/` (running, with the `mlflow` and `mlflow_auth` databases) and a bucket in MinIO.
1. **MinIO.** In the MinIO console create a bucket (`mlflow`) and an access key for MLflow. Do not use the root user: give the
   key a policy that allows only that bucket (`s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, `s3:ListBucket` on
   `arn:aws:s3:::mlflow` and `arn:aws:s3:::mlflow/*`).
2. **Folder.** In File Station create `docker/mlflow` (`/volume1/docker/mlflow`). Copy this folder's `Dockerfile`,
   `docker-compose.yml`, `basic_auth.ini`, `entrypoint.py` and `.env.example` into it (`basic_auth.ini` and `entrypoint.py` are
   copied into the image at build time; nothing is mounted from the NAS, so file permissions there do not matter).
3. **Settings.** Copy `.env.example` to `.env` and fill it in:
   - `MLFLOW_BIND`: an IP address **the NAS really has** (DSM Control Panel -> Network, or `ip -4 addr` over SSH; give it a fixed
     address or a DHCP reservation). A wrong one fails with `cannot assign requested address`. `MLFLOW_ALLOWED_HOSTS`: that
     address and port, plus any host name you will use (`192.168.0.10:5050,nas.local:5050`).
   - `MLFLOW_DB_PASSWORD`: the same value as in `deploy/postgres/.env`.
   - `MINIO_ENDPOINT_URL` (the NAS address and MinIO's S3 port, not `localhost`), `MINIO_BUCKET`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`.
   - `MLFLOW_AUTH_ADMIN_PASSWORD`: a long password. `MLFLOW_FLASK_SERVER_SECRET_KEY`: `openssl rand -hex 32`.
   - `.env` stays on the NAS and is gitignored here.
4. **Start.** Container Manager -> Project -> Create: path `/volume1/docker/mlflow`, use the existing `docker-compose.yml`,
   build and start. (Or over SSH: `cd /volume1/docker/mlflow && sudo docker compose up -d --build`.)
   The first build downloads MLflow from PyPI. If you built an older version of this folder before (SQLite, no Postgres driver),
   compose reuses that image because its tag exists: remove it first (`sudo docker compose down && sudo docker rmi resonance-mlflow:<MLFLOW_VERSION>`),
   or rebuild with `sudo docker compose build --no-cache`. `ModuleNotFoundError: No module named 'psycopg2'` means an old image is running.
   On the first start MLflow creates its tables in Postgres and the `admin`
   user with your password. The `PIDs limit` warning on an old DSM kernel is harmless.
5. **Firewall.** Control Panel -> Security -> Firewall: allow TCP 5050 from the desktop's IP only, deny the rest. Do
   not forward the port on the router.

## Check it from the desktop
```
curl http://<NAS IP>:5050/health                                              # OK, the only open URL
curl -s -o /dev/null -w "%{http_code}\n" http://<NAS IP>:5050/api/2.0/mlflow/experiments/search?max_results=1   # 401
curl -s -o /dev/null -w "%{http_code}\n" -u admin:<password> "http://<NAS IP>:5050/api/2.0/mlflow/experiments/search?max_results=1"   # 200
```
Then, in the resonance-lab repo, copy `.env.mlflow.example` to `.env.mlflow` and fill in the URL and password
(gitignored: the URL and the password never go into git). Open `http://<NAS IP>:5050` in a browser to see the runs.

## What the settings do
| Requirement | Setting |
|---|---|
| only you | basic-auth with one `admin` user, `default_permission = NO_PERMISSIONS` (MLflow's own default is READ), no self sign-up (every API call and `/signup` answer 401), `--allowed-hosts` (a request naming another host gets 403), no CORS origin, bound to the NAS LAN address, firewall to one IP |
| runs no Python | `mlflow server` only (no Projects, model serving or registry in use). MLflow 3.x starts job workers **by default**: `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` and `MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS=false` switch them off; the assistant and its docker sandbox are off; no `docker.sock`; read-only root file system, no capabilities, non-root, 1 GB, 200 processes |
| metadata + small files | runs and users in Postgres (`deploy/postgres/`, databases `mlflow` and `mlflow_auth`), artifacts in MinIO through `--serve-artifacts` (the desktop never talks to Postgres or MinIO, so it needs none of their credentials). resonance-lab logs only reports, predictions and configs, never data, weights or GGUFs |
| private | telemetry is off (`MLFLOW_DISABLE_TELEMETRY`, `DO_NOT_TRACK`); the client example sets it too |

The container writes nothing itself (read-only root, `/tmp` as tmpfs): its data is in Postgres and MinIO. The basic-auth
config in git is a template; `entrypoint.py` writes the real one (with the Postgres URI) to `/tmp` at start, mode 600.

## Backup and upgrade
- **Backup:** the Postgres dumps (`deploy/postgres/README.md`) and the MinIO bucket (Hyper Backup, or `mc mirror`).
- **Upgrade:** back up, set the new `MLFLOW_VERSION` in `.env` **and** in resonance-lab's requirements (client and server
  must be the same version), rebuild, then migrate the store once:
  `sudo docker compose run --rm mlflow db upgrade "postgresql+psycopg2://mlflow:<MLFLOW_DB_PASSWORD>@postgres:5432/mlflow"`.

## What was and was not verified
Verified (MLflow 3.16.1, a real PostgreSQL 16 server (the compose file uses 18, not available there) with scram passwords, a moto S3 server standing in for MinIO, a throwaway
environment): `deploy/postgres/initdb/10-mlflow.sh` creates the role and databases; `entrypoint.py` renders the auth config (mode 600)
and starts `mlflow server` with the flags of `docker-compose.yml`; MLflow migrates its schema in Postgres and creates the `admin`
user in `mlflow_auth`; 401 without credentials, 403 for a foreign `Host`, 200 for the admin; a client logs params, tags, a metric and
an artifact, and the artifact lands in the bucket under `artifacts/<experiment>/<run>/artifacts/`. Earlier (SQLite version):
`/signup` and user creation answer 401, 1000 metrics in one batch, a backdated start time.
**Not verified:** the image build and the containers on the NAS (no Docker daemon in the session that wrote this; `docker compose config`
was not run for this version): the read-only root file system with `/tmp` as tmpfs, the join of the `resonance-db` network, a real
MinIO (path-style addressing, bucket policy), the DSM firewall rule, Container Manager's project import.
