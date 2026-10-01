# MLflow tracking server on the Synology NAS

Records every training run of resonance-lab (parameters, metrics, tags, small reports) so runs can be compared without
pasting reports into chat. **One user (you), LAN only, runs no code, keeps metadata and small files only.** The plan and
the reasons are in `.memory/roadmap/mlflow-plan.md`.

Target: Synology DSM 7.2 with Container Manager, x86_64 (Intel Celeron J4025), 10 GB RAM. The server container is
limited to 1 GB and one worker.

## Set up on the NAS
1. **Folders.** In File Station create `docker/mlflow/data` (so `/volume1/docker/mlflow/data`). Copy this folder's
   `Dockerfile`, `docker-compose.yml`, `basic_auth.ini` and `.env.example` to `/volume1/docker/mlflow/`.
2. **Settings.** In `/volume1/docker/mlflow/` copy `.env.example` to `.env` and fill it in:
   - `NAS_UID` / `NAS_GID`: your DSM user (`id <user>` over SSH).
   - `MLFLOW_BIND`: the NAS LAN IP (give the NAS a fixed address or a DHCP reservation); `MLFLOW_ALLOWED_HOSTS`: that
     address and port, plus any host name you will use (`192.168.0.10:5050,nas.local:5050`).
   - `MLFLOW_AUTH_ADMIN_PASSWORD`: a long password. `MLFLOW_FLASK_SERVER_SECRET_KEY`: `openssl rand -hex 32`.
   - `.env` stays on the NAS and is gitignored here.
3. **Start.** Container Manager -> Project -> Create: path `/volume1/docker/mlflow`, use the existing
   `docker-compose.yml`, build and start. (Or over SSH: `cd /volume1/docker/mlflow && sudo docker compose up -d --build`.)
   The first build downloads MLflow from PyPI. On the first start MLflow creates the `admin` user with your password.
4. **Firewall.** Control Panel -> Security -> Firewall: allow TCP 5050 from the desktop's IP only, deny the rest. Do
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
| metadata + small files | SQLite (`/data/mlflow.db`) and artifacts under `/data/artifacts` through `--serve-artifacts` (the desktop needs no access to the NAS file system). resonance-lab logs only reports, predictions and configs, never data, weights or GGUFs |
| private | telemetry is off (`MLFLOW_DISABLE_TELEMETRY`, `DO_NOT_TRACK`); the client example sets it too |

The data directory must be on a local volume (SQLite does not like SMB/NFS shares).

## Backup and upgrade
- **Backup:** add `/volume1/docker/mlflow/data` to Hyper Backup. For a consistent copy of the SQLite files stop the
  project first (it is tiny: a few KB per run).
- **Upgrade:** back up, set the new `MLFLOW_VERSION` in `.env` **and** in resonance-lab's requirements (client and server
  must be the same version), rebuild, then migrate the store once:
  `sudo docker compose run --rm mlflow db upgrade sqlite:////data/mlflow.db`.

## What was and was not verified
Verified (MLflow 3.16.1, a real server with these flags and settings, in a throwaway environment): it starts with
basic-auth, the secret key and the job workers off; answers 401 without credentials, 403 for a foreign `Host`, 200 for the
admin; `/signup` and user creation answer 401; a client logs params, tags, 1000 metrics in one batch, a backdated start
time and an artifact through the proxy, and finds the run again by tag. `docker compose config` accepts the compose
file and refuses to start without `.env`.
**Not verified:** the image build and the container on the NAS (no Docker daemon in the session that wrote this): the
read-only root file system with `/tmp` as tmpfs, the DSM firewall rule, Container Manager's project import.
