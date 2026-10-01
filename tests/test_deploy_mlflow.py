"""Guards for deploy/mlflow/: the MLflow tracking server on the maintainer's NAS.

The server must be reachable only by one person, run no user code, and keep no secret in git. It keeps its records in
the Postgres of deploy/postgres/ and its small files in the NAS's MinIO. These tests read the files; the server with
these flags was smoke-tested (see the README), the image build and the containers on the NAS were not.
"""

import importlib.util
import os
import re

import pytest
import yaml

from config import BASE_DIR

DEPLOY = os.path.join(BASE_DIR, "deploy", "mlflow")


def _text(*parts):
    with open(os.path.join(BASE_DIR, *parts), encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(_text("deploy", "mlflow", "docker-compose.yml"))


@pytest.fixture(scope="module")
def service(compose):
    assert list(compose["services"]) == ["mlflow"]  # one container: the tracking server, nothing else
    return compose["services"]["mlflow"]


@pytest.fixture(scope="module")
def entrypoint():
    spec = importlib.util.spec_from_file_location("mlflow_entrypoint", os.path.join(DEPLOY, "entrypoint.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(path):
    values = {}
    for line in _text(*path).splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


# --- the container is locked down ----------------------------------------------------------------------------


def test_the_container_cannot_touch_the_host(service):
    assert service.get("privileged") is not True
    assert service.get("network_mode") != "host"
    assert service.get("pid") != "host"
    volumes = " ".join(str(v) for v in service.get("volumes", []))
    assert "docker.sock" not in volumes  # MLflow's assistant sandbox would drive docker through it


def test_the_container_is_read_only_unprivileged_and_limited(service):
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"] and "cap_add" not in service
    assert "no-new-privileges:true" in service["security_opt"]
    assert not str(service["user"]).startswith("0")  # never root; the NAS user owns the data files
    assert service["mem_limit"] and service["pids_limit"]
    assert service["restart"] == "unless-stopped"


def test_the_only_writable_place_is_tmp_because_the_data_lives_in_postgres_and_minio(service):
    assert all(str(v).endswith(":ro") for v in service["volumes"])  # config files only
    assert service["tmpfs"] == ["/tmp"]
    assert not re.search(r"MLFLOW_DATA_DIR|/data\b|NAS_UID", _text("deploy", "mlflow", "docker-compose.yml"))


def test_the_container_reaches_postgres_over_the_private_network_of_deploy_postgres(compose, service):
    assert service["networks"] == ["resonance-db"]
    assert compose["networks"]["resonance-db"] == {"name": "resonance-db", "external": True}


# --- reachable by one person on the LAN -----------------------------------------------------------------------


def test_the_port_is_bound_to_the_lan_address_you_set_never_to_every_interface(service):
    (port,) = service["ports"]
    assert port.startswith("${MLFLOW_BIND:?")  # required: no default, so no silent 0.0.0.0
    assert port.endswith(":5050")  # DSM itself uses 5000
    assert "5000:" not in port.replace("5050", "")


def _command(service):
    command = service["command"]
    assert isinstance(command, list)
    return command


def test_basic_auth_is_on_and_the_host_header_is_checked(service):
    command = _command(service)

    assert command[command.index("--app-name") + 1] == "basic-auth"
    assert "--allowed-hosts" in command and "${MLFLOW_ALLOWED_HOSTS:?" in command[command.index("--allowed-hosts") + 1]
    assert command[command.index("--cors-allowed-origins") + 1] == ""  # no browser origin may call the API
    assert "--disable-security-middleware" not in command
    assert "--dev" not in command


def test_the_server_only_serves_tracking_and_proxies_small_artifacts(service):
    command = _command(service)

    assert command[0] == "server"
    backend = command[command.index("--backend-store-uri") + 1]
    assert backend.startswith("postgresql+psycopg2://mlflow:${MLFLOW_DB_PASSWORD:?")  # the role of deploy/postgres/
    assert backend.endswith("@postgres:5432/mlflow")
    assert "--serve-artifacts" in command  # the desktop talks to MLflow only, never to MinIO
    assert command[command.index("--artifacts-destination") + 1].startswith("s3://${MINIO_BUCKET:?")
    assert command[command.index("--workers") + 1] == "1"  # one user, a 2-core NAS


def test_minio_is_the_artifact_store_and_its_key_comes_from_the_env_file(service):
    env = service["environment"]

    assert env["MLFLOW_S3_ENDPOINT_URL"].startswith("${MINIO_ENDPOINT_URL:?")
    assert env["AWS_ACCESS_KEY_ID"].startswith("${MINIO_ACCESS_KEY:?")
    assert env["AWS_SECRET_ACCESS_KEY"].startswith("${MINIO_SECRET_KEY:?")
    assert env["AWS_DEFAULT_REGION"]  # boto3 wants one; MinIO ignores it
    assert env["MLFLOW_AUTH_DB_URI"].startswith("postgresql+psycopg2://mlflow:${MLFLOW_DB_PASSWORD:?")
    assert env["MLFLOW_AUTH_DB_URI"].endswith("@postgres:5432/mlflow_auth")
    assert "MLFLOW_AUTH_CONFIG_PATH" not in env  # entrypoint.py sets it to the file it renders


def test_nothing_in_the_server_runs_user_code(service):
    env = service["environment"]

    # MLflow 3.x starts job workers by default (huey consumers that execute job functions).
    assert env["MLFLOW_SERVER_ENABLE_JOB_EXECUTION"] == "false"
    assert env["MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS"] == "false"
    assert env["MLFLOW_ENABLE_REMOTE_ASSISTANT"] == "false"
    assert env["MLFLOW_ENABLE_ASSISTANT_SANDBOX"] == "false"
    assert env["MLFLOW_DISABLE_TELEMETRY"] == "true" and env["DO_NOT_TRACK"] == "true"


def test_secrets_come_from_the_nas_env_file_and_must_be_set(service):
    env = service["environment"]

    assert env["MLFLOW_AUTH_ADMIN_PASSWORD"].startswith("${MLFLOW_AUTH_ADMIN_PASSWORD:?")
    assert env["MLFLOW_FLASK_SERVER_SECRET_KEY"].startswith("${MLFLOW_FLASK_SERVER_SECRET_KEY:?")
    literal = re.sub(r"\$\{[^}]*\}", "", _text("deploy", "mlflow", "docker-compose.yml"))  # without the ${VAR:?msg} references
    assert not re.search(r"(password|secret_key)[ \t]*[:=][ \t]*[^\s#]", literal, re.I)  # a value written into the file


def test_the_auth_config_grants_nothing_by_default_and_holds_no_secret():
    ini = _text("deploy", "mlflow", "basic_auth.ini")

    assert re.search(r"(?m)^default_permission\s*=\s*NO_PERMISSIONS\s*$", ini)  # MLflow's own default is READ
    assert re.search(r"(?m)^database_uri\s*=\s*@AUTH_DB_URI@\s*$", ini)  # filled in at start from the env, never written here
    assert not re.search(r"(?m)^admin_password\s*=", ini)
    assert "postgresql" not in ini and "sqlite" not in ini


# --- the entrypoint renders the auth config from the environment ------------------------------------------------


def test_the_auth_config_is_rendered_with_the_database_uri_from_the_environment(entrypoint):
    template = "[mlflow]\ndatabase_uri = @AUTH_DB_URI@\ndefault_permission = NO_PERMISSIONS\n"

    out = entrypoint.render_auth_config(template, "postgresql+psycopg2://mlflow:abc123@postgres:5432/mlflow_auth")

    assert "database_uri = postgresql+psycopg2://mlflow:abc123@postgres:5432/mlflow_auth\n" in out
    assert "default_permission = NO_PERMISSIONS" in out
    assert "@AUTH_DB_URI@" not in out


def test_a_percent_sign_in_the_uri_survives_the_ini_parser(entrypoint):
    import configparser

    out = entrypoint.render_auth_config("[mlflow]\ndatabase_uri = @AUTH_DB_URI@\n", "postgresql://u:p%40ss@h/db")
    parser = configparser.ConfigParser()
    parser.read_string(out)

    assert parser["mlflow"]["database_uri"] == "postgresql://u:p%40ss@h/db"  # MLflow reads it through ConfigParser


def test_the_real_template_renders_to_a_valid_config(entrypoint):
    import configparser

    out = entrypoint.render_auth_config(_text("deploy", "mlflow", "basic_auth.ini"), "postgresql://u:p@h/db")
    parser = configparser.ConfigParser()
    parser.read_string(out)

    assert parser["mlflow"]["database_uri"] == "postgresql://u:p@h/db"
    assert parser["mlflow"]["default_permission"] == "NO_PERMISSIONS"


def test_without_the_database_uri_the_server_does_not_start(entrypoint):
    with pytest.raises(SystemExit, match="MLFLOW_AUTH_DB_URI"):
        entrypoint.prepare({}, template_path="/nonexistent", out_path="/nonexistent")


def test_the_entrypoint_writes_the_config_privately_and_starts_mlflow(entrypoint, tmp_path, monkeypatch):
    template = tmp_path / "basic_auth.ini"
    template.write_text("[mlflow]\ndatabase_uri = @AUTH_DB_URI@\n", encoding="utf-8")
    out = tmp_path / "rendered.ini"
    started = {}
    monkeypatch.setattr(entrypoint.os, "execvpe", lambda file, args, env: started.update(file=file, args=args, env=env))

    entrypoint.main(["server", "--port", "5050"], {"MLFLOW_AUTH_DB_URI": "postgresql://u:p@h/db", "PATH": "/bin"},
                    template_path=str(template), out_path=str(out))

    assert started["file"] == "mlflow" and started["args"] == ["mlflow", "server", "--port", "5050"]
    assert started["env"]["MLFLOW_AUTH_CONFIG_PATH"] == str(out)
    assert "postgresql://u:p@h/db" in out.read_text(encoding="utf-8")
    if os.name == "posix":
        assert (out.stat().st_mode & 0o777) == 0o600  # it holds a password


# --- the image ----------------------------------------------------------------------------------------------------


def test_the_image_is_built_from_pinned_versions_with_the_auth_extra_and_the_postgres_and_s3_drivers():
    dockerfile = _text("deploy", "mlflow", "Dockerfile")

    assert re.search(r"(?m)^FROM python:3\.\d+-slim\s*$", dockerfile)
    assert 'mlflow[auth]==${MLFLOW_VERSION}' in dockerfile  # basic-auth needs Flask-WTF: the plain package does not start it
    assert re.search(r"psycopg2-binary==\d+\.\d+\.\d+", dockerfile)  # Postgres
    assert re.search(r"boto3==\d+\.\d+\.\d+", dockerfile)  # MinIO speaks S3
    assert "COPY entrypoint.py /opt/entrypoint.py" in dockerfile
    assert re.search(r'(?m)^ENTRYPOINT \["python", "/opt/entrypoint.py"\]\s*$', dockerfile)
    assert "latest" not in dockerfile


def test_compose_builds_that_image_at_the_version_in_the_env_file(service):
    assert service["build"]["args"]["MLFLOW_VERSION"].startswith("${MLFLOW_VERSION:?")
    # The tag names the drivers too: an image built before Postgres (SQLite era) had the plain `:<version>` tag, and compose reuses
    # an image that already has its tag instead of rebuilding ("No module named 'psycopg2'").
    assert service["image"] == "resonance-mlflow:${MLFLOW_VERSION}-pg"


# --- nothing secret is committed ------------------------------------------------------------------------------------


def test_the_example_env_files_hold_no_secret_and_pin_a_version():
    nas = _env(("deploy", "mlflow", ".env.example"))
    client = _env((".env.mlflow.example",))

    assert re.fullmatch(r"\d+\.\d+\.\d+", nas["MLFLOW_VERSION"])
    for secret in ("MLFLOW_AUTH_ADMIN_PASSWORD", "MLFLOW_FLASK_SERVER_SECRET_KEY", "MLFLOW_DB_PASSWORD", "MINIO_SECRET_KEY"):
        assert nas[secret] == "", secret
    assert client["MLFLOW_TRACKING_PASSWORD"] == "" and client["MLFLOW_TRACKING_URI"].startswith("http://")
    assert client["MLFLOW_DISABLE_TELEMETRY"] == "true"


def test_every_variable_the_compose_file_needs_is_in_the_example_env():
    compose = _text("deploy", "mlflow", "docker-compose.yml")
    needed = set(re.findall(r"\$\{([A-Z_]+)[:}-]", compose))

    assert needed <= set(_env(("deploy", "mlflow", ".env.example")))


def test_the_real_env_files_and_the_local_run_queue_are_gitignored():
    ignored = {line.strip() for line in _text(".gitignore").splitlines()}

    for pattern in ("deploy/mlflow/.env", "deploy/postgres/.env", ".env.mlflow", ".run.result.backup.json", ".run.result.backup.files/"):
        assert pattern in ignored, pattern
    assert ".env.mlflow.example" not in ignored and "deploy/mlflow/.env.example" not in ignored
    assert "deploy/postgres/.env.example" not in ignored


def test_the_desktop_client_is_pinned_to_the_servers_version():
    # a client and server of different versions break the API and the database schema
    server = _env(("deploy", "mlflow", ".env.example"))["MLFLOW_VERSION"]
    requirements = _text("requirements-llamafactory.txt")

    assert re.search(rf"(?m)^mlflow-skinny=={re.escape(server)}\s*(#.*)?$", requirements)
