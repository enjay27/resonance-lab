"""Guards for deploy/mlflow/: the MLflow tracking server on the maintainer's NAS.

The server must be reachable only by one person, run no user code, and keep no secret in git. These tests read the
files; the real server with these flags and settings was smoke-tested (401 without credentials, 403 for a foreign
Host header, params/metrics/tags/artifacts through the authenticated proxy), the image build on the NAS was not.
"""

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
def service():
    compose = yaml.safe_load(_text("deploy", "mlflow", "docker-compose.yml"))
    assert list(compose["services"]) == ["mlflow"]  # one container: the tracking server, nothing else
    return compose["services"]["mlflow"]


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


def test_the_only_writable_places_are_the_data_volume_and_tmp(service):
    mounts = [v for v in service["volumes"]]
    assert any(str(v).endswith(":/data") for v in mounts)
    assert all(str(v).endswith(":ro") or str(v).endswith(":/data") for v in mounts)
    assert service["tmpfs"] == ["/tmp"]


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
    assert command[command.index("--backend-store-uri") + 1].startswith("sqlite:////data/")
    assert "--serve-artifacts" in command
    assert command[command.index("--artifacts-destination") + 1].startswith("/data/")
    assert command[command.index("--workers") + 1] == "1"  # sqlite, one user, a 2-core NAS


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
    assert re.search(r"(?m)^database_uri\s*=\s*sqlite:////data/", ini)
    assert not re.search(r"(?m)^admin_password\s*=", ini)


# --- the image ----------------------------------------------------------------------------------------------------


def test_the_image_is_built_from_a_pinned_version_with_the_auth_extra():
    dockerfile = _text("deploy", "mlflow", "Dockerfile")

    assert re.search(r"(?m)^FROM python:3\.\d+-slim\s*$", dockerfile)
    assert 'mlflow[auth]==${MLFLOW_VERSION}' in dockerfile  # basic-auth needs Flask-WTF: the plain package does not start it
    assert re.search(r'(?m)^ENTRYPOINT \["mlflow"\]\s*$', dockerfile)
    assert "latest" not in dockerfile


def test_compose_builds_that_image_at_the_version_in_the_env_file(service):
    assert service["build"]["args"]["MLFLOW_VERSION"].startswith("${MLFLOW_VERSION:?")
    assert service["image"] == "resonance-mlflow:${MLFLOW_VERSION}"


# --- nothing secret is committed ------------------------------------------------------------------------------------


def test_the_example_env_files_hold_no_secret_and_pin_a_version():
    nas = _env(("deploy", "mlflow", ".env.example"))
    client = _env((".env.mlflow.example",))

    assert re.fullmatch(r"\d+\.\d+\.\d+", nas["MLFLOW_VERSION"])
    assert nas["MLFLOW_AUTH_ADMIN_PASSWORD"] == "" and nas["MLFLOW_FLASK_SERVER_SECRET_KEY"] == ""
    assert client["MLFLOW_TRACKING_PASSWORD"] == "" and client["MLFLOW_TRACKING_URI"].startswith("http://")
    assert client["MLFLOW_DISABLE_TELEMETRY"] == "true"


def test_every_variable_the_compose_file_needs_is_in_the_example_env():
    compose = _text("deploy", "mlflow", "docker-compose.yml")
    needed = set(re.findall(r"\$\{([A-Z_]+)[:}-]", compose))

    assert needed <= set(_env(("deploy", "mlflow", ".env.example")))


def test_the_real_env_files_and_the_local_run_queue_are_gitignored():
    ignored = {line.strip() for line in _text(".gitignore").splitlines()}

    for pattern in ("deploy/mlflow/.env", ".env.mlflow", ".run.result.backup.json", ".run.result.backup.files/"):
        assert pattern in ignored, pattern
    assert ".env.mlflow.example" not in ignored and "deploy/mlflow/.env.example" not in ignored


def test_the_desktop_client_is_pinned_to_the_servers_version():
    # a client and server of different versions break the API and the database schema
    server = _env(("deploy", "mlflow", ".env.example"))["MLFLOW_VERSION"]
    requirements = _text("requirements-llamafactory.txt")

    assert re.search(rf"(?m)^mlflow-skinny=={re.escape(server)}\s*(#.*)?$", requirements)
