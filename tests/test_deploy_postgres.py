"""Guards for deploy/postgres/: the PostgreSQL server on the maintainer's NAS that holds MLflow's records.

It must be reachable only from the other containers of the project (never published on the LAN), run unprivileged,
and keep no secret in git. These tests read the files; the container itself was not run on a NAS.
"""

import os
import re

import pytest
import yaml

from config import BASE_DIR

DEPLOY = os.path.join(BASE_DIR, "deploy", "postgres")


def _text(*parts):
    with open(os.path.join(BASE_DIR, *parts), encoding="utf-8") as f:
        return f.read()


def _env(path):
    values = {}
    for line in _text(*path).splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(_text("deploy", "postgres", "docker-compose.yml"))


@pytest.fixture(scope="module")
def service(compose):
    assert list(compose["services"]) == ["postgres"]  # one container: the database, nothing else
    return compose["services"]["postgres"]


def test_the_image_is_postgres_18_pinned_to_its_major_version(service):
    assert re.fullmatch(r"postgres:18(\.\d+)?-alpine", service["image"])  # 18-alpine = the newest 18.x; 18.N-alpine pins a minor


def test_the_database_is_never_published_on_the_lan(compose, service):
    assert "ports" not in service  # only containers on the resonance-db network reach it
    assert service["networks"] == ["resonance-db"]
    assert compose["networks"]["resonance-db"] == {"name": "resonance-db"}  # deploy/mlflow joins it by this name
    assert service["container_name"] == "resonance-postgres"
    assert service.get("network_mode") != "host"


def test_the_container_is_unprivileged_read_only_and_limited(service):
    assert service["user"] == "70:70"  # the image's own postgres user: no root start, no chown, no capabilities
    assert service["read_only"] is True
    assert set(service["tmpfs"]) == {"/tmp", "/run/postgresql"}
    assert service["cap_drop"] == ["ALL"] and "cap_add" not in service
    assert "no-new-privileges:true" in service["security_opt"]
    assert service.get("privileged") is not True
    assert service["mem_limit"]
    assert service["restart"] == "unless-stopped"


def test_the_data_is_a_named_volume_and_the_init_script_is_read_only(compose, service):
    mounts = service["volumes"]

    # A named volume keeps the image's postgres ownership. Since 18 the image keeps its data in /var/lib/postgresql/18/docker:
    # the volume goes on /var/lib/postgresql (a mount on .../data would be a second, anonymous volume).
    assert "resonance-postgres-data:/var/lib/postgresql" in mounts
    assert "./initdb:/docker-entrypoint-initdb.d:ro" in mounts
    assert all(not str(v).startswith("/") for v in mounts)  # no host path, no docker.sock
    assert "resonance-postgres-data" in compose["volumes"]


def test_it_answers_a_health_check(service):
    assert "pg_isready" in " ".join(service["healthcheck"]["test"])


def test_passwords_come_from_the_nas_env_file_and_must_be_set(service):
    env = service["environment"]

    assert env["POSTGRES_PASSWORD"].startswith("${POSTGRES_PASSWORD:?")
    assert env["MLFLOW_DB_PASSWORD"].startswith("${MLFLOW_DB_PASSWORD:?")
    assert env["POSTGRES_HOST_AUTH_METHOD"] == "scram-sha-256"  # never `trust`
    literal = re.sub(r"\$\{[^}]*\}", "", _text("deploy", "postgres", "docker-compose.yml"))
    assert not re.search(r"(password)[ \t]*[:=][ \t]*[^\s#]", literal, re.I)


def test_the_init_script_creates_the_mlflow_role_and_both_databases_without_a_password_in_it():
    script = _text("deploy", "postgres", "initdb", "10-mlflow.sh")

    assert script.startswith("#!/bin/sh") or script.startswith("#!/usr/bin/env sh")
    assert "set -e" in script
    assert re.search(r"CREATE ROLE mlflow LOGIN PASSWORD :'mlflow_password'", script)  # psql quotes it: no shell interpolation in SQL
    assert 'mlflow_password="$MLFLOW_DB_PASSWORD"' in script
    assert re.search(r"CREATE DATABASE mlflow OWNER mlflow", script)
    assert re.search(r"CREATE DATABASE mlflow_auth OWNER mlflow", script)  # basic-auth's users, apart from the runs
    assert "REVOKE ALL ON DATABASE mlflow FROM PUBLIC" in script
    assert "SUPERUSER" not in script.upper().replace("POSTGRES_USER", "")
    assert "\r" not in script  # CRLF would break it inside the container


def test_the_example_env_holds_no_secret():
    env = _env(("deploy", "postgres", ".env.example"))

    assert env["POSTGRES_PASSWORD"] == "" and env["MLFLOW_DB_PASSWORD"] == ""


def test_every_variable_the_compose_file_needs_is_in_the_example_env():
    needed = set(re.findall(r"\$\{([A-Z_]+)[:}-]", _text("deploy", "postgres", "docker-compose.yml")))

    assert needed <= set(_env(("deploy", "postgres", ".env.example")))


def test_the_real_env_file_is_gitignored():
    ignored = {line.strip() for line in _text(".gitignore").splitlines()}

    assert "deploy/postgres/.env" in ignored
