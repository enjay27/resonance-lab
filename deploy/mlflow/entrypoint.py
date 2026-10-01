"""Container entrypoint: render MLflow's basic-auth config from the environment, then start `mlflow`.

MLflow reads the basic-auth database URI only from its config file, which would put a password into a committed file.
So the file in git (basic_auth.ini) holds a placeholder, and this script writes the real one to /tmp (a tmpfs) at start.
Standard library only: it runs in the image before MLflow does.
"""

import os
import sys

TEMPLATE = "/etc/mlflow/basic_auth.ini"
RENDERED = "/tmp/basic_auth.ini"
PLACEHOLDER = "@AUTH_DB_URI@"
URI_VARIABLE = "MLFLOW_AUTH_DB_URI"


def render_auth_config(template, db_uri):
    """The template with the database URI in place; `%` is doubled because MLflow reads the file with ConfigParser."""
    return template.replace(PLACEHOLDER, db_uri.replace("%", "%%"))


def prepare(environ, template_path=TEMPLATE, out_path=RENDERED):
    """Write the rendered config (owner-only: it holds a password); returns the environment for `mlflow`."""
    db_uri = environ.get(URI_VARIABLE)
    if not db_uri:
        raise SystemExit(f"{URI_VARIABLE} is not set: the basic-auth database URI comes from the environment (.env)")
    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(render_auth_config(template, db_uri))
    return {**environ, "MLFLOW_AUTH_CONFIG_PATH": out_path}


def main(argv, environ, template_path=TEMPLATE, out_path=RENDERED):
    env = prepare(environ, template_path, out_path)
    os.execvpe("mlflow", ["mlflow", *argv], env)


if __name__ == "__main__":
    main(sys.argv[1:], dict(os.environ))
