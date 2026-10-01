# The MLflow tracking server. Built on the NAS from PyPI so it does not depend on what the official image ships:
# basic-auth needs the `auth` extra (Flask-WTF), and the plain package refuses to start it.
FROM python:3.12-slim
ARG MLFLOW_VERSION
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
RUN test -n "${MLFLOW_VERSION}" && pip install "mlflow[auth]==${MLFLOW_VERSION}"
WORKDIR /tmp
ENTRYPOINT ["mlflow"]
