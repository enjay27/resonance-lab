# The MLflow tracking server. Built on the NAS from PyPI so it does not depend on what the official image ships:
# basic-auth needs the `auth` extra (Flask-WTF), and the plain package refuses to start it. Records go to Postgres
# (psycopg2), small files to MinIO (boto3, MinIO speaks S3). Bump the two drivers like MLflow: deliberately.
FROM python:3.12-slim
ARG MLFLOW_VERSION
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
RUN test -n "${MLFLOW_VERSION}" && pip install "mlflow[auth]==${MLFLOW_VERSION}" psycopg2-binary==2.9.13 boto3==1.43.106
COPY entrypoint.py /opt/entrypoint.py
WORKDIR /tmp
ENTRYPOINT ["python", "/opt/entrypoint.py"]
