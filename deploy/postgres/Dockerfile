# The official Postgres image plus the init script. The script is copied into the image instead of bind-mounted from the NAS:
# a bind-mounted folder keeps the NAS's file permissions, which the unprivileged postgres user (70) may not be able to read.
# Bump the major version deliberately (README: a major upgrade needs dump and restore).
FROM postgres:18-alpine
COPY initdb/ /docker-entrypoint-initdb.d/
RUN chmod -R a+rX /docker-entrypoint-initdb.d
