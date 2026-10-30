set -euo pipefail

PG_USER="asp_user"
PG_PASSWORD="asp_password"
PG_DB="asp"
BACKUP_DIR="/workspace/pg-backups"
BACKUP_FILE="${BACKUP_DIR}/latest.dump"
BACKUP_INTERVAL_SECONDS="${BACKUP_INTERVAL_SECONDS:-300}" 

if [ "$(id -u)" -eq 0 ]; then
    PGRUN="runuser -u postgres --"
else
    PGRUN="sudo -u postgres"
fi

if [ ! -d /workspace ]; then
    echo "!! /workspace not found. This script expects a RunPod network volume"
    echo "mounted at /workspace. Aborting."
    exit 1
fi
mkdir -p "$BACKUP_DIR"

if ! command -v psql >/dev/null 2>&1; then
    echo "Installing PostgreSQL"
    apt-get update -y
    apt-get install -y postgresql
fi

PG_VERSION="$(ls /etc/postgresql | sort -V | tail -n1)"
PG_CLUSTER="main"
NATIVE_DATA_DIR="/var/lib/postgresql/${PG_VERSION}/${PG_CLUSTER}"
echo "Found PostgreSQL ${PG_VERSION}, cluster '${PG_CLUSTER}'"

NEEDS_REPAIR=false
if [ -L "$NATIVE_DATA_DIR" ]; then
    NEEDS_REPAIR=true
elif [ -d "$NATIVE_DATA_DIR" ] && [ "$(stat -c %U "$NATIVE_DATA_DIR" 2>/dev/null)" != "postgres" ]; then
    NEEDS_REPAIR=true
fi
if [ "$NEEDS_REPAIR" = true ]; then
    echo "Found a broken/root-owned data directory"
    echo "recreating the local cluster from scratch"
    pg_dropcluster --stop "${PG_VERSION}" "${PG_CLUSTER}" 2>/dev/null || true
    rm -rf "$NATIVE_DATA_DIR"
    rm -rf "/etc/postgresql/${PG_VERSION}/${PG_CLUSTER}"
    pg_createcluster "${PG_VERSION}" "${PG_CLUSTER}"
fi

echo "Starting PostgreSQL..."
service postgresql start || pg_ctlcluster "${PG_VERSION}" "${PG_CLUSTER}" start
for i in $(seq 1 30); do
    if ${PGRUN} pg_isready -q; then break; fi
    sleep 1
done

FRESH_CLUSTER=false
ROLE_EXISTS="$(${PGRUN} psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${PG_USER}'")"
if [ "$ROLE_EXISTS" != "1" ]; then
    echo "Creating role '${PG_USER}' and database '${PG_DB}'."
    ${PGRUN} psql -c "CREATE ROLE ${PG_USER} LOGIN PASSWORD '${PG_PASSWORD}';"
    ${PGRUN} createdb -O "${PG_USER}" "${PG_DB}"
    FRESH_CLUSTER=true
else
    echo "Role '${PG_USER}' already exists; Skipping creation"
fi

if [ "$FRESH_CLUSTER" = true ] && [ -f "$BACKUP_FILE" ]; then
    echo "Found existing backup at ${BACKUP_FILE} — restoring"
    ${PGRUN} pg_restore -d "${PG_DB}" --clean --if-exists "$BACKUP_FILE" \
        && echo " Restore complete." \
        || echo "!! Restore reported errors. Check output above."
else
    echo "No restore needed (either not a fresh cluster, or no backup found yet)."
fi

if ! pgrep -f "pg_dump.*${PG_DB}.*autobackup-loop" >/dev/null 2>&1; then
    echo "Starting background auto-backup loop (every ${BACKUP_INTERVAL_SECONDS}s)..."
    nohup bash -c "
        # marker string 'autobackup-loop' below is just so pgrep can find this loop
        while true; do
            sleep ${BACKUP_INTERVAL_SECONDS}
            ${PGRUN} pg_dump -Fc '${PG_DB}' -f '${BACKUP_FILE}.tmp' \
                && mv '${BACKUP_FILE}.tmp' '${BACKUP_FILE}' # autobackup-loop
        done
    " > "${BACKUP_DIR}/autobackup.log" 2>&1 &
    disown
fi

echo ""
echo "To take a manual backup right now, run: bash backup_postgres.sh"
