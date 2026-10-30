set -euo pipefail

PG_DB="asp"
BACKUP_DIR="/workspace/pg-backups"
BACKUP_FILE="${BACKUP_DIR}/latest.dump"

mkdir -p "$BACKUP_DIR"

if [ "$(id -u)" -eq 0 ]; then
    PGRUN="runuser -u postgres --"
else
    PGRUN="sudo -u postgres"
fi
${PGRUN} pg_dump -Fc "${PG_DB}" -f "${BACKUP_FILE}.tmp"
mv "${BACKUP_FILE}.tmp" "$BACKUP_FILE"

echo "✅ Backup written to ${BACKUP_FILE}"
