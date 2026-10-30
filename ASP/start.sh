set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./setup_env.sh
bash ./setup_postgres.sh
uv sync --frozen
uv run python env_defaults.py || true
exec uv run python main.py
