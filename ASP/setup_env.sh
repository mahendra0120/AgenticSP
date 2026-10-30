(return 0 2>/dev/null) || {
    echo "!! setup_env.sh must be SOURCED, not executed:"
    echo "     source setup_env.sh"
    exit 1
}

if [ ! -d /workspace ]; then
    echo "!! /workspace not found — attach a RunPod network volume at /workspace"
    echo "   (model weights, uv cache and DB backups all live there)."
    return 1
fi

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv/cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/.uv/python}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export npm_config_cache="${npm_config_cache:-/workspace/.npm}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$npm_config_cache"

if ! command -v curl >/dev/null 2>&1 || ! command -v gpg >/dev/null 2>&1; then
    echo ">> Installing curl/gnupg/ca-certificates..."
    $SUDO apt-get update -y
    $SUDO apt-get install -y --no-install-recommends curl ca-certificates gnupg
fi

if ! command -v npx >/dev/null 2>&1; then
    echo ">> Installing Node.js (npx not found on PATH)..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | $SUDO bash -
    $SUDO apt-get install -y nodejs
else
    echo ">> npx already installed ($(npx --version)) — skipping."
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo ">> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo ">> uv already installed ($(uv --version)) — skipping."
fi

if ! npm ls -g --depth=0 @modelcontextprotocol/server-filesystem >/dev/null 2>&1; then
    echo ">> Pre-installing @modelcontextprotocol/server-filesystem..."
    npm install -g @modelcontextprotocol/server-filesystem \
        || echo "!! Pre-install failed — the Profiler will try npx at request time instead."
fi

export PROFILER_AGENT_PORT="${PROFILER_AGENT_PORT:-8011}"
export FORENSIC_AGENT_PORT="${FORENSIC_AGENT_PORT:-8002}"

if [ -z "${HF_HUB_OFFLINE+x}" ]; then
    if ls -d "$HF_HOME"/hub/models--* >/dev/null 2>&1; then
        export HF_HUB_OFFLINE=1
    else
        echo ">> HF cache is empty — leaving HF_HUB_OFFLINE off so weights can download."
        echo "   (Set HF_TOKEN in .env first if either model repo is private/gated.)"
        echo "   Re-source this script after the first run to switch to offline mode."
    fi
fi

