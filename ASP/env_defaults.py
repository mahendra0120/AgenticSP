import os
from pathlib import Path

DIR_ROOT = Path(__file__).resolve().parent
VAL_DEFAULTS = {
    "HF_HOME": "/workspace/hf_cache",
    "PROFILER_AGENT_PORT": "8011",
    "FORENSIC_AGENT_PORT": "8002",
}
KEYS_DEF = ("HF_HOME", "HF_HUB_OFFLINE", "PROFILER_AGENT_PORT", "FORENSIC_AGENT_PORT")


def apply() -> dict:
    for key in (*VAL_DEFAULTS, "HF_HUB_OFFLINE"):   
        if os.environ.get(key) == "":
            os.environ.pop(key)
    try:                                     
        from dotenv import load_dotenv
        load_dotenv(DIR_ROOT / ".env")
    except Exception:                       
        pass

    env = os.environ
    for key, default in VAL_DEFAULTS.items():
        if not env.get(key):                 
            env[key] = default

    if not env.get("HF_HUB_OFFLINE"):
        hub = Path(env["HF_HOME"]) / "hub"
        if hub.is_dir() and any(hub.glob("models--*")):
            env["HF_HUB_OFFLINE"] = "1"
        else:
            env.pop("HF_HUB_OFFLINE", None)  
    return {k: env.get(k) for k in KEYS_DEF}


SETTINGS = apply()

if __name__ == "__main__":
    for k in KEYS_DEF:
        print(f"   {k}={SETTINGS[k] if SETTINGS[k] is not None else '<unset>'}")
