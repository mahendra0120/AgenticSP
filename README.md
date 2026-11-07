# Agentic_Serial-killer_Profiler

A two-agent, multimodal pipeline that turns case photographs into a structured forensic report and then, a behavioral profile. Both agents are built on **Qwen3-VL-8B-Thinking**, run locally, and talk to each other over the **A2A (Agent-to-Agent)** protocol. A **Gradio** app drives the workflow, **PostgreSQL** checkpoints every step so interrupted runs can resume, and a local **RAG** layer adds reference summaries on injury types for educational purposes.

> **Prototype for research and education.** Outputs can be wrong and must never be treated as forensic findings, evidence, or a basis for accusing anyone. See [Disclaimer](#disclaimer).

---

## Table of contents

- [How it works](#how-it-works)
- [Models](#models)
- [Requirements](#requirements)
- [Quick start (RunPod)](#quick-start-runpod)
- [Running without RunPod](#running-without-runpod)
- [Using the app](#using-the-app)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Logs, backups and troubleshooting](#logs-backups-and-troubleshooting)
- [Disclaimer](#disclaimer)

---

## How it works

```
Gradio UI (port 7860)
   |  up to 15 images + optional context
   v
PipelineRun  --  case id = SHA-256(images + context)[:12]  --  PostgreSQL checkpoints
   |
   |-- 1. Forensic agent (A2A, port 8002)
   |        -> Forensic Report
   |
   '-- 2. Run concurrently
         |-- RAG: read the EOT section, select reference files (FAISS + Octen embeddings)
         |  -> Reference summaries section: Forensic agent outputs each summary of listed injury
         |
         '-- Profiler agent (A2A, port 8011): images + forensic report
                 (+ MCP image viewer for the Advanced reference pages)
   |
   v
Final report -> pipeline_result.md and the Postgres 'final' step
```

- **Forensic agent**: Reads only the photographs and writes a forensic-style report that includes every type of each listed injury.
- **RAG step**: These aforementioned listed injury types would be selecting files from `ASP/forensic_knowledge/` via keyword trigger function, and the Forensic agent writes a short educational summary of each selected files.
- **Profiler agent**: Treats the forensic report as evidence and writes a five-section behavioral profile: Crime Scene Report, Victimology, Organization Level, Offender Typology, Rationale & Confidence.
- **Resumable**: Every step is saved to Postgres, so a crash or pod restart does not lose a long run.

## Models

The code refers to models only by Hugging Face id and downloads them on first start.

| Role | Model | Notes |
|---|---|---|
| Forensic agent | [`mahendra0120/Forensic-Agent-4.0-2026-09-08_14.31.12`](https://huggingface.co/mahendra0120/Forensic-Agent-4.0-2026-09-08_14.31.12) | LoRA adapter (r=8, SFT) on the Profiler's base model; merged at load time |
| Profiler agent | [`Kizzington/Qwen3-VL-8B-Thinking-heretic`](https://huggingface.co/Kizzington/Qwen3-VL-8B-Thinking-heretic) | About 17.5 GB in 4 shards; abliterated variant of Qwen3-VL-8B-Thinking |
| Processor / tokenizer | [`Qwen/Qwen3-VL-8B-Thinking`](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) | Config and tokenizer files only |
| RAG embeddings | [`Octen/Octen-Embedding-0.6B`](https://huggingface.co/Octen/Octen-Embedding-0.6B) | 1024-dim, runs on CPU |

The Hugging Face repository [`mahendra0120/Agentic_Serial-killer_Profiler`](https://huggingface.co/mahendra0120/Agentic_Serial-killer_Profiler) contains a ready-made `hf_cache/` folder with these models that cannot be found in the [`github repo`](https://github.com/mahendra0120/Agentic_Serial-killer_Profiler). It is not needed if the machine can download from the Hub.

## Requirements

- Linux with an NVIDIA GPU and driver. **Two GPUs are recommended**: each 8B model needs about 17.5 GB for weights in bfloat16, so one GPU holding both needs roughly 35 GB before activations. With one smaller GPU, enable 4-bit loading (see [Configuration](#configuration)).
- Python 3.13 (installed automatically by `uv`)
- [`uv`](https://docs.astral.sh/uv/) package manager
- PostgreSQL
- Disk space for model weights (about 20 GB) and the Python environment
- A Hugging Face token (`HF_TOKEN`) only if a model repo is private or gated

## Quick start (RunPod)

The scripts target a RunPod GPU pod with a **network volume mounted at `/workspace`** (it keeps model weights, the uv cache and database backups across restarts) and **HTTP port 7860** exposed.

```bash
cd /workspace
git clone https://github.com/mahendra0120/Agentic_Serial-killer_Profiler.git
cd Agentic_Serial-killer_Profiler/ASP
cp .env.example .env      # add HF_TOKEN if needed
bash start.sh
```

`start.sh` is safe to re-run after every pod restart. It does, in order:

| Step | Script | What it does |
|---|---|---|
| 1 | `setup_env.sh` | Installs `uv` if missing, points the Hugging Face and uv caches at `/workspace`, sets ports, switches the Hub to offline mode once models are cached |
| 2 | `setup_postgres.sh` | Installs and starts PostgreSQL, creates the `asp` role and database, restores `/workspace/pg-backups/latest.dump`, starts a backup loop (every 5 minutes) |
| 3 | `uv sync --frozen` | Installs Python dependencies from the lock file |
| 4 | `main.py` | Starts the Gradio UI on port 7860 |

Open the pod's proxy URL for port 7860, then press **Start Servers** (see below). The first start downloads the models, so the agent servers can take several minutes to become ready.

## Running without RunPod

`setup_env.sh` and `setup_postgres.sh` expect `/workspace`. On another machine you can either create that directory (`sudo mkdir -p /workspace`) and use `start.sh` as above, or set things up by hand:

```bash
git clone https://github.com/mahendra0120/Agentic_Serial-killer_Profiler.git
cd Agentic_Serial-killer_Profiler/ASP
cp .env.example .env

# 1. PostgreSQL (matches the defaults in db.py / .env.example)
sudo -u postgres psql -c "CREATE ROLE asp_user LOGIN PASSWORD 'asp_password';"
sudo -u postgres psql -c "CREATE DATABASE asp OWNER asp_user;"
# the pipeline_steps table is created automatically on first connection

# 2. Environment
export HF_HOME="$HOME/hf_cache"        # where models are stored
export PROFILER_AGENT_PORT=8011
export FORENSIC_AGENT_PORT=8002

# 3. Install and run
uv sync --frozen
uv run python main.py
```

If you change the database credentials, set `DATABASE_URL` in `.env`.

**Optional: reuse the pre-downloaded cache.** To copy the models from the [`Hugging Face repository`](https://huggingface.co/mahendra0120/Agentic_Serial-killer_Profiler) instead of downloading each one from its own repo:

```bash
pip install -U huggingface_hub
hf download mahendra0120/Agentic_Serial-killer_Profiler \
  --include "hf_cache/hub/*" --local-dir /workspace
```

The repository stores each file twice (under `blobs/` and `snapshots/`), so this download is larger than the weights themselves. Letting the first run fetch each model from the Hub is usually simpler.

## Using the app

1. **Start Servers.** Launches the Forensic and Profiler agent servers as separate processes. The status box shows each port and GPU assignment. **Stop Servers** shuts them down.
2. **Add images.** Upload or click thumbnails in the *Examples* gallery to reuse earlier uploads, up to 15 photographs.
3. **Add context (optional).** A short note about the scene or setting is passed to the Forensic agent.
4. **Choose the Profiler reference version.**
   - *Auto* (default): the model decides whether to consult the Advanced pages.
   - *Basic*: only the 5 Basic reference images are used; the viewer tools are removed.
   - *Advanced*: the Profiler is instructed to consult the 23 Advanced pages first. This is an instruction, not something the code can force; a warning is logged if the model skips it.
5. **Run Pipeline.** Output streams into the Forensic, Profiler and Reference Summaries panels. The combined report is shown under *Full Result* and saved to `pipeline_result.md`.

**Resuming.** Progress is saved to Postgres under a case id derived from the images and context, so running the same inputs again finds the same case. Choose how to treat saved work:

| Option | Behavior |
|---|---|
| Resume; continue partial reports | Finished steps are reused; an interrupted report is continued from where it stopped |
| Resume; use partial reports as-is | Accepts a partial report as complete |
| Start fresh | Deletes saved progress for the case and starts over |

You can also pick a saved case from the dropdown. Changing the reference version on a saved case re-runs only the Profiler. The *Manage saved progress* panel deletes one case, cases older than N days, or all cases (permanent; confirmation box required).

## Configuration

Set values in `ASP/.env` (copied from `.env.example`). Secrets such as `HF_TOKEN` belong only there; `.env` is gitignored.

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | unset; You have to manually configure it. | Needed only to download private or gated repos |
| `HF_HOME` | `/workspace/hf_cache` | Model cache location |
| `HF_HUB_OFFLINE` | auto | Set to `1` automatically once the cache holds models |
| `FORENSIC_AGENT_PORT` / `PROFILER_AGENT_PORT` | `8002` / `8011` | Agent server ports |
| `DATABASE_URL` | `postgresql://asp_user:asp_password@localhost:5432/asp` | Checkpoint database |
| `FORENSIC_MODEL_ID` / `PROFILER_MODEL_ID` | see [Models](#models) | Swap in different models or adapters |
| `FORENSIC_GPU` / `PROFILER_GPU` | auto | Pin each agent to a GPU index |
| `ASP_MULTI_GPU` | `true` | Set to `false` to keep both agents on the default GPU |
| `FORENSIC_LOAD_IN_4BIT` / `PROFILER_LOAD_IN_4BIT` | `false` | Load the model in 4-bit (nf4) to save memory |
| `ATTN_IMPLEMENTATION` | `sdpa` | Attention backend |
| `OCTEN_MODEL_ID` | `Octen/Octen-Embedding-0.6B` | Embedding model |
| `RAG_REBUILD_ON_STARTUP` | `true` | Rebuild the FAISS index on each server start |
| `RAG_CHARS` | `0` (no limit) | Truncate long reference files in summaries |
| `SERVER_READY_TIMEOUT_SECONDS` | `300` | How long to wait for the agent servers to open their ports |

With two or more visible GPUs, the Forensic agent uses the first and the Profiler the second.

## Project layout

Everything runs from the `ASP/` folder.

| Path | Role |
|---|---|
| `main.py` | Gradio UI, pipeline orchestration, agent server start/stop/restart |
| `a2a_server.py` | Serves one agent over A2A: `python a2a_server.py forensic` or `profiler` |
| `a2a_client.py` | Streaming A2A client used by `main.py` |
| `qwen_agents/model_utils.py` | `QwenVLModel`: Local Qwen3-VL inference (adapter loading, streaming, tool calls) as a pydantic-ai model |
| `qwen_agents/Forensic_agent/` | Forensic agent and its system prompt |
| `qwen_agents/Profiler_agent/` | Profiler agent, `image_viewer_mcp.py`, and the `criminal-behavioral-analysis` skill with its reference images |
| `octen_rag.py` | Chunking, FAISS index, and trauma-type routing over `forensic_knowledge/` |
| `forensic_knowledge/` | Seven reference markdown files used for summaries |
| `pipeline_state.py`, `db.py` | Postgres checkpointing and resume logic |
| `reference_version.py` | Basic / Advanced / Auto reference-version handling |
| `example_images.py` | Thumbnail gallery of earlier uploads |
| `env_defaults.py` | Loads `.env` and sets defaults before Hugging Face libraries are imported |
| `setup_env.sh`, `setup_postgres.sh`, `start.sh`, `backup_postgres.sh` | Bootstrap, database setup, startup and manual backup |
| `../pg-backups/` | `latest.dump` and `autobackup.log` | Results from the agentic pipeline are stored and backed up in this PostgreSQL directory |

## Logs, backups and troubleshooting

- **Logs:** `ASP/server_logs/forensic_server.log` and `profiler_server.log`. Each launch keeps the previous log as `*.log.prev`. `exit_codes.log` records how each agent process ended (for example `SIGKILL` usually means the GPU or system ran out of memory).
- **Backups:** The database is dumped to `/workspace/pg-backups/latest.dump` every 5 minutes. Run `bash backup_postgres.sh` for an immediate backup before stopping a pod.
- **Agent shows "not responding after 300s":** The first start is slow because models download and load(if you haven't downloaded the hf_cache directory from the [`huggingface repo`](https://huggingface.co/mahendra0120/Agentic_Serial-killer_Profiler)). Check the server log, or raise `SERVER_READY_TIMEOUT_SECONDS`.
- **"process exited early":** Read the matching log. Out-of-memory is the most common cause; Enable 4-bit loading or use two GPUs.
- **Output starts with `FAILURE:`:** The model ran out of its token budget while reasoning, or generation stalled. Re-run the case (resume mode reuses finished steps).
- **"Database unavailable":** The pipeline still runs, but progress is not saved. Check that PostgreSQL is running and `DATABASE_URL` is correct.
- **Models re-download every start:** Make sure `HF_HOME` points at the same persistent folder each time.
- **Changing models:** If `FORENSIC_MODEL_ID` points to an adapter repo (one containing `adapter_config.json`), the base model named inside it is downloaded and the adapter is merged automatically. Use `ADAPTER_BASE_MODEL_ID` to override the base.

## Disclaimer

- This is a **prototype**. The repository contains some fully abliterated vision-language models that can misread images and doesn't follow the system prompt to the fullest. A behavioral profile built on a mistaken forensic report inherits the mistake.
- Profiles describe a *most likely interpretation* of the evidence and rely on typology frameworks (Palermo/Mastronardi, Holmes & Holmes, organized/disorganized) that are debated and not validated predictors.
- The base model has had its refusal behavior removed. Run the system only in controlled settings with authorized, lawfully obtained material.
- Case images, generated reports and database dumps may contain sensitive material. Keep them private, do not commit them, and delete saved cases when finished.
