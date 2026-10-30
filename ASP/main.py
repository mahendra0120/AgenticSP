import env_defaults 
import asyncio
import json
import logging
import mimetypes
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import aclosing
from pathlib import Path
import gradio as gr
import httpx
import example_images
import octen_rag
import pipeline_state as pstate
from a2a_client import (
    A2AMessage, TextPart, ImagePart,
    PROFILER_AGENT, FORENSIC_AGENT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PIPELINE] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "server_logs"
TEMP_IMAGE_DIR = Path("temp_uploads")
TEMP_IMAGE_DIR.mkdir(exist_ok=True)
PROFILER_PORT = int(os.getenv("PROFILER_AGENT_PORT", "8011"))
FORENSIC_PORT = int(os.getenv("FORENSIC_AGENT_PORT", "8002"))
AGENT_PORTS = {"Profiler": PROFILER_PORT, "Forensic": FORENSIC_PORT}

SERVER_READY_TIMEOUT = float(os.getenv("SERVER_READY_TIMEOUT_SECONDS", "300"))

MAX_IMAGES = 15

FORENSIC_USER_MESSAGE = "Analyze the attached autopsy photograph(s) and produce your forensic report."

def gpu_locator() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [t.strip() for t in visible.split(",") if t.strip()]
        return [] if not ids or ids[0] == "-1" else ids
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True
        ).stdout
        ids = [line.strip() for line in out.splitlines() if line.strip().isdigit()]
        if ids:
            return ids
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:                      
        return []


def plan_gpu_placement() -> dict[str, str | None]:
    """{"Forensic": gpu_or_None, "Profiler": gpu_or_None}. None = leave CUDA_VISIBLE_DEVICES alone."""
    forensic = os.getenv("FORENSIC_GPU", "").strip() or None
    profiler = os.getenv("PROFILER_GPU", "").strip() or None
    multi_ok = os.getenv("ASP_MULTI_GPU", "true").strip().lower() not in ("false", "0", "no")
    if forensic is None and profiler is None and multi_ok:
        gpus = gpu_locator()
        if len(gpus) >= 2:
            forensic, profiler = gpus[0], gpus[1]
    return {"Forensic": forensic, "Profiler": profiler}


gpu_placement: dict[str, str | None] | None = None


def agent_env(name: str) -> dict | None:
    global gpu_placement
    if gpu_placement is None:
        gpu_placement = plan_gpu_placement()
    gpu = gpu_placement.get(name)
    return {**os.environ, "CUDA_VISIBLE_DEVICES": gpu} if gpu is not None else None


def gpu_tag(name: str) -> str:
    gpu = (gpu_placement or {}).get(name)
    return f" (GPU {gpu})" if gpu is not None else ""

agent_procs: dict[str, subprocess.Popen] = {}


def open_port(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_ports(ports: dict[str, int], timeout: float = 60.0, interval: float = 1.0) -> dict[str, bool]:
    remaining = dict(ports)
    deadline = time.monotonic() + timeout
    ready: dict[str, bool] = {}
    while remaining and time.monotonic() < deadline:
        for label, port in list(remaining.items()):
            if open_port(port):
                ready[label] = True
                del remaining[label]
        if remaining:
            time.sleep(interval)
    for label in remaining:
        ready[label] = False
    return ready


def live_process(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def watch_exit(name: str, proc: subprocess.Popen) -> None:
    def wait_period():
        rc = proc.wait()
        try:
            sig = signal.Signals(-rc).name if rc < 0 else ""
        except ValueError:
            sig = ""
        msg = (f"[server-watch] {name} exited: returncode={rc} {sig} "
               f"at {time.strftime('%H:%M:%S')} (expected if you pressed Stop)")
        print(msg, flush=True)
        with open(LOG_DIR / "exit_codes.log", "a") as f:
            f.write(msg + "\n")

    threading.Thread(target=wait_period, daemon=True).start()


def spawn_agent(name: str) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    logfile = LOG_DIR / f"{name.lower()}_server.log"
    if logfile.exists():
        os.replace(logfile, logfile.with_name(logfile.name + ".prev"))
    agent_procs[name] = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "a2a_server.py"), name.lower()],
        stdout=open(logfile, "w"), stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT), env=agent_env(name)
    )
    watch_exit(name, agent_procs[name])
    return logfile

def launch_servers():
    global gpu_placement
    if len(agent_procs) == len(AGENT_PORTS) and all(live_process(p) for p in agent_procs.values()):
        return "Servers already running. Stop them first if you want to restart."

    if os.getenv("RAG_REBUILD_ON_STARTUP", "true").strip().lower() not in ("false", "0", "no"):
        if octen_rag.clear_index_cache():
            log.info("Cleared cached RAG index (.octen_faiss_index/). It will rebuild on first use.")

    gpu_placement = plan_gpu_placement()
    if gpu_placement["Forensic"] is not None or gpu_placement["Profiler"] is not None:
        log.info(
            f"GPU placement: Forensic -> GPU {gpu_placement['Forensic']}, "
            f"Profiler -> GPU {gpu_placement['Profiler']}"
        )
    else:
        log.info("GPU placement: not split (one GPU, none detected, or ASP_MULTI_GPU=false); Agents share the default GPU")

    try:
        logfiles = {name: spawn_agent(name) for name in AGENT_PORTS}
    except Exception as e:
        return f"Server launch failed: {e}"

    ready = wait_for_ports(AGENT_PORTS, timeout=SERVER_READY_TIMEOUT)

    lines = []
    for label, port in AGENT_PORTS.items():
        if ready.get(label):
            lines.append(f"{label}:{port}{gpu_tag(label)}")
        elif not live_process(agent_procs[label]):
            lines.append(f"{label}:{port}:- process exited early (see {logfiles[label]})")
        else:
            lines.append(f"{label}:{port}:- not responding after {int(SERVER_READY_TIMEOUT)}s (see {logfiles[label]})")
    return "\n".join(lines)

def stop_servers():
    stopped = []
    for name, proc in agent_procs.items():
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        stopped.append(name)
    agent_procs.clear()
    return f"Stopped: {', '.join(stopped)}" if stopped else "No servers were running."

async def agent_running(name: str) -> None:
    port = AGENT_PORTS[name]
    if open_port(port):
        return
    log.warning(f"{name} agent server is not listening on :{port} - restarting just that server")
    logfile = await asyncio.to_thread(spawn_agent, name)
    ready = await asyncio.to_thread(wait_for_ports, {name: port}, SERVER_READY_TIMEOUT)
    log.info(f"{name}:{port} restarted" if ready.get(name) else f"{name}:{port} did not come up (see {logfile})")

def guess_image_media_type(url_or_path: str) -> str:
    mime_type, mt = mimetypes.guess_type(url_or_path)
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    return "image/jpeg"


def save_uploaded_images(images) -> list[str]:
    urls = []
    for image in images or []:
        temp_path = TEMP_IMAGE_DIR / f"{uuid.uuid4()}_{Path(image.name).name if hasattr(image, 'name') else 'image.jpg'}"
        shutil.copy(image.name if hasattr(image, 'name') else image, temp_path)
        urls.append(f"http://127.0.0.1:7860/gradio_api/file={temp_path}")
        log.info(f"Image received and saved -> {temp_path.name}")
    return urls


def a2a_image_parts(image_urls: list[str]) -> list[ImagePart]:
    return [ImagePart(url=url, media_type=guess_image_media_type(url)) for url in image_urls]

async def run_pipeline_async(image_urls: list[str], extra_context: str, case_id: str, resume_mode: str = "continue"):
    forensic_text = profiler_text = rag_text = edu_text = ""
    store = pstate.CaseStore(case_id)

    def panels(msg: str, final: str = ""):
        return forensic_text, profiler_text, edu_text or rag_text, final, msg + store.status_suffix()

    try:
        forensic_message = FORENSIC_USER_MESSAGE
        if extra_context and extra_context.strip():
            forensic_message += f"\n\nAdditional Context:\n{extra_context}"

        saved: dict = {}
        if resume_mode == "fresh":
            await store.delete_all()
        else:
            saved = await store.load_all()
        await store.save("meta", pstate.meta_json(image_urls, extra_context), "complete")

        summary = ", ".join(f"{k}={v['status']}" for k, v in saved.items() if k != "meta")
        if summary:
            log.info(f"[case {case_id}] resuming with saved progress: {summary} (mode={resume_mode})")
        yield panels(
            f"Case {case_id}: found saved progress ({summary}); Resuming..." if summary
            else f"Case {case_id}: starting..."
        )

        sf = saved.get("forensic")
        forensic_done = False
        prefix = ""
        if sf and sf["status"] == "complete":
            forensic_text, forensic_done = sf["content"], True
            yield panels("Forensic report restored from the database; skipping that step")
        elif sf and sf["status"] == "partial":
            if resume_mode == "asis":
                forensic_text, forensic_done = sf["content"], True
                await store.save("forensic", forensic_text, "complete")
                await store.invalidate(["rag", "education", "profiler", "final"])
                yield panels("Using the saved partial Forensic report as-is")
            else:
                prefix = pstate.trim_partial(sf["content"])
                forensic_text = prefix
        forensic_reused = bool(sf and sf["status"] == "complete")

        if not forensic_done:
            await store.invalidate(["rag", "education", "profiler", "final"])
            if not open_port(FORENSIC_PORT):
                yield panels("Forensic server is down ; Restarting it (the model reloads on first use)")
                await agent_running("Forensic")
            prompt = forensic_message
            if prefix:
                prompt += pstate.continuation_block("forensic report", prefix)
            running = "Continuing the interrupted Forensic report" if prefix else "Running Forensic agent"
            yield panels(running)
            async for forensic_text in pstate.checkpointed(
                step_forensic_stream(image_urls, prompt), store, "forensic", prefix=prefix
            ):
                yield panels(running)
        yield panels("Forensic agent complete; Running Profiler agent and reference summaries concurrently")

        saved_rag = saved.get("rag") if forensic_reused else None
        sp = saved.get("profiler") if forensic_reused else None
        if sp:
            bad = profiler_output_problem(sp["content"])
            if bad:
                log.warning(f"[case {case_id}] saved Profiler result is invalid ({bad}); discarding it")
                await store.save("profiler_failed", sp["content"], "partial")
                await store.invalidate(["profiler", "final"])
                sp = None
                yield panels(
                    f"Saved Profiler result was invalid ({bad}) - discarded; "
                    "re-running only the Profiler (Forensic, RAG and summary are reused)"
                )

        saved_edu = saved.get("education") if forensic_reused else None
        rag_future: asyncio.Future = asyncio.get_running_loop().create_future()

        async def rag_step():
            try:
                data = None
                if saved_rag and saved_rag["status"] == "complete":
                    try:
                        data = json.loads(saved_rag["content"])
                    except ValueError:
                        pass
                if data is None:
                    data = await step_forensic_rag(case_id, forensic_text)
                    await store.save("rag", pstate.dumps(data), "complete")
            except asyncio.CancelledError:
                rag_future.cancel()
                raise
            except Exception as e:
                rag_future.set_exception(e)
                rag_future.exception()      
                raise
            rag_future.set_result(data)
            return data

        async def edu_step():
            try:
                if (saved_edu and saved_edu["status"] == "complete"
                        and "## Reference Summary" in saved_edu["content"]):
                    yield saved_edu["content"]
                    return
                section = octen_rag.extract_evidence_of_trauma_section(forensic_text)
                rag = {"triggered": True, "query": section or ""}
                sources = (
                    octen_rag.select_trauma_sources(section)
                    if section and octen_rag.section_says_yes(section) else []
                )
                if not sources:
                    try:
                        rag = await rag_future
                    except asyncio.CancelledError:
                        if rag_future.cancelled():      
                            return
                        raise
                    except Exception:                   
                        return
                    sources = education_sources(rag)
                if not sources:
                    return
                if not open_port(FORENSIC_PORT):
                    await agent_running("Forensic")
                async for text in pstate.checkpointed(
                    step_forensic_education_multi_stream(rag, sources), store, "education"
                ):
                    yield text
            except Exception as e:           
                log.warning(f"Reference summaries skipped: {e}")
                yield f"## Reference Summary\n\n_Summary unavailable: {e}"

        async def saved_text(t: str):
            yield t

        profiler_source = None
        p_prefix = ""
        if sp and sp["status"] == "complete":
            profiler_source = saved_text(sp["content"])
        elif sp and sp["status"] == "partial":
            if resume_mode == "asis":
                await store.save("profiler", sp["content"], "complete")
                profiler_source = saved_text(sp["content"])
            else:
                p_prefix = pstate.trim_partial(sp["content"])
                profiler_text = p_prefix
        if profiler_source is None:
            if not open_port(PROFILER_PORT):
                yield panels("Profiler server is down; Restarting it (the model reloads on first use)")
                await agent_running("Profiler")
            cont = pstate.continuation_block("behavioral profile", p_prefix) if p_prefix else None
            profiler_source = quarantine_bad_profiler(
                pstate.checkpointed(
                    guard_profiler_output(step_profiler_stream(image_urls, forensic_text, cont)),
                    store, "profiler", prefix=p_prefix
                ),
                store
            )

        rag_text = "Checking the Forensic report's Evidence of Trauma section"
        async with aclosing(concurrent_run(
            {"rag": flush_stream(rag_step()), "profiler": profiler_source, "edu": edu_step()}
        )) as stream:
            async for latest, errors in stream:
                if errors:
                    raise RuntimeError(
                        "; ".join(f"{k} step: {e}" for k, e in errors.items())
                    )
                if latest["rag"] != "" and not latest["rag"]["triggered"]:
                    rag_text = "_No evidence of trauma reported in the Forensic report; No reference summaries."
                if latest["profiler"] != "":
                    profiler_text = latest["profiler"]
                if latest["edu"] != "":
                    edu_text = latest["edu"]
                yield panels("Running Profiler agent and reference summaries concurrently")
        yield panels("Finalizing report")

        log.info(f"Combining Forensic, reference summaries, Profiler results into final profiling report (task_id={case_id})")
        sections = [
            f"# Case {case_id}",
            f"## Forensic Analysis\n\n{forensic_text}"
        ]
        if edu_text:
            sections.append(edu_text)    
        sections.append(f"## Profiler Analysis\n\n{profiler_text}")
        final_markdown = "\n\n".join(sections) + "\n"

        Path("pipeline_result.md").write_text(final_markdown)
        await store.save("final", final_markdown, "complete")

        yield panels("Pipeline completed successfully!", final_markdown)
    except Exception as e:
        await asyncio.sleep(0.3)   
        hint = (
            f"Progress is saved as case {case_id}; press Run Pipeline again "
            f"(resume mode) to continue from where it stopped."
            if store.ok else "the database is unavailable, so this progress was NOT saved."
        )
        yield forensic_text, profiler_text, rag_text, "", f"{e}{hint}"


async def flush_stream(coro):
    yield await coro


async def concurrent_run(gens: dict):
    queue: asyncio.Queue = asyncio.Queue()
    latest: dict = {k: "" for k in gens}
    pending = set(gens)

    async def pump(key, gen):
        try:
            async for chunk in gen:
                await queue.put((key, "chunk", chunk))
        except Exception as e:
            await queue.put((key, "error", e))
        finally:
            await queue.put((key, "done", None))

    tasks = [asyncio.create_task(pump(k, g)) for k, g in gens.items()]
    errors: dict[str, Exception] = {}

    try:
        while pending:
            key, kind, payload = await queue.get()
            if kind == "chunk":
                latest[key] = payload
                yield dict(latest), dict(errors)
            elif kind == "error":
                errors[key] = payload
                pending.discard(key)
                yield dict(latest), dict(errors)
            elif kind == "done":
                pending.discard(key)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


async def run_pipeline(images, image_url, extra_context, resume_mode="continue", saved_case=None):
    image_urls: list[str] = []

    if saved_case:
        meta = await pstate.CaseStore(saved_case).load_meta()
        if not meta or not meta.get("image_urls"):
            yield tuple([f"Saved case {saved_case} was not found in the database."] * 5)
            return
        image_urls = list(meta["image_urls"])
        extra_context = meta.get("extra_context", "")
        missing = [u for u in image_urls if not pstate.image_available(u)]
        if missing:
            yield tuple([
                f"Saved case {saved_case}: {len(missing)} image files no longer exist on disk. "
                f"Re-upload the original images with the same context and run again; the case id "
                f"will match and your saved progress will be used."
            ] * 5)
            return
        case_id = saved_case
    else:
        if images:
            if len(images) > MAX_IMAGES:
                yield tuple([f"Too many images ({len(images)}). Max is {MAX_IMAGES}."] * 5)
                return
            image_urls.extend(save_uploaded_images(images))

        if image_url and image_url.strip():
            image_urls.append(image_url.strip())

        if not image_urls:
            yield tuple(["Please provide at least one image upload or image URL"] * 5)
            return

        if len(image_urls) > MAX_IMAGES:
            yield tuple([f"Too many images ({len(image_urls)}). Max is {MAX_IMAGES}."] * 5)
            return
        case_id = pstate.case_id_for(image_urls, extra_context)

    log.info(f"Pipeline starting with {len(image_urls)} image(s) (case {case_id}, mode={resume_mode})")

    try:
        async for update in run_pipeline_async(image_urls, extra_context, case_id, resume_mode or "continue"):
            yield update
    except Exception as e:
        error = f"Critical error: {e}"
        yield tuple([error] * 5)

async def step_forensic_stream(image_urls: list[str], prompt: str):
    log.info(f"Streaming {len(image_urls)} images to Forensic agent")
    start = time.monotonic()
    text = ""
    try:
        async for delta in FORENSIC_AGENT.stream_task(
            A2AMessage(parts=[TextPart(prompt), *a2a_image_parts(image_urls)])
        ):
            text += delta
            yield text
    except (RuntimeError, httpx.HTTPError) as e:
        elapsed = time.monotonic() - start
        log.info(f"Forensic agent stream FAILED after {elapsed:.1f}s: {e}")
        raise RuntimeError(f"Forensic agent failed: {e}") from e
    elapsed = time.monotonic() - start
    log.info(f"Forensic agent stream finished in {elapsed:.1f}s ({len(text)} chars).Now, forwarding to Profiler via A2A")
    if not text:
        raise RuntimeError("Forensic agent returned an empty result")


async def step_forensic_rag(case_id: str, forensic_report: str) -> dict:
    log.info(f"Reading Forensic report's 'Evidence of Trauma' section (task_id={case_id})")
    rag_data = await asyncio.to_thread(octen_rag.check_trauma_and_retrieve, forensic_report)

    if rag_data["triggered"]:
        log.info(
            f"Triggered via {rag_data['source']} then "
            f"fetched {len(rag_data['results'])} chunk(s) from forensic_knowledge/ "
            f"(corpus_size={rag_data['corpus_size']}, task_id={case_id})"
        )
    else:
        log.info(f"No evidence of trauma reported; RAG pipeline not triggered (task_id={case_id})")

    return rag_data

RAG_CHARS = int(os.getenv("RAG_CHARS", "0"))


def education_sources(rag_data: dict) -> list[str]:
    if not rag_data.get("triggered"):
        return []
    sources = [
        s for s in dict.fromkeys(rag_data.get("trauma_sources") or [])
        if octen_rag.load_full_source_text(s) is not None
    ]
    if sources:
        return sources
    single = rag_data.get("primary_source") or octen_rag.select_primary_source(
        rag_data.get("results", [])
    )
    return [single] if single else []


def education_title(source: str) -> str:
    return Path(source).stem.replace("_", " ").replace("-", " ").strip().title()


def build_education_prompt(rag_data: dict, source: str, file_text: str, n_files: int = 1) -> str:
    findings = (rag_data.get("query") or "").strip()
    if len(findings) > 2000:
        findings = findings[:2000] + "…"
    truncation_note = ""
    if RAG_CHARS and len(file_text) > RAG_CHARS:
        file_text = file_text[:RAG_CHARS]
        truncation_note = (
            " NOTE: the file below was cut short to fit the model's context, "
            "so mention that your summary covers only the first part of it."
        )
    scope = (
        "the trauma type(s) named above"
        if n_files <= 1 else
        "those of the trauma types above that THIS file covers (the other types "
        "have their own reference files - do not discuss them, just skip them)"
    )
    return (
        "You are now in reference-summary mode (educational). No photographs "
        "are attached and you are NOT examining a case.\n\n"
        "The case report listed these trauma findings (context only - do not "
        f"analyse or re-examine the case):\n{findings}\n\n"
        f'Below is the complete text of the reference file "{source}". Write an '
        f"educational summary of what this file says about {scope}: key definitions, how the injury is produced, its "
        "characteristic features, how it is told apart from look-alikes, and "
        "any caveats or limitations the file mentions. Use short bold labels and "
        "bullet points, and do not use '#' markdown headings. Use only the file's content; if it does not cover one "
        f"of these, say so.{truncation_note}\n\n"
        f'<reference_document source="{source}">\n{file_text}\n</reference_document>'
    )


async def step_forensic_education_stream(rag_data: dict, source: str, n_files: int = 1):
    file_text = await asyncio.to_thread(octen_rag.load_full_source_text, source)
    if not file_text or not file_text.strip():
        raise RuntimeError(f"Reference file {source!r} not found in forensic_knowledge/")

    prompt = build_education_prompt(rag_data, source, file_text, n_files)
    log.info(
        f"Loading full {source} ({len(file_text):,} chars) into the Forensic agent "
        f"for an educational summary"
    )
    start = time.monotonic()
    text = ""
    try:
        async for delta in FORENSIC_AGENT.stream_task(A2AMessage(parts=[TextPart(prompt)])):
            text += delta
            yield text
    except (RuntimeError, httpx.HTTPError) as e:
        raise RuntimeError(f"Summary failed: {e}") from e
    log.info(f"Educational summary finished in {time.monotonic() - start:.1f}s ({len(text)} chars)")
    if not text:
        raise RuntimeError("Forensic agent returned an empty educational summary")


async def step_forensic_education_multi_stream(rag_data: dict, sources: list[str]):
    done = ""
    for source in sources:
        header = f"## Reference Summary - {education_title(source)}\n\n"
        text = ""
        yield done + header + "Loading the full reference file into the Forensic agent"
        async for text in step_forensic_education_stream(rag_data, source, len(sources)):
            yield done + header + text
        done += header + text + "\n\n"


async def step_profiler_stream(image_urls: list[str], forensic_report: str, continuation: str | None = None):
    log.info(f"Streaming {len(image_urls)} images and forensic result to Profiler agent")
    combined_prompt = f"Forensic agent report (from A2A):\n{forensic_report}"
    if continuation:
        combined_prompt += continuation
    start = time.monotonic()
    text = ""
    try:
        async for delta in PROFILER_AGENT.stream_task(
            A2AMessage(parts=[TextPart(combined_prompt), *a2a_image_parts(image_urls)])
        ):
            text += delta
            yield text
    except (RuntimeError, httpx.HTTPError) as e:
        elapsed = time.monotonic() - start
        log.info(f"Profiler agent stream failed after {elapsed:.1f}s: {e}")
        raise RuntimeError(f"Profiler agent failed: {e}") from e
    elapsed = time.monotonic() - start
    log.info(f"Profiler agent stream finished in {elapsed:.1f}s ({len(text)} chars). Now, combining into final profile")
    if not text:
        raise RuntimeError("Profiler agent returned an empty result")


class BadProfilerOutput(RuntimeError):
    def __init__(self, text: str, reason: str):
        super().__init__(reason)
        self.text = text


def profiler_output_problem(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return "empty output"
    if t.startswith("FAILURE: "):
        return "the model reported a generation failure instead of a report"
    if t.lower().startswith("<tool_call>"):
        return "the model emitted a raw tool-call block instead of a report"
    inner = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    if inner and inner[0] in "{[":
        try:
            obj = json.loads(inner)
        except ValueError:
            return None
        calls = obj if isinstance(obj, list) else [obj]
        if calls and all(isinstance(c, dict) and "name" in c and "arguments" in c for c in calls):
            names = ", ".join(str(c["name"]) for c in calls)
            return f"the model emitted a raw tool call ({names}) instead of a report"
    return None


async def guard_profiler_output(gen):
    last = ""
    async with aclosing(gen) as g:
        async for text in g:
            last = text
            yield text
    problem = profiler_output_problem(last)
    if problem:
        raise BadProfilerOutput(last, problem)


async def quarantine_bad_profiler(gen, store):
    try:
        async with aclosing(gen) as g:
            async for text in g:
                yield text
    except BadProfilerOutput as e:
        await store.save("profiler_failed", e.text, "partial")
        await store.invalidate(["profiler", "final"])
        raise RuntimeError(f"Profiler output was invalid ({e}) and was discarded") from e

async def list_cases_for_ui():
    rows = await pstate.list_cases_safe()
    return gr.Dropdown(choices=[(pstate.describe_case(r), r["case_id"]) for r in rows], value=None)


async def delete_reply(msg: str):
    return msg, await list_cases_for_ui(), gr.Checkbox(value=False)


confirm_popup = "Tick the confirmation box first; nothing was deleted."


async def delete_selected_case_ui(confirm, case_id):
    if not confirm:
        return await delete_reply(confirm_popup)
    if not case_id:
        return await delete_reply("Pick a saved case in the dropdown above first.")
    ok, res = await pstate.run_safe(pstate.db.forget_case(str(case_id)))
    msg = (f"Deleted case {case_id} ({res} saved steps)." if res else f"Nothing was saved under case {case_id}.") if ok \
        else f"Database unavailable; Nothing deleted: {res}"
    log.info(f"{msg}")
    return await delete_reply(msg)


async def delete_old_cases_ui(confirm, days):
    if not confirm:
        return await delete_reply(confirm_popup)
    try:
        days = int(days)
        if days < 1:
            raise ValueError
    except (TypeError, ValueError):
        return await delete_reply("Enter a whole number of days.")
    ok, res = await pstate.run_safe(pstate.db.delete_cases_older_than(days))
    msg = f"Deleted {res} cases with no activity in the last {days} days." if ok \
        else f"Database unavailable; Nothing deleted: {res}"
    log.info(f"{msg}")
    return await delete_reply(msg)


async def delete_all_cases_ui(confirm):
    if not confirm:
        return await delete_reply(confirm_popup)
    ok, res = await pstate.run_safe(pstate.db.delete_all_cases())
    msg = f"Deleted all {res} saved cases." if ok else f"Database unavailable; Nothing deleted: {res}"
    log.info(f"{msg}")
    return await delete_reply(msg)


def refresh_examples():
    if os.getenv("SHOW_UPLOAD_EXAMPLES", "1") == "0":
        return [], []
    items = example_images.list_example_images(TEMP_IMAGE_DIR, limit=24)
    return [(it["path"], it["name"]) for it in items], [it["path"] for it in items]


def add_example(current_files, example_list, evt: gr.SelectData):
    files = example_images.normalize_paths(current_files)
    try:
        chosen = (example_list or [])[int(evt.index)]
    except (TypeError, ValueError, IndexError):
        return files or None
    return example_images.add_unique(files, [chosen], MAX_IMAGES, warn=gr.Warning) or None


def add_all_examples(current_files, example_list):
    files = example_images.normalize_paths(current_files)
    return example_images.add_unique(files, list(example_list or []), MAX_IMAGES, warn=gr.Warning) or None


with gr.Blocks(title="A2A Multi-Agent Pipeline") as demo:
    gr.Markdown("# A2A 2-Agent Vision Pipeline\nForensic → Profiler | Image Upload + Extra Context")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Server Control")
            server_status = gr.Textbox(label="Status", value="Not started", lines=4, interactive=False)
            with gr.Row():
                gr.Button("Start Servers", variant="primary").click(launch_servers, outputs=server_status)
                gr.Button("Stop Servers").click(stop_servers, outputs=server_status)

        with gr.Column(scale=2):
            gr.Markdown("### Input")
            with gr.Tabs():
                with gr.Tab("Upload Images"):
                    image_input = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload Photos (max 15)",
                    )
                with gr.Tab("Image URL"):
                    url_input = gr.Textbox(label="Image URL", placeholder="https://...", lines=1)

            extra_context = gr.Textbox(
                label="Additional Context (Optional)",
                placeholder="This photo was taken in a forest during autumn. The person is a biologist studying mushrooms.",
                lines=3
            )

            with gr.Accordion("Examples :- photos you uploaded before", open=True):
                gr.Markdown(
                    "Click a thumbnail to add it to **Upload Photos** above. The same photo is never added twice. "
                    "Pick the same photos in the same order as before to find that case's saved progress."
                )
                example_gallery = gr.Gallery(
                    show_label=False, columns=6, height=220, object_fit="cover",
                    allow_preview=False, interactive=False,
                )
                example_paths = gr.State([])
                with gr.Row():
                    add_all_btn = gr.Button("Add all shown (up to 15)")
                    refresh_examples_btn = gr.Button("Refresh examples")

    run_btn = gr.Button("Run Pipeline", variant="primary", size="large")

    with gr.Row():
        resume_mode = gr.Radio(
            choices=[
                ("Resume; continue partial reports", "continue"),
                ("Resume; use partial reports as-is", "asis"),
                ("Start fresh (discard saved progress)", "fresh"),
            ],
            value="continue",
            label="Progress is saved to Postgres. If this exact case (same images + context) was run before:",
        )
        saved_case = gr.Dropdown(
            choices=[], value=None, allow_custom_value=True,
            label="or resume a saved case (overrides the inputs above)",
        )
        refresh_cases = gr.Button("Refresh saved cases")

    with gr.Accordion("Manage saved progress", open=False):
        gr.Markdown(
            "Saved cases accumulate in Postgres until you delete them. Deleting is **permanent**: "
            "tick the box, then press one button. Image files in `temp_uploads/` are not touched. "
            "Avoid deleting a case while it is still running."
        )
        delete_confirm = gr.Checkbox(label="I understand this permanently deletes saved progress", value=False)
        with gr.Row():
            delete_case_btn = gr.Button("Delete the case selected above")
            older_days = gr.Number(label="Older than (days)", value=7, minimum=1, precision=0)
            delete_old_btn = gr.Button("Delete cases older than N days")
            delete_all_btn = gr.Button("Delete ALL saved cases", variant="stop")
        memory_status = gr.Textbox(label="Result", interactive=False)

    with gr.Row():
        with gr.Column():
            forensic_out = gr.Markdown(label="Forensic Agent")
        with gr.Column():
            profiler_out = gr.Markdown(label="Profiler Agent")

    gr.Markdown(
        "### Reference Summaries"
    )
    rag_out = gr.Markdown(label="Reference Summaries")

    with gr.Accordion("Full Result", open=False):
        full_out = gr.Markdown(label="Complete Report")

    status_out = gr.Textbox(label="Pipeline Status", interactive=False)

    run_btn.click(
        fn=run_pipeline,
        inputs=[image_input, url_input, extra_context, resume_mode, saved_case],
        outputs=[forensic_out, profiler_out, rag_out, full_out, status_out],
    ).then(fn=list_cases_for_ui, outputs=saved_case).then(
        fn=refresh_examples, outputs=[example_gallery, example_paths]
    )
    refresh_cases.click(fn=list_cases_for_ui, outputs=saved_case)
    demo.load(fn=list_cases_for_ui, outputs=saved_case)
    demo.load(fn=refresh_examples, outputs=[example_gallery, example_paths])
    refresh_examples_btn.click(fn=refresh_examples, outputs=[example_gallery, example_paths])
    example_gallery.select(fn=add_example, inputs=[image_input, example_paths], outputs=image_input)
    add_all_btn.click(fn=add_all_examples, inputs=[image_input, example_paths], outputs=image_input)
    delete_case_btn.click(fn=delete_selected_case_ui, inputs=[delete_confirm, saved_case],
                          outputs=[memory_status, saved_case, delete_confirm])
    delete_old_btn.click(fn=delete_old_cases_ui, inputs=[delete_confirm, older_days],
                         outputs=[memory_status, saved_case, delete_confirm])
    delete_all_btn.click(fn=delete_all_cases_ui, inputs=[delete_confirm],
                         outputs=[memory_status, saved_case, delete_confirm])

    gr.Markdown("**Tip:** Output from the agents could be **inaccurate** and so **please** keep in mind this is just a **prototype**.")

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
        allowed_paths=[str(TEMP_IMAGE_DIR.resolve())],
    )
