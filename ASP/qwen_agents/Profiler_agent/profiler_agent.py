import logging
import os
import sys
from pathlib import Path
from fastmcp.client import Client as FastMCPClient
from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPToolset
from qwen_agents.model_utils import QwenVLModel
from reference_version import ADVANCED, BASIC, version_from_messages

log = logging.getLogger(__name__)

PROFILER_MODEL_ID = os.getenv(
    "PROFILER_MODEL_ID", "Kizzington/Qwen3-VL-8B-Thinking-heretic"
)

PROFILER_LOAD_IN_4BIT = os.getenv("PROFILER_LOAD_IN_4BIT", "false").strip().lower() not in (
    "false", "0", "no"
)

SKILLS_DIR = Path(__file__).parent / ".agents" / "skills"
CBA_SKILL_DIR = SKILLS_DIR / "criminal-behavioral-analysis"
PROFILER_SKILL = CBA_SKILL_DIR / "SKILL.md"

SHORT_REFERENCE_IMAGES = sorted(
    (CBA_SKILL_DIR / "assets" / "short").glob("*.jpg")
)
ADVANCED_IMAGE_COUNT = len(list((CBA_SKILL_DIR / "assets" / "long").rglob("*.jpg")))

IMAGE_VIEWER_SERVER = Path(__file__).parent / "image_viewer_mcp.py"

skills_client = FastMCPClient(
    StdioTransport(
        command=sys.executable,
        args=[str(IMAGE_VIEWER_SERVER), str(CBA_SKILL_DIR)],
        env={**os.environ}
    ),
    init_timeout=60
)
ALLOWED_SKILL_TOOLS = {"list_sections", "view_section", "view_image"}
skills_toolset = MCPToolset(skills_client).filtered(
    lambda ctx, tool_def: (
        tool_def.name in ALLOWED_SKILL_TOOLS and version_from_messages(ctx.messages) != BASIC
    )
)


def load_skill_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Profiler agent's required skill file is missing: {path} "
            "this file defines the agent's core workflow and output "
            "format; the agent cannot run correctly without it."
        ) from e


SKILL_MD = load_skill_text(PROFILER_SKILL)
OUTPUT_MD = load_skill_text(CBA_SKILL_DIR / "references" / "OUTPUT.md")


profiler_agent: Agent[None, str] = Agent(
    model=QwenVLModel(
        PROFILER_MODEL_ID,
        processor_path="Qwen/Qwen3-VL-8B-Thinking",
        reference_images=SHORT_REFERENCE_IMAGES,
        temperature=0.3,
        max_new_tokens=12288,
        load_in_4bit=PROFILER_LOAD_IN_4BIT
    ).function_model(),
    system_prompt=(
        "You are a criminal behavioral profiler analyzing case images plus "
        "a Forensic agent's report (received via A2A).\n\n"
        "You are NOT deriving forensic conclusions yourself — use the "
        "Forensic report's findings as evidence for your profile. Do NOT "
        "just repeat or extend the Forensic report's own format (cause of "
        "death, manner of death, etc.) — your job is a DIFFERENT, "
        "behavioral-profiling report, as described in your skill file "
        "below.\n\n"
        "Note: some runs of this pipeline supply the Forensic report "
        "concurrently with RAG grounding lookup rather than waiting for "
        "it, so grounded reference material from forensic_knowledge/ may "
        "or may not be present below — proceed with whatever you're given "
        "and don't assume its absence means none exists.\n\n"
        "You have already been given several reference images (the 'Basic "
        "version' of your training material) summarizing established "
        "criminological classification frameworks for serial offenders, "
        "including:\n"
        "  - Palermo/Mastronardi's typology (Visionary, Missionary [with "
        "subtypes: social protection, pseudo-religious, pseudo-political, "
        "pseudo-cultural, racial, sexological-moralist, of the social "
        "order], Hedonistic, Power/Control-oriented, Lust-oriented, plus "
        "atypical variants)\n"
        "  - The Holmes & Holmes typology (male and female variants)\n"
        "  - Organized vs. Disorganized vs. Mixed crime-scene classification\n\n"
        "--- Your skill file (criminal-behavioral-analysis/SKILL.md) ---\n"
        "This defines your workflow. It is included here directly so you "
        "always have it regardless of whether you call any tool — you may "
        "still optionally use the image viewer tools to go deeper into the "
        "Advanced version images as it describes below.\n\n"
        f"TOOL RULES: your only tools are list_sections, view_section and view_image, "
        f"and they only show the {ADVANCED_IMAGE_COUNT} Advanced version reference images. "
        "list_sections lists the sections with a description of each image; view_section "
        "returns the actual images of one section (a few at a time, use offset to page); "
        "view_image returns one image by the path list_sections gave you. The case images "
        "are ALREADY attached to this message: never try to open, locate or re-read them "
        "with a tool, and never invent file paths such as /data/.... If a tool returns an "
        "error, do NOT repeat the same call; continue with what you already have.\n\n"
        f"{SKILL_MD}\n\n"
        "--- Your required output format (references/OUTPUT.md), verbatim ---\n"
        "You MUST use exactly these Markdown sections, in this order:\n\n"
        f"{OUTPUT_MD}\n\n"
        "---\n\n"
        "CRITICAL — avoid speculation: base every claim on what is "
        "actually visible in the images or stated in the Forensic report. "
        "Never invent specific unsupported narrative details (e.g. a "
        "specific relationship between victim and offender, a specific "
        "motive scenario, or a named category of crime) unless the "
        "evidence in front of you actually supports it — if the evidence "
        "is weak or absent for a claim, say so explicitly rather than "
        "filling the gap with an invented specific."
    ),
    toolsets=[skills_toolset],
    retries=4
)


@profiler_agent.output_validator
def warn_if_advanced_unused(ctx: RunContext[None], output: str) -> str:
    if version_from_messages(ctx.messages) == ADVANCED:
        viewed = any(
            getattr(part, "tool_name", None) in ALLOWED_SKILL_TOOLS
            for message in ctx.messages
            for part in getattr(message, "parts", ())
            if getattr(part, "part_kind", None) == "tool-call"
        )
        if not viewed:
            log.warning("Advanced reference version was selected but the model never viewed an Advanced page")
    return output
