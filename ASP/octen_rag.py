from __future__ import annotations
import os
import re
import shutil
from pathlib import Path
from typing import Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

FORENSIC_KNOWLEDGE_DIR = Path(__file__).parent / "forensic_knowledge"
INDEX_DIR = Path(__file__).parent / ".octen_faiss_index"
OCTEN_MODEL_ID = os.getenv("OCTEN_MODEL_ID", "Octen/Octen-Embedding-0.6B")

def patch_legacy_normalize_module() -> None:
    try:
        from sentence_transformers.base.modules.normalize import Normalize
    except ImportError:
        return
    if getattr(Normalize, "_octen_legacy_patch_applied", False):
        return
    original_init = Normalize.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self)

    Normalize.__init__ = patched_init
    Normalize._octen_legacy_patch_applied = True


def build_embeddings(model_id: str = OCTEN_MODEL_ID, device: str = "cpu"):
    patch_legacy_normalize_module()

    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": device, "trust_remote_code": True}
    )

def load_corpus_documents(chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    if not FORENSIC_KNOWLEDGE_DIR.exists():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    docs: list[Document] = []
    for md_path in sorted(FORENSIC_KNOWLEDGE_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(Document(
                page_content=chunk,
                metadata={"source": md_path.name, "chunk_index": i}
            ))
    return docs


vectorstore = None


def get_vectorstore():
    global vectorstore
    from langchain_community.vectorstores import FAISS

    if vectorstore is not None:
        return vectorstore

    embeddings = build_embeddings()

    if INDEX_DIR.exists():
        vectorstore = FAISS.load_local(
            str(INDEX_DIR), embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore

    docs = load_corpus_documents()
    if not docs:
        docs = [Document(page_content="", metadata={"source": "empty"})]
    vectorstore = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def clear_index_cache() -> bool:
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        return True
    return False


def query(text: str, top_k: int = 5) -> dict:
    store = get_vectorstore()
    raw = store.similarity_search_with_score(text, k=top_k)

    results = []
    for doc, distance in raw:
        if not doc.page_content:
            continue
        results.append({
            "text":        doc.page_content,
            "source":      doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
            "score":       round(1.0 / (1.0 + float(distance)), 4)
        })

    corpus_size = store.index.ntotal if hasattr(store, "index") else len(results)
    return {"results": results, "corpus_size": corpus_size}


def select_primary_source(results: list[dict]) -> Optional[str]:
    totals: dict[str, float] = {}
    for r in results:
        src = r.get("source")
        if src and r.get("text", "").strip():
            totals[src] = totals.get(src, 0.0) + float(r.get("score", 0.0))
    return max(totals, key=totals.get) if totals else None


def load_full_source_text(
    source: str, knowledge_dir: Path = FORENSIC_KNOWLEDGE_DIR
) -> Optional[str]:
    path = knowledge_dir / Path(source).name
    if path.suffix.lower() != ".md" or not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")

TRAUMA_KEYWORD_CATEGORIES: dict[str, list[str]] = {
    "general": [
        "evidence of trauma",
        "trauma",
        "traumatic injury",
        "signs of trauma",
        "penetrating trauma"
    ],
    "gunshot": [
        "gunshot wound",
        "gunshot",
        "bullet wound",
        "bullet hole",
        "entrance wound",
        "exit wound",
        "firearm injury",
        "projectile wound",
        "powder stippling",
        "muzzle imprint"
    ],
    "sharp_force": [
        "sharp force injury",
        "sharp force trauma",
        "stab wound",
        "incised wound",
        "incision wound",
        "puncture wound",
        "cut wound",
        "laceration",
        "chop wound"
    ],
    "asphyxia": [
        "asphyxia",
        "asphyxiation",
        "asphyxial",
        "strangulation",
        "ligature mark",
        "ligature furrow",
        "ligature",
        "petechial hemorrhage",
        "petechiae",
        "hanging",
        "smothering",
        "suffocation",
        "manual strangulation"
    ],
    "thermal": [
        "thermal injury",
        "burn injury",
        "burns",
        "charring",
        "scald",
        "fire-related death",
        "smoke inhalation"
    ],
    "blunt_force": [
        "blunt force trauma",
        "blunt force injury",
        "blunt trauma",
        "contusion",
        "abrasion",
        "hematoma",
        "fracture",
        "crush injury"
    ],
    "postmortem_and_natural": [
        "postmortem change",
        "livor mortis",
        "rigor mortis",
        "algor mortis",
        "decomposition",
        "sudden natural death"
    ]
}

TRAUMA_TRIGGER_KEYWORDS: list[str] = [
    kw for kws in TRAUMA_KEYWORD_CATEGORIES.values() for kw in kws
]

EVIDENCE_OF_TRAUMA_HEADING_RE = re.compile(
    r"^#{1,6}\s*(?:evidence of trauma(?:\s*\(\s*eot\s*\))?|eot)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE
)


def extract_evidence_of_trauma_section(report: str) -> Optional[str]:
    match = EVIDENCE_OF_TRAUMA_HEADING_RE.search(report)
    if not match:
        return None

    start = match.end()
    rest = report[start:]
    next_heading = re.search(r"^#{1,6}\s+\S", rest, re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(report)
    return report[start:end].strip()


def section_says_yes(section_text: str) -> bool:
    first_word = section_text.strip().split(None, 1)[0].strip(".:,").lower() if section_text.strip() else ""
    return first_word == "yes"


NEGATION_CUES_RE = re.compile(
    r"\b(no|not|without|absence of|negative for|denies|denied|"
    r"ruled out|free of|excludes?|unremarkable for)\b",
    re.IGNORECASE
)


def is_negated_before(sentence_lower: str, match_start: int, window_chars: int = 40) -> bool:
    preceding = sentence_lower[max(0, match_start - window_chars):match_start]
    return bool(NEGATION_CUES_RE.search(preceding))


def check_keyword_trigger(text: str, keywords: list[str]) -> dict:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    matched_keywords: list[str] = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        for kw in keywords:
            pattern = rf"\b{re.escape(kw.lower())}s?\b"
            for m in re.finditer(pattern, sentence_lower):
                if is_negated_before(sentence_lower, m.start()):
                    continue
                if kw not in matched_keywords:
                    matched_keywords.append(kw)
                break

    return {"triggered": bool(matched_keywords), "matched_keywords": matched_keywords}


def extract_trigger_context(text: str, matched_keywords: list[str], window: int = 1) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    hit_indices = set()
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        for kw in matched_keywords:
            pattern = rf"\b{re.escape(kw.lower())}s?\b"
            for m in re.finditer(pattern, sentence_lower):
                if not is_negated_before(sentence_lower, m.start()):
                    hit_indices.update(range(max(0, i - window), min(len(sentences), i + window + 1)))
                    break

    if not hit_indices:
        return text

    return " ".join(sentences[i] for i in sorted(hit_indices))

TRAUMA_CATEGORY_FILES: dict[str, str] = {
    "gunshot":     "Gunshot_wounds.md",
    "sharp_force": "sharp_force_injury.md",
    "asphyxia":    "Asphyxiation.md",
    "thermal":     "Thermal_injuries.md",
    "blunt_force": "Blunt_Force_Trauma.md"
}

TYPE_LABEL_TO_CATEGORY: list[tuple[str, str]] = [
    ("sharp force", "sharp_force"), ("blunt force", "blunt_force"),
    ("gunshot", "gunshot"), ("firearm", "gunshot"),
    ("asphyx", "asphyxia"), ("thermal", "thermal"), ("burn", "thermal"),
    ("sharp", "sharp_force"), ("blunt", "blunt_force")
]
TYPE_LABEL_RE = re.compile(
    r"^[\s>*\-\d.)]*\**\s*type\s*\**\s*[:\-\u2013\u2014]\s*\**\s*([A-Za-z][A-Za-z \-]*)",
    re.IGNORECASE | re.MULTILINE
)


def _labelled_categories(text: str) -> list[str]:
    cats: list[str] = []
    for m in TYPE_LABEL_RE.finditer(text):
        label = m.group(1).strip().lower()
        for key, category in TYPE_LABEL_TO_CATEGORY:
            if label.startswith(key):
                if category not in cats:
                    cats.append(category)
                break
    return cats


def select_trauma_sources(text: str, knowledge_dir: Path = FORENSIC_KNOWLEDGE_DIR) -> list[str]:
    labelled = [
        TRAUMA_CATEGORY_FILES[c] for c in _labelled_categories(text)
        if c in TRAUMA_CATEGORY_FILES and (knowledge_dir / TRAUMA_CATEGORY_FILES[c]).is_file()
    ]
    if labelled:
        return labelled

    lowered = text.lower()
    found: list[tuple[int, str]] = []
    for category, filename in TRAUMA_CATEGORY_FILES.items():
        if not (knowledge_dir / filename).is_file():
            continue
        hit = check_keyword_trigger(text, TRAUMA_KEYWORD_CATEGORIES[category])
        if not hit["triggered"]:
            continue
        pos = min(
            (lowered.find(kw.lower()) for kw in hit["matched_keywords"] if kw.lower() in lowered),
            default=len(lowered),
        )
        found.append((pos, filename))
    return [filename for f, filename in sorted(found)]


def check_trauma_and_retrieve(text: str, top_k: int = 5) -> dict:
    section = extract_evidence_of_trauma_section(text)
    if section is not None:
        if not section_says_yes(section):
            return {"triggered": False, "source": "evidence_of_trauma_section",
                    "query": None, "results": [], "corpus_size": 0}
        retrieval = query(section, top_k=top_k)
        return {
            "triggered": True, "source": "evidence_of_trauma_section",
            "query": section, **retrieval,
            "primary_source": select_primary_source(retrieval["results"]),
            "trauma_sources": select_trauma_sources(section)
        }

    trigger = check_keyword_trigger(text, TRAUMA_TRIGGER_KEYWORDS)
    if not trigger["triggered"]:
        return {"triggered": False, "source": "keyword_fallback",
                "query": None, "results": [], "corpus_size": 0}
    search_query = extract_trigger_context(text, trigger["matched_keywords"])
    retrieval = query(search_query, top_k=top_k)
    return {
        "triggered": True, "source": "keyword_fallback",
        "query": search_query, **retrieval,
        "primary_source": select_primary_source(retrieval["results"]),
        "trauma_sources": select_trauma_sources(text)
    }
