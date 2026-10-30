import hashlib
import re
from pathlib import Path
from typing import Callable, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
PREFIX_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_")


def original_name(path: Path) -> str:
    return PREFIX_UUID.sub("", path.name, count=1) or path.name


def digest(path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def list_example_images(folder: Path, limit: int = 24, scan_cap: int = 400) -> list[dict]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    seen, out = set(), []
    for p in files[:scan_cap]:
        sha = digest(p)
        if sha is None or sha in seen:
            continue
        seen.add(sha)
        out.append({"path": str(p), "name": original_name(p), "sha": sha})
        if len(out) >= limit:
            break
    return out


def normalize_paths(files) -> list[str]:
    out = []
    for f in files or []:
        if isinstance(f, (str, Path)):
            out.append(str(f))
        elif isinstance(f, dict):
            p = f.get("path") or f.get("name")
            if p:
                out.append(str(p))
        elif hasattr(f, "name"):
            out.append(str(f.name))
    return out


def add_unique(current: list[str], new: list[str], max_images: int,
               warn: Optional[Callable[[str], None]] = None) -> list[str]:
    result = list(current)
    have = {digest(p) for p in result} - {None}
    for p in new:
        if len(result) >= max_images:
            if warn:
                warn(f"Maximum {max_images} images per case - the rest were not added.")
            break
        d = digest(p)
        if d is None or d in have:
            continue
        have.add(d)
        result.append(str(p))
    return result
