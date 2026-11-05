import io
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.utilities.types import Image
from PIL import Image as PILImage

DEFAULT_SKILL_DIR = Path(__file__).parent / ".agents" / "skills" / "criminal-behavioral-analysis"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_MARKDOWN = re.compile(r"!\[(?P<alt>.*?)\]\((?P<path>.*?)\)")
ADVANCED_HEADING = "# Advanced version"


@dataclass
class Section:
    number: int
    label: str
    title: str
    images: list[Path] = field(default_factory=list)


class AdvancedImageLibrary:
    def __init__(self, skill_dir: Path, max_side: int = 1600, max_per_call: int = 4):
        self.skill_dir = skill_dir.resolve()
        self.long_dir = self.skill_dir / "assets" / "long"
        self.reference_file = self.skill_dir / "references" / "REFERENCE.md"
        self.max_side = max_side
        self.max_per_call = max_per_call
        titles, self.descriptions = self.read_reference()
        self.sections = self.read_sections(titles)

    def read_reference(self) -> tuple[list[str], dict[Path, str]]:
        titles: list[str] = []
        alts: dict[Path, str] = {}
        if not self.reference_file.exists():
            return titles, alts
        in_advanced = False
        for line in self.reference_file.read_text(encoding="utf-8").splitlines():
            if line.strip() == ADVANCED_HEADING:
                in_advanced = True
            elif in_advanced and line.startswith("## "):
                titles.append(line[3:].strip())
            elif in_advanced:
                for match in IMAGE_MARKDOWN.finditer(line):
                    alts[(self.skill_dir / match["path"]).resolve()] = match["alt"].strip()
        return titles, alts

    def read_sections(self, titles: list[str]) -> list[Section]:
        folders = sorted(
            (d for d in self.long_dir.iterdir() if d.is_dir()),
            key=lambda d: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", d.name)],
        ) if self.long_dir.is_dir() else []
        use_titles = len(titles) == len(folders)
        sections = []
        for index, folder in enumerate(folders, start=1):
            images = sorted(
                (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
                key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)],
            )
            sections.append(Section(
                number=index,
                label=folder.name,
                title=titles[index - 1] if use_titles else folder.name,
                images=images,
            ))
        return sections

    @property
    def image_count(self) -> int:
        return sum(len(s.images) for s in self.sections)

    def describe(self, path: Path) -> str:
        return self.descriptions.get(path.resolve(), "")

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.long_dir).as_posix()

    def find_section(self, query: str) -> Section:
        wanted = str(query).strip().lower()
        if wanted.isdigit():
            for section in self.sections:
                if section.number == int(wanted):
                    return section
        for section in self.sections:
            if wanted and (wanted in section.title.lower() or wanted in section.label.lower()):
                return section
        choices = "; ".join(f"{s.number}: {s.title}" for s in self.sections)
        raise ToolError(f"No section matches {query!r}. Available sections: {choices}")

    def resolve_image(self, relative_path: str) -> Path:
        candidate = (self.long_dir / relative_path).resolve()
        if not candidate.is_relative_to(self.long_dir) or candidate.suffix.lower() not in IMAGE_SUFFIXES:
            raise ToolError(f"{relative_path!r} is not an Advanced version image")
        if not candidate.is_file():
            raise ToolError(f"{relative_path!r} does not exist; call list_sections to see valid paths")
        return candidate

    def load(self, path: Path) -> Image:
        with PILImage.open(path) as picture:
            if max(picture.size) <= self.max_side:
                return Image(path=str(path))
            picture = picture.convert("RGB")
            picture.thumbnail((self.max_side, self.max_side), PILImage.Resampling.LANCZOS)
            buffer = io.BytesIO()
            picture.save(buffer, format="JPEG", quality=88)
        return Image(data=buffer.getvalue(), format="jpeg")


def build_server(library: AdvancedImageLibrary) -> FastMCP:
    server = FastMCP("image-viewer")

    @server.tool
    def list_sections() -> list[dict]:
        return [
            {
                "section": s.number,
                "title": s.title,
                "image_count": len(s.images),
                "images": [
                    {"path": library.relative(p), "description": library.describe(p)}
                    for p in s.images
                ],
            }
            for s in library.sections
        ]

    @server.tool
    def view_section(section: str, offset: int = 0) -> list:
        chosen = library.find_section(section)
        start = max(int(offset), 0)
        batch = chosen.images[start:start + library.max_per_call]
        if not batch:
            raise ToolError(
                f"Section {chosen.number} has {len(chosen.images)} images; offset {start} is past the end"
            )
        end = start + len(batch)
        header = f"Section {chosen.number}: {chosen.title} - images {start + 1} to {end} of {len(chosen.images)}"
        footer = (
            f"{len(chosen.images) - end} more image(s) in this section; call view_section again with offset={end}."
            if end < len(chosen.images) else "End of section."
        )
        content: list = [header]
        for path in batch:
            description = library.describe(path)
            content.append(f"{library.relative(path)}" + (f" - {description}" if description else ""))
            content.append(library.load(path))
        content.append(footer)
        return content

    @server.tool
    def view_image(path: str) -> list:
        resolved = library.resolve_image(path)
        description = library.describe(resolved)
        return [library.relative(resolved) + (f" - {description}" if description else ""), library.load(resolved)]

    return server


def main() -> None:
    skill_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SKILL_DIR
    library = AdvancedImageLibrary(
        skill_dir,
        max_side=int(os.getenv("IMAGE_VIEWER_MAX_SIDE", "1600")),
        max_per_call=int(os.getenv("IMAGE_VIEWER_MAX_PER_CALL", "4")),
    )
    build_server(library).run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
