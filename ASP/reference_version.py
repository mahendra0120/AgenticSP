import re
from typing import Iterable, Optional

AUTO = "auto"
BASIC = "basic"
ADVANCED = "advanced"
VERSIONS = (AUTO, BASIC, ADVANCED)

CHOICES = [
    ("Auto: the model decides whether to use the Basic or Advanced pages", AUTO),
    ("Basic: the 5 reference pages only", BASIC),
    ("Advanced: also consult the Advanced pages", ADVANCED),
]

MARKER = re.compile(r"\[\[asp:reference-version=(basic|advanced)\]\]")

DIRECTIVES = {
    BASIC: (
        "[[asp:reference-version=basic]]\n"
        "Reference version selected by the user: BASIC. Use only the 5 reference images already "
        "attached to this conversation. The image viewer tools are disabled for this run."
    ),
    ADVANCED: (
        "[[asp:reference-version=advanced]]\n"
        "Reference version selected by the user: ADVANCED. In addition to the 5 attached reference "
        "images, you MUST consult the Advanced pages before writing your profile: call list_sections, "
        "then view_section for the sections relevant to this case, and use what you see in your analysis."
    ),
}


def normalize(version) -> str:
    return version if version in VERSIONS else AUTO


def label(version) -> str:
    names = {value: text.split(" - ")[0] for text, value in CHOICES}
    return names[normalize(version)]


def directive(version) -> Optional[str]:
    return DIRECTIVES.get(normalize(version))


def texts_of(content) -> Iterable[str]:
    if isinstance(content, str):
        yield content
    elif isinstance(content, (list, tuple)):
        for item in content:
            if isinstance(item, str):
                yield item


def version_from_messages(messages) -> str:
    for message in reversed(list(messages)):
        for part in getattr(message, "parts", ()):
            if getattr(part, "part_kind", None) != "user-prompt":
                continue
            for text in texts_of(getattr(part, "content", None)):
                found = MARKER.search(text)
                if found:
                    return found.group(1)
    return AUTO
