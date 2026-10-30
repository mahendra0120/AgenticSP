import json
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Optional
import httpx

@dataclass
class TextPart:
    text: str

    def to_dict(self) -> dict:
        return {"text": self.text}


@dataclass
class ImagePart:
    url: str
    media_type: Optional[str] = None

    def to_dict(self) -> dict:
        part: dict = {"url": self.url}
        if self.media_type:
            part["mediaType"] = self.media_type
        return part


@dataclass
class A2AMessage:
    parts: list[TextPart | ImagePart]
    role: str = "user"
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts],
            "messageId": self.message_id
        }


class A2AClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def stream_task(self, message: A2AMessage) -> AsyncIterator[str]:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/stream",
            "params": {"message": message.to_dict()}
        }

        assembled: dict[str, str] = {}
        emitted_len: dict[str, int] = {}

        stream_timeout = httpx.Timeout(self._timeout, read=1800.0)
        async with httpx.AsyncClient(timeout=stream_timeout) as c:
            async with c.stream(
                "POST", self.base_url, json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: "):])

                    if "error" in event:
                        raise RuntimeError(f"[A2A stream:{self.base_url}] {event['error']}")

                    result = event.get("result") or {}

                    artifact_update = result.get("artifactUpdate")
                    if artifact_update:
                        artifact = artifact_update["artifact"]
                        artifact_id = artifact["artifactId"]
                        chunk_text = "".join(
                            part.get("text", "") for part in artifact.get("parts", [])
                        )
                        if artifact_update.get("append", False):
                            assembled[artifact_id] = assembled.get(artifact_id, "") + chunk_text
                        else:
                            assembled[artifact_id] = chunk_text

                        new_text = assembled[artifact_id][emitted_len.get(artifact_id, 0):]
                        if new_text:
                            emitted_len[artifact_id] = len(assembled[artifact_id])
                            yield new_text

                    status_update = result.get("statusUpdate")
                    if status_update:
                        state = status_update["status"]["state"]
                        if state in ("failed", "rejected"):
                            err = status_update["status"].get("message")
                            raise RuntimeError(f"[A2A stream:{self.base_url}] task {state}: {err}")

PROFILER_AGENT = A2AClient(f"http://localhost:{os.getenv('PROFILER_AGENT_PORT', '8011')}")
FORENSIC_AGENT = A2AClient(f"http://localhost:{os.getenv('FORENSIC_AGENT_PORT', '8002')}")
