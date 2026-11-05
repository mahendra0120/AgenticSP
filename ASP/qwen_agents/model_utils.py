from __future__ import annotations
import env_defaults 
import torch
import asyncio
import io
import json
import logging
import os
import re
import threading
import uuid
from contextlib import aclosing
from functools import lru_cache
from pathlib import Path
from typing import Optional
from PIL import Image
from transformers import AutoModelForImageTextToText
from huggingface_hub import login, snapshot_download
from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from transformers import AutoProcessor, TextIteratorStreamer

log = logging.getLogger("model_utils")

if os.getenv("HF_TOKEN"):
    if os.getenv("HF_HUB_OFFLINE"):
        log.info("HF_HUB_OFFLINE is set;skipping huggingface_hub login().")
    else:
        login(token=os.getenv("HF_TOKEN"))

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def download(repo_id: str) -> str:
    return snapshot_download(repo_id=repo_id, token=os.getenv("HF_TOKEN"))


def load_with_transformers(local_path: str, processor_path: Optional[str], load_in_4bit: bool):
    kwargs = dict(
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else None,
        attn_implementation=os.getenv("ATTN_IMPLEMENTATION", "sdpa")
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

    root = Path(local_path)
    adapter_cfg_file = root / "adapter_config.json"
    base_path = None
    if adapter_cfg_file.exists():
        base_id = os.getenv("ADAPTER_BASE_MODEL_ID") or json.loads(
            adapter_cfg_file.read_text()
        ).get("base_model_name_or_path")
        if not base_id:
            raise RuntimeError(
                f"{local_path!r} is an adapter repo but adapter_config.json names no "
                f"base model; set ADAPTER_BASE_MODEL_ID."
            )
        if "bnb-4bit" in base_id and not load_in_4bit:
            log.warning(
                f"adapter base {base_id!r} is a pre-quantized bnb-4bit repo; if loading "
                f"or merging fails, set ADAPTER_BASE_MODEL_ID to a full-precision base."
            )
        base_path = base_id if Path(base_id).exists() else download(base_id)
        log.info(f"Adapter repo detected; loading base {base_id!r} then applying {local_path!r}")
        model = AutoModelForImageTextToText.from_pretrained(base_path, **kwargs)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, local_path)
        if not load_in_4bit:
            try:
                model = model.merge_and_unload()
            except Exception as e:
                log.warning(f"Could not merge adapter ({e!r}); running with the adapter attached.")
    else:
        model = AutoModelForImageTextToText.from_pretrained(local_path, **kwargs)
    model.eval()

    def is_real_vlm_processor(proc) -> bool:
        return proc is not None and hasattr(proc, "image_processor")

    candidates = [processor_path] if processor_path else [local_path] + ([base_path] if base_path else [])
    processor = None
    errors = []
    for src in candidates:
        try:
            proc = AutoProcessor.from_pretrained(src)
        except Exception as e:
            errors.append(f"{src}: {e!r}")
            continue
        if is_real_vlm_processor(proc):
            processor = proc
            break
        errors.append(f"{src}: loaded {type(proc).__name__} without an image_processor")
    if processor is None:
        raise RuntimeError(
            f"Model loaded, but no usable image processor was found. Tried: {errors}. "
            f"Pass processor_path (e.g. 'Qwen/Qwen3-VL-8B-Thinking') to QwenVLModel()."
        )
    return model, processor


@lru_cache(maxsize=4)
def load(model_id: str, processor_path: Optional[str], load_in_4bit: bool):
    return load_with_transformers(download(model_id), processor_path, load_in_4bit)


def is_image_content(piece) -> bool:
    return isinstance(piece, BinaryContent) and piece.is_image


def tool_return_items(part: ToolReturnPart) -> list[dict]:
    pieces = part.content if isinstance(part.content, (list, tuple)) else [part.content]
    header = f"Tool '{part.tool_name}' returned:\n"
    if not any(is_image_content(p) for p in pieces):
        return [{"type": "text", "text": header + part.model_response_str()}]
    items: list[dict] = [{"type": "text", "text": header.rstrip()}]
    for piece in pieces:
        if is_image_content(piece):
            picture = Image.open(io.BytesIO(piece.data)).convert("RGB")
            items.append({"type": "image", "image": picture})
        elif isinstance(piece, str):
            items.append({"type": "text", "text": piece})
        else:
            items.append({"type": "text", "text": json.dumps(piece, default=str)})
    return items


def extract_chat_messages(messages: list[ModelMessage]) -> list[dict]:
    chat: list[dict] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            user_content_items: list[dict] = []
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    chat.append(
                        {"role": "system", "content": [{"type": "text", "text": part.content}]}
                    )
                elif isinstance(part, UserPromptPart):
                    raw = part.content
                    pieces = raw if isinstance(raw, list) else [raw]
                    for piece in pieces:
                        if isinstance(piece, str):
                            user_content_items.append({"type": "text", "text": piece})
                        elif isinstance(piece, ImageUrl):
                            user_content_items.append({"type": "image", "image": piece.url})
                        elif isinstance(piece, BinaryContent) and piece.is_image:
                            user_content_items.append({"type": "image", "image": piece.data})
                        elif isinstance(piece, DocumentUrl):
                            user_content_items.append({"type": "image", "image": piece.url})
                elif isinstance(part, ToolReturnPart):
                    user_content_items.extend(tool_return_items(part))
            if user_content_items:
                chat.append({"role": "user", "content": user_content_items})
        elif isinstance(msg, ModelResponse):
            text_parts = []
            for part in msg.parts:
                if isinstance(part, TextPart):
                    text_parts.append(part.content)
                elif isinstance(part, ToolCallPart):
                    call_repr = json.dumps({"name": part.tool_name, "arguments": part.args})
                    text_parts.append(f"<tool_call>{call_repr}</tool_call>")
            if text_parts:
                chat.append(
                    {"role": "assistant", "content": [{"type": "text", "text": "\n".join(text_parts)}]}
                )
    return chat


def build_tool_instructions(agent_info: AgentInfo) -> Optional[str]:
    tools = list(agent_info.function_tools or [])
    if not tools:
        return None

    tool_lines = []
    for t in tools:
        schema = json.dumps(t.parameters_json_schema)
        tool_lines.append(f"- {t.name}: {t.description}\n  Arguments schema: {schema}")

    return (
        "You have access to the following tools:\n\n"
        + "\n".join(tool_lines)
        + "\n\nTo call a tool, respond with ONLY this and nothing else "
        "(no prose, no explanation, exact format):\n"
        '<tool_call>{"name": "<tool_name>", "arguments": {<matching the schema above>}}</tool_call>\n\n'
        "After a tool result is shown to you, continue and give your real "
        "answer as plain text (no tool_call tag) once you have what you need."
    )


def strip_thinking(raw_text: str) -> str:
    if "</think>" in raw_text:
        return raw_text.split("</think>", 1)[1]
    if "<think>" in raw_text:
        return raw_text.split("<think>", 1)[0]
    return raw_text


def strip_special_tokens(text: str, tokenizer) -> str:
    return replace_special_tokens(text, tokenizer).strip()


def replace_special_tokens(text: str, tokenizer) -> str:
    for special_tok in getattr(tokenizer, "all_special_tokens", []):
        text = text.replace(special_tok, "")
    return text


@lru_cache(maxsize=None)
def load_reference_image(path: str, max_side: int) -> Image.Image:
    picture = Image.open(path).convert("RGB")
    picture.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return picture


class StreamParser:
    TOOL_CALL_PREFIX = "<tool_call>"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buffer = ""
        self.past_think = False
        self.mode: Optional[str] = None
        self.emitted_len = 0
        self.yielded_anything = False

    @property
    def has_tool_call(self) -> bool:
        return self.mode == "tool_call" or (self.mode is None and self.TOOL_CALL_PREFIX in self.buffer)

    def visible(self, text: str) -> str:
        shown = replace_special_tokens(text, self.tokenizer)
        if shown:
            self.yielded_anything = True
        return shown

    def feed(self, chunk: str) -> str:
        self.buffer += chunk
        if not self.past_think:
            if "</think>" not in self.buffer:
                return ""
            self.past_think = True

        after = self.buffer.split("</think>", 1)[1]

        if self.mode is None:
            stripped = after.lstrip()
            if not stripped:
                return ""
            if stripped.startswith(self.TOOL_CALL_PREFIX):
                self.mode = "tool_call"
                return ""
            if self.TOOL_CALL_PREFIX.startswith(stripped):
                return ""
            self.mode = "text"
            self.emitted_len = len(after)
            return self.visible(after)

        if self.mode == "text":
            new_part = after[self.emitted_len:]
            self.emitted_len = len(after)
            return self.visible(new_part)
        return ""


class QwenVLModel:
    CHUNK_TIMEOUT_S = 300.0
    STOP_STRINGS = ["</tool_call>"]
    REPETITION_PENALTY = 1.15

    def __init__(
        self,
        model_id: str,
        *,
        processor_path: Optional[str] = None,
        temperature: float = 0.15,
        max_new_tokens: int = 6144,
        reference_images: Optional[list] = None,
        reference_max_side: Optional[int] = 1600,
        load_in_4bit: bool = False,
    ):
        self.model_id = model_id
        self.processor_path = processor_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.reference_images = reference_images
        self.reference_max_side = reference_max_side
        self.load_in_4bit = load_in_4bit

    def load(self):
        return load(self.model_id, self.processor_path, self.load_in_4bit)

    def reference_content(self) -> list[dict]:
        content = [
            {"type": "text", "text": "Reference material (read this before analyzing the case):"}
        ]
        for img_path in self.reference_images:
            image = (
                load_reference_image(str(img_path), self.reference_max_side)
                if self.reference_max_side else str(img_path)
            )
            content.append({"type": "image", "image": image})
        return content

    def build_inputs(self, model, processor, messages: list[ModelMessage], agent_info: AgentInfo):
        chat = extract_chat_messages(messages)

        is_first_turn = not any(isinstance(m, ModelResponse) for m in messages)
        if self.reference_images and is_first_turn:
            chat.insert(0, {"role": "user", "content": self.reference_content()})
            log.info(f"[{self.model_id}] Attached {len(self.reference_images)} reference image(s)")

        img_prsnt = any(
            isinstance(m.get("content"), list)
            and any(c.get("type") == "image" for c in m["content"])
            for m in chat
            if m["role"] == "user"
        )
        log.info(f"[{self.model_id}] Built chat with {len(chat)} turns (image attached: {img_prsnt})")

        tool_instructions = build_tool_instructions(agent_info)
        if tool_instructions:
            chat.append(
                {"role": "system", "content": [{"type": "text", "text": tool_instructions}]}
            )
            log.info(
                f"[{self.model_id}] {len(agent_info.function_tools)} tool(s) available: "
                f"{[t.name for t in agent_info.function_tools]}"
            )

        if agent_info.output_tools:
            schema_hint = json.dumps([t.parameters_json_schema for t in agent_info.output_tools])
            chat.append(
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": (
                            "Respond with ONLY a single JSON object matching this "
                            f"schema, no prose, no markdown fences: {schema_hint}"
                        ),
                    }],
                }
            )

        inputs = processor.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        tokenizer = getattr(processor, "tokenizer", processor)
        return inputs, tokenizer

    def generation_kwargs(self, inputs, tokenizer) -> dict:
        return dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            repetition_penalty=self.REPETITION_PENALTY,
            stop_strings=self.STOP_STRINGS,
            tokenizer=tokenizer
        )

    def generate(self, messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        model, processor = self.load()
        inputs, tokenizer = self.build_inputs(model, processor, messages, agent_info)

        generated_ids = model.generate(**self.generation_kwargs(inputs, tokenizer))
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_text = processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        output_text = strip_special_tokens(strip_thinking(raw_text), tokenizer)
        match = TOOL_CALL_RE.search(output_text)
        if match:
            try:
                call_data = json.loads(match.group(1))
                tool_name = call_data["name"]
                tool_args = call_data.get("arguments", {})
                tool_call_id = str(uuid.uuid4())
                log.info(f"[{self.model_id}] Tool call detected -> {tool_name}({tool_args})")
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool_name, args=tool_args, tool_call_id=tool_call_id)]
                )
            except (json.JSONDecodeError, KeyError) as e:
                log.info(f"[{self.model_id}] Malformed tool_call block, treating as plain text: {e}")

        log.info(f"[{self.model_id}] Final text answer ({len(output_text)} chars), no tool call")
        return ModelResponse(parts=[TextPart(content=output_text)])

    @staticmethod
    def run_generation(model, gen_kwargs: dict, streamer, gen_error: list[BaseException]) -> None:
        try:
            model.generate(**gen_kwargs)
        except BaseException as e:
            gen_error.append(e)
        finally:
            streamer.end()

    async def read_chunks(self, streamer, gen_error: list[BaseException]):
        sentinel = object()
        while True:
            try:
                chunk = await asyncio.wait_for(
                    asyncio.to_thread(next, streamer, sentinel), timeout=self.CHUNK_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                log.warning(
                    f"[{self.model_id}] (stream) No token received for {self.CHUNK_TIMEOUT_S:.0f}s — "
                    "treating generation as stalled."
                )
                gen_error.append(TimeoutError(f"Generation stalled for over {self.CHUNK_TIMEOUT_S:.0f}s"))
                return
            if chunk is sentinel:
                return
            yield chunk

    def tool_call_delta(self, final_text: str) -> Optional[dict]:
        match = TOOL_CALL_RE.search(final_text)
        if not match:
            return None
        try:
            call_data = json.loads(match.group(1))
            tool_call_id = str(uuid.uuid4())
            log.info(f"[{self.model_id}] (stream) Tool call detected -> {call_data.get('name')}")
            return {
                0: DeltaToolCall(
                    name=call_data["name"],
                    json_args=json.dumps(call_data.get("arguments", {})),
                    tool_call_id=tool_call_id
                )
            }
        except (json.JSONDecodeError, KeyError) as e:
            log.info(f"[{self.model_id}] (stream) Malformed tool_call block, falling back to text: {e}")
            return None

    def failure_text(self, parser: StreamParser, gen_error: list[BaseException]) -> str:
        if gen_error:
            log.warning(f"[{self.model_id}] (stream) Generation failed: {gen_error[0]!r}")
            return (
                "FAILURE: Generation failed before producing an answer "
                f"({type(gen_error[0]).__name__}: {gen_error[0]}). Try again."
            )
        if not parser.past_think:
            log.warning(
                f"[{self.model_id}] (stream) Hit max_new_tokens ({self.max_new_tokens}) "
                "while still inside <think>;no final answer was produced."
            )
            return (
                "FAILURE: The model ran out of its generation budget while still "
                "reasoning and never produced a final answer. Try again, or "
                "increase `max_new_tokens` for this agent."
            )
        log.warning(
            f"[{self.model_id}] (stream) Reached </think> but produced no "
            "usable final-answer text before generation ended."
        )
        return (
            "FAILURE: The model finished reasoning but generation ended "
            "before it wrote a final answer. Try again, or increase "
            "`max_new_tokens` for this agent."
        )

    async def stream(self, messages: list[ModelMessage], agent_info: AgentInfo):
        try:
            model, processor = self.load()
            inputs, tokenizer = self.build_inputs(model, processor, messages, agent_info)

            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=False
            )
            gen_kwargs = {**self.generation_kwargs(inputs, tokenizer), "streamer": streamer}
            gen_error: list[BaseException] = []
            gen_thread = threading.Thread(
                target=self.run_generation, args=(model, gen_kwargs, streamer, gen_error), daemon=True
            )
            gen_thread.start()

            parser = StreamParser(tokenizer)
            async with aclosing(self.read_chunks(streamer, gen_error)) as chunks:
                async for chunk in chunks:
                    visible = parser.feed(chunk)
                    if visible:
                        yield visible

            gen_thread.join(timeout=10.0)
            if gen_thread.is_alive():
                log.warning(
                    f"[{self.model_id}] (stream) Generation thread still alive "
                    "10s after stall/completion was detected; abandoning "
                    "it and returning the fallback response."
                )

            if parser.has_tool_call:
                final_text = strip_special_tokens(strip_thinking(parser.buffer), tokenizer)
                delta = self.tool_call_delta(final_text)
                if delta is not None:
                    yield delta
                    return
                if final_text:
                    parser.yielded_anything = True
                    yield final_text

            if not parser.yielded_anything:
                yield self.failure_text(parser, gen_error)

            log.info(f"[{self.model_id}] (stream) Done ({len(parser.buffer)} raw chars)")

        except Exception as e:
            log.exception(f"[{self.model_id}] (stream) Unhandled error in stream_run")
            yield f"FAILURE: Internal error while generating this agent's response: {type(e).__name__}: {e}._"

    def function_model(self) -> FunctionModel:
        return FunctionModel(self.generate, stream_function=self.stream, model_name=self.model_id)
