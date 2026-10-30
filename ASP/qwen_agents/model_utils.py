from __future__ import annotations
import env_defaults 
import torch
import asyncio
import json
import logging
import os
import re
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional
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
        log.info("HF_HUB_OFFLINE is set — skipping huggingface_hub login() (relying on locally cached model weights).")
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
            f"Pass processor_path (e.g. 'Qwen/Qwen3-VL-8B-Thinking') to make_model()."
        )
    return model, processor


@lru_cache(maxsize=4)
def load(model_id: str, processor_path: Optional[str], load_in_4bit: bool):
    return load_with_transformers(download(model_id), processor_path, load_in_4bit)


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
                    tool_text = (
                        f"Tool '{part.tool_name}' returned:\n"
                        f"{part.model_response_str()}"
                    )
                    user_content_items.append({"type": "text", "text": tool_text})
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


def build_chat_and_inputs(
    model,
    processor,
    model_id: str,
    messages: list[ModelMessage],
    agent_info: AgentInfo,
    reference_images: Optional[list],
):
    chat = extract_chat_messages(messages)

    is_first_turn = not any(isinstance(m, ModelResponse) for m in messages)
    if reference_images and is_first_turn:
        ref_content = [
            {"type": "text", "text": "Reference material (read this before analyzing the case):"}
        ]
        for img_path in reference_images:
            ref_content.append({"type": "image", "image": str(img_path)})
        chat.insert(0, {"role": "user", "content": ref_content})
        log.info(f"[{model_id}] Attached {len(reference_images)} reference image(s)")

    img_prsnt = any(
        isinstance(m.get("content"), list)
        and any(c.get("type") == "image" for c in m["content"])
        for m in chat
        if m["role"] == "user"
    )
    log.info(f"[{model_id}] Built chat with {len(chat)} turns (image attached: {img_prsnt})")

    tool_instructions = build_tool_instructions(agent_info)
    if tool_instructions:
        chat.append(
            {"role": "system", "content": [{"type": "text", "text": tool_instructions}]}
        )
        log.info(
            f"[{model_id}] {len(agent_info.function_tools)} tool(s) available: "
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


def make_model(
    model_id: str,
    *,
    processor_path: Optional[str] = None,
    temperature: float = 0.15,
    max_new_tokens: int = 6144,
    reference_images: Optional[list] = None,
    load_in_4bit: bool = False,
) -> FunctionModel:
    def out_kwargs(inputs, tokenizer) -> dict:
        return dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            repetition_penalty=1.15,
            stop_strings=["</tool_call>"],
            tokenizer=tokenizer
        )

    def run(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        model, processor = load(model_id, processor_path, load_in_4bit)
        inputs, tokenizer = build_chat_and_inputs(
            model, processor, model_id, messages, agent_info, reference_images
        )

        generated_ids = model.generate(**out_kwargs(inputs, tokenizer))
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
                log.info(f"[{model_id}] Tool call detected -> {tool_name}({tool_args})")
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool_name, args=tool_args, tool_call_id=tool_call_id)]
                )
            except (json.JSONDecodeError, KeyError) as e:
                log.info(f"[{model_id}] Malformed tool_call block, treating as plain text: {e}")

        log.info(f"[{model_id}] Final text answer ({len(output_text)} chars), no tool call")
        return ModelResponse(parts=[TextPart(content=output_text)])

    async def stream_run(messages: list[ModelMessage], agent_info: AgentInfo):
        buffer = ""
        past_think = False
        try:
            model, processor = load(model_id, processor_path, load_in_4bit)
            inputs, tokenizer = build_chat_and_inputs(
                model, processor, model_id, messages, agent_info, reference_images
            )

            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=False
            )
            gen_kwargs = {**out_kwargs(inputs, tokenizer), "streamer": streamer}
            gen_error: list[BaseException] = []

            def generate():
                try:
                    model.generate(**gen_kwargs)
                except BaseException as e:
                    gen_error.append(e)
                finally:
                    streamer.end()

            gen_thread = threading.Thread(target=generate, daemon=True)
            gen_thread.start()

            mode: Optional[str] = None   
            already_yielded_len = 0      
            yielded_anything = False     
            TOOL_CALL_PREFIX = "<tool_call>"
            sentinel = object()
            CHUNK_TIMEOUT_S = 300.0

            while True:
                try:
                    chunk = await asyncio.wait_for(
                        asyncio.to_thread(next, streamer, sentinel), timeout=CHUNK_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        f"[{model_id}] (stream) No token received for {CHUNK_TIMEOUT_S:.0f}s — "
                        "treating generation as stalled."
                    )
                    gen_error.append(TimeoutError(f"Generation stalled for over {CHUNK_TIMEOUT_S:.0f}s"))
                    break
                if chunk is sentinel:
                    break
                buffer += chunk

                if not past_think:
                    if "</think>" not in buffer:
                        continue  
                    past_think = True

                after = buffer.split("</think>", 1)[1]

                if mode is None:
                    stripped = after.lstrip()
                    if not stripped:
                        continue 
                    if stripped.startswith(TOOL_CALL_PREFIX):
                        mode = "tool_call"
                        continue 
                    if TOOL_CALL_PREFIX.startswith(stripped):
                        continue 
                    mode = "text"
                    already_yielded_len = len(after)
                    visible = replace_special_tokens(after, tokenizer)
                    if visible:
                        yielded_anything = True
                        yield visible
                    continue

                if mode == "text":
                    new_part = after[already_yielded_len:]
                    already_yielded_len = len(after)
                    visible = replace_special_tokens(new_part, tokenizer)
                    if visible:
                        yielded_anything = True
                        yield visible

            gen_thread.join(timeout=10.0)
            if gen_thread.is_alive():
                log.warning(
                    f"[{model_id}] (stream) Generation thread still alive "
                    "10s after stall/completion was detected — abandoning "
                    "it and returning the fallback response."
                )

            if mode == "tool_call" or (mode is None and "<tool_call>" in buffer):
                final_text = strip_special_tokens(strip_thinking(buffer), tokenizer)
                match = TOOL_CALL_RE.search(final_text)
                if match:
                    try:
                        call_data = json.loads(match.group(1))
                        tool_call_id = str(uuid.uuid4())
                        log.info(f"[{model_id}] (stream) Tool call detected -> {call_data.get('name')}")
                        yield {
                            0: DeltaToolCall(
                                name=call_data["name"],
                                json_args=json.dumps(call_data.get("arguments", {})),
                                tool_call_id=tool_call_id
                            )
                        }
                        return
                    except (json.JSONDecodeError, KeyError) as e:
                        log.info(f"[{model_id}] (stream) Malformed tool_call block, falling back to text: {e}")

                if final_text:
                    yielded_anything = True
                    yield final_text

            if not yielded_anything:
                if gen_error:
                    log.warning(f"[{model_id}] (stream) Generation failed: {gen_error[0]!r}")
                    yield (
                        "FAILURE: Generation failed before producing an answer "
                        f"({type(gen_error[0]).__name__}: {gen_error[0]}). Try again."
                    )
                elif not past_think:
                    log.warning(
                        f"[{model_id}] (stream) Hit max_new_tokens ({max_new_tokens}) "
                        "while still inside <think> — no final answer was produced."
                    )
                    yield (
                        "FAILURE: The model ran out of its generation budget while still "
                        "reasoning and never produced a final answer. Try again, or "
                        "increase `max_new_tokens` for this agent."
                    )
                else:
                    log.warning(
                        f"[{model_id}] (stream) Reached </think> but produced no "
                        "usable final-answer text before generation ended."
                    )
                    yield (
                        "FAILURE: The model finished reasoning but generation ended "
                        "before it wrote a final answer. Try again, or increase "
                        "`max_new_tokens` for this agent."
                    )

            log.info(f"[{model_id}] (stream) Done ({len(buffer)} raw chars)")

        except Exception as e:
            log.exception(f"[{model_id}] (stream) Unhandled error in stream_run")
            yield f"FAILURE: Internal error while generating this agent's response: {type(e).__name__}: {e}._"

    return FunctionModel(run, stream_function=stream_run, model_name=model_id)
