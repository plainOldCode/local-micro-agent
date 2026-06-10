from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Protocol


StreamChunk = str | dict[str, str]
StreamCallback = Callable[[StreamChunk], None]


@dataclass(frozen=True)
class ModelResponse:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)


class ChatModel(Protocol):
    async def chat(
        self,
        messages: list[dict[str, str]],
        stream_callback: StreamCallback | None = None,
    ) -> str | ModelResponse:
        ...


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _ollama_usage(data: dict[str, Any]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    if isinstance(prompt_tokens, int):
        usage["prompt_tokens"] = prompt_tokens
        usage["provider_prompt_eval_count"] = prompt_tokens
    if isinstance(completion_tokens, int):
        usage["completion_tokens"] = completion_tokens
        usage["provider_eval_count"] = completion_tokens
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        usage["total_tokens"] = prompt_tokens + completion_tokens
    for source_key, dest_key in (
        ("prompt_eval_duration", "provider_prompt_eval_duration_ns"),
        ("eval_duration", "provider_eval_duration_ns"),
        ("total_duration", "provider_total_duration_ns"),
    ):
        value = data.get(source_key)
        if isinstance(value, int):
            usage[dest_key] = value
    return usage


def _openai_usage(data: dict[str, Any]) -> dict[str, Any]:
    raw_usage = data.get("usage")
    if not isinstance(raw_usage, dict):
        return {}
    usage: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw_usage.get(key)
        if isinstance(value, int):
            usage[key] = value
    return usage


def _openai_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    return message.get("content") or ""


def _merge_openai_payload(base: dict[str, Any], extra_body: dict[str, Any] | None) -> dict[str, Any]:
    if not extra_body:
        return base
    merged = dict(base)
    merged.update(extra_body)
    return merged


def _openai_stream_payload(payload: dict[str, Any]) -> dict[str, Any]:
    stream_payload = {**payload, "stream": True}
    options = dict(stream_payload.get("stream_options") or {})
    options.setdefault("include_usage", True)
    stream_payload["stream_options"] = options
    return stream_payload


def _apply_assistant_think_prefill(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [*messages, {"role": "assistant", "content": "<think>\n\n</think>\n\n"}]


def _post_openai_stream(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    stream_callback: StreamCallback | None,
) -> ModelResponse:
    stream_payload = _openai_stream_payload(payload)
    body = json.dumps(stream_payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    done_data: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_text = line.removeprefix("data:").strip()
                if data_text == "[DONE]":
                    break
                data = json.loads(data_text)
                done_data = data
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta")
                if not isinstance(delta, dict):
                    continue
                reasoning_chunk = delta.get("reasoning_content") or ""
                if reasoning_chunk:
                    reasoning_chunks.append(reasoning_chunk)
                    if stream_callback is not None:
                        stream_callback(reasoning_chunk)
                chunk = delta.get("content") or ""
                if chunk:
                    chunks.append(chunk)
                    if stream_callback is not None:
                        stream_callback(chunk)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    usage = _openai_usage(done_data)
    if reasoning_chunks:
        usage["reasoning_content_chars"] = len("".join(reasoning_chunks))
    return ModelResponse("".join(chunks), usage=usage)


def _post_ollama_stream(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: int,
    stream_callback: StreamCallback | None,
) -> ModelResponse:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    done_data: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                data = json.loads(line)
                message = data.get("message")
                if isinstance(message, dict):
                    reasoning_chunk = (
                        message.get("thinking") or message.get("reasoning_content") or ""
                    )
                    if reasoning_chunk:
                        reasoning_chunks.append(reasoning_chunk)
                        if stream_callback is not None:
                            stream_callback({"kind": "reasoning", "content": reasoning_chunk})
                    chunk = message.get("content") or ""
                    if chunk:
                        chunks.append(chunk)
                        if stream_callback is not None:
                            stream_callback(chunk)
                if data.get("done"):
                    done_data = data
                    break
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    usage = _ollama_usage(done_data)
    if reasoning_chunks:
        usage["reasoning_content_chars"] = len("".join(reasoning_chunks))
        usage["reasoning_only_response"] = not bool(chunks)
    return ModelResponse("".join(chunks), usage=usage)


@dataclass(frozen=True)
class OpenAICompatibleModel:
    supports_streaming: ClassVar[bool] = True

    base_url: str
    model: str
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: int = 120
    think: bool | None = None
    disable_thinking_with_assistant_prefill: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    extra_options: dict[str, Any] = field(default_factory=dict)

    async def chat(
        self,
        messages: list[dict[str, str]],
        stream_callback: StreamCallback | None = None,
    ) -> ModelResponse:
        headers = {}
        if self.api_key_env:
            headers["Authorization"] = f"Bearer {os.getenv(self.api_key_env, 'local')}"
        request_messages = messages
        if self.think is False and self.disable_thinking_with_assistant_prefill:
            request_messages = _apply_assistant_think_prefill(messages)
        payload = {
            "model": self.model,
            "messages": request_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        payload.update(self.extra_options)
        if self.think is not None:
            payload["think"] = self.think
            payload["enable_thinking"] = self.think
            payload["enableThinking"] = self.think
        payload = _merge_openai_payload(payload, self.extra_body)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        if stream_callback is not None:
            return await asyncio.to_thread(
                _post_openai_stream,
                url,
                payload,
                headers,
                self.timeout_seconds,
                stream_callback,
            )
        data = await asyncio.to_thread(
            _post_json,
            url,
            payload,
            headers,
            self.timeout_seconds,
        )
        content = _openai_chat_content(data)
        return ModelResponse(content, usage=_openai_usage(data))


@dataclass(frozen=True)
class OllamaNativeModel:
    supports_streaming: ClassVar[bool] = True

    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    num_ctx: int | None = None
    think: bool = False
    timeout_seconds: int = 120
    extra_options: dict[str, Any] = field(default_factory=dict)

    async def chat(
        self,
        messages: list[dict[str, str]],
        stream_callback: StreamCallback | None = None,
    ) -> ModelResponse:
        options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            **self.extra_options,
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream_callback is not None,
            "think": self.think,
            "options": options,
        }
        if self.num_ctx:
            payload["options"]["num_ctx"] = self.num_ctx
        url = f"{self.base_url.rstrip('/')}/api/chat"
        if stream_callback is not None:
            return await asyncio.to_thread(
                _post_ollama_stream,
                url,
                payload,
                {},
                self.timeout_seconds,
                stream_callback,
            )
        data = await asyncio.to_thread(_post_json, url, payload, {}, self.timeout_seconds)
        message = data.get("message") or {}
        content = message.get("content") or ""
        usage = _ollama_usage(data)
        reasoning = message.get("thinking") or message.get("reasoning_content") or ""
        if reasoning:
            usage["reasoning_content_chars"] = len(reasoning)
            usage["reasoning_only_response"] = not bool(content)
        return ModelResponse(content, usage=usage)


class ModelManager:
    def __init__(self, config: dict):
        self.config = config
        self._models: dict[str, ChatModel] = {}

    def get(self, role: str) -> ChatModel:
        model_name = self.config["models"].get(role) or self.config["models"]["default"]
        if model_name not in self._models:
            self._models[model_name] = self._build(model_name)
        return self._models[model_name]

    def _build(self, name: str) -> ChatModel:
        spec = self.config["providers"][name]
        if spec["kind"] == "openai_compatible":
            return OpenAICompatibleModel(
                base_url=spec["base_url"],
                model=spec["model"],
                api_key_env=spec.get("api_key_env"),
                temperature=spec.get("temperature", 0.0),
                max_tokens=spec.get("max_tokens", 2048),
                timeout_seconds=spec.get("timeout_seconds", 120),
                think=spec.get("think"),
                disable_thinking_with_assistant_prefill=spec.get(
                    "disable_thinking_with_assistant_prefill", False
                ),
                extra_body=spec.get("extra_body") or {},
                extra_options=spec.get("extra_options") or {},
            )
        if spec["kind"] == "ollama_native":
            return OllamaNativeModel(
                base_url=spec["base_url"],
                model=spec["model"],
                temperature=spec.get("temperature", 0.0),
                max_tokens=spec.get("max_tokens", 2048),
                num_ctx=spec.get("num_ctx"),
                think=spec.get("think", False),
                timeout_seconds=spec.get("timeout_seconds", 120),
                extra_options=spec.get("extra_options") or {},
            )
        raise ValueError(f"Unsupported provider kind: {spec['kind']}")
