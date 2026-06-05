from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class ChatModel(Protocol):
    async def chat(self, messages: list[dict[str, str]]) -> str:
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


@dataclass(frozen=True)
class OpenAICompatibleModel:
    base_url: str
    model: str
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: int = 120

    async def chat(self, messages: list[dict[str, str]]) -> str:
        headers = {}
        if self.api_key_env:
            headers["Authorization"] = f"Bearer {os.getenv(self.api_key_env, 'local')}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = await asyncio.to_thread(
            _post_json,
            f"{self.base_url.rstrip('/')}/chat/completions",
            payload,
            headers,
            self.timeout_seconds,
        )
        return data["choices"][0]["message"].get("content") or ""


@dataclass(frozen=True)
class OllamaNativeModel:
    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    num_ctx: int | None = None
    think: bool = False
    timeout_seconds: int = 120

    async def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if self.num_ctx:
            payload["options"]["num_ctx"] = self.num_ctx
        data = await asyncio.to_thread(
            _post_json,
            f"{self.base_url.rstrip('/')}/api/chat",
            payload,
            {},
            self.timeout_seconds,
        )
        return data["message"].get("content") or ""


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
            )
        raise ValueError(f"Unsupported provider kind: {spec['kind']}")
