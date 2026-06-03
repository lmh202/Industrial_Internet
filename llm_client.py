"""OpenAI-compatible chat completions client."""

import json
import re
import socket
from typing import Optional
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    """Raised when the LLM endpoint cannot return a usable response."""


class OpenAICompatibleClient:
    def __init__(self, base_model: str, base_url: str, api_key: str,
                 timeout: float = 30.0,
                 keep_alive: Optional[str] = None):
        self.base_model = base_model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.keep_alive = keep_alive

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.base_model)

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        if not self.is_configured:
            raise LLMError("LLM is not configured; API_KEY is empty.")

        if self._uses_native_ollama():
            return self._chat_json_native_ollama(system_prompt, user_prompt)

        payload = {
            "model": self.base_model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"/no_think\n{user_prompt}"},
            ],
        }
        if self.keep_alive:
            payload["keep_alive"] = self._native_keep_alive_value()
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
            payload = self._parse_json_content(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM response is not valid JSON: {data}") from exc
        self.preload_model()
        return payload

    def _chat_json_native_ollama(self, system_prompt: str, user_prompt: str) -> dict:
        payload = {
            "model": self.base_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"/no_think\n{user_prompt}"},
            ],
            "format": "json",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 4096,
            },
        }
        if self.keep_alive:
            payload["keep_alive"] = self._native_keep_alive_value()

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._native_ollama_base_url()}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            content = data["message"]["content"]
            return self._parse_json_content(content)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM response is not valid JSON: {data}") from exc

    def preload_model(self) -> None:
        """Refresh Ollama keep_alive through its native API."""
        if not self.keep_alive:
            return
        if not self.is_configured:
            raise LLMError("LLM is not configured; API_KEY is empty.")

        payload = {
            "model": self.base_model,
            "prompt": "",
            "stream": False,
            "keep_alive": self._native_keep_alive_value(),
        }
        self._post_native_generate(payload, "LLM preload request failed")

    def unload_model(self) -> None:
        """Unload the current Ollama model through its native API."""
        if not self.is_configured:
            raise LLMError("LLM is not configured; API_KEY is empty.")

        payload = {
            "model": self.base_model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        self._post_native_generate(payload, "LLM unload request failed")

    def loaded_models(self, timeout: float = 1.0) -> list:
        """Return models currently loaded by Ollama."""
        request = urllib.request.Request(
            f"{self._native_ollama_base_url()}/api/ps",
            method="GET",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM status request failed: {exc}") from exc

        models = data.get("models", [])
        if not isinstance(models, list):
            raise LLMError(f"LLM status response is not valid: {data}")
        return models

    def is_model_loaded(self, timeout: float = 1.0) -> bool:
        """Return True when the configured model is currently loaded."""
        for model in self.loaded_models(timeout=timeout):
            if not isinstance(model, dict):
                continue
            names = {
                str(model.get("name", "")),
                str(model.get("model", "")),
            }
            if self.base_model in names:
                return True
        return False

    def _parse_json_content(self, content: str) -> dict:
        if not isinstance(content, str):
            raise json.JSONDecodeError("content is not a string", "", 0)

        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        cleaned = cleaned.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        candidate = self._extract_first_json_object(cleaned)
        return json.loads(candidate)

    def _extract_first_json_object(self, text: str) -> str:
        start = text.find("{")
        if start < 0:
            raise json.JSONDecodeError("no JSON object found", text, 0)

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]

        raise json.JSONDecodeError("unterminated JSON object", text, start)

    def _native_ollama_base_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3]
        return self.base_url

    def _uses_native_ollama(self) -> bool:
        return "11434" in self.base_url or "ollama" in self.base_url.lower()

    def _native_keep_alive_value(self):
        if isinstance(self.keep_alive, str):
            try:
                return int(self.keep_alive)
            except ValueError:
                return self.keep_alive
        return self.keep_alive

    def _post_native_generate(self, payload: dict, error_message: str) -> None:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._native_ollama_base_url()}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                resp.read()
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise LLMError(f"{error_message}: {exc}") from exc
