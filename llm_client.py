"""OpenAI-compatible chat completions client."""

import json
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    """Raised when the LLM endpoint cannot return a usable response."""


class OpenAICompatibleClient:
    def __init__(self, base_model: str, base_url: str, api_key: str,
                 timeout: float = 30.0):
        self.base_model = base_model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.base_model)

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        if not self.is_configured:
            raise LLMError("LLM is not configured; API_KEY is empty.")

        payload = {
            "model": self.base_model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
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
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM response is not valid JSON: {data}") from exc
