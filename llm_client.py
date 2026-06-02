"""OpenAI-compatible chat completions client."""

import json
import re
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
            "stream": False,
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
            return self._parse_json_content(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM response is not valid JSON: {data}") from exc

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
