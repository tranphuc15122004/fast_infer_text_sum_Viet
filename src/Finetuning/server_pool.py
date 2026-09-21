"""Small dependency-free client for SGLang/vLLM OpenAI-compatible servers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import threading
import time
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request


class ServerPoolError(RuntimeError):
    """A request failed on every available generation endpoint."""


@dataclass(frozen=True)
class _Endpoint:
    url: str
    semaphore: threading.BoundedSemaphore


def _chat_completions_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


class OpenAICompatibleServerPool:
    """Dispatch independent chat requests across SGLang or vLLM servers.

    The caller starts one OpenAI-compatible server per B200 (or one server per
    tensor-parallel group).  This client supplies CPU-side concurrency and
    preserves input order when responses complete out of order.
    """

    def __init__(
        self,
        server_urls: Sequence[str],
        *,
        model: str | None = None,
        concurrency_per_server: int = 8,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.25,
    ) -> None:
        urls = [str(url).strip().rstrip("/") for url in server_urls if str(url).strip()]
        if not urls:
            raise ValueError("at least one generation server URL is required")
        if concurrency_per_server <= 0:
            raise ValueError("concurrency_per_server must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._endpoints = [
            _Endpoint(url, threading.BoundedSemaphore(concurrency_per_server))
            for url in urls
        ]
        self.max_in_flight = len(self._endpoints) * concurrency_per_server

    @property
    def server_urls(self) -> tuple[str, ...]:
        return tuple(endpoint.url for endpoint in self._endpoints)

    def _post_json(self, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib_request.Request(
            _chat_completions_url(endpoint),
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ServerPoolError(
                f"generation server {endpoint} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            raise ServerPoolError(f"generation server {endpoint} is unavailable: {exc}") from exc
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ServerPoolError(f"generation server {endpoint} returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ServerPoolError(f"generation server {endpoint} returned a non-object response")
        return value

    @staticmethod
    def _content(response: Mapping[str, Any], endpoint: str) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise ServerPoolError(f"generation server {endpoint} returned no choices")
        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            raise ServerPoolError(f"generation server {endpoint} returned no message")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            ]
            return "".join(parts)
        raise ServerPoolError(f"generation server {endpoint} returned non-text content")

    def _generate_one(self, index: int, payload: Mapping[str, object]) -> str:
        last_error: Exception | None = None
        request_payload = dict(payload)
        if self.model and "model" not in request_payload:
            request_payload["model"] = self.model
        for attempt in range(self.max_retries + 1):
            endpoint = self._endpoints[(index + attempt) % len(self._endpoints)]
            try:
                with endpoint.semaphore:
                    response = self._post_json(endpoint.url, request_payload)
                return self._content(response, endpoint.url)
            except Exception as exc:  # endpoint failover must include parser errors
                last_error = exc
                if attempt < self.max_retries and self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise ServerPoolError(
            f"generation request {index} failed after {self.max_retries + 1} attempts: "
            f"{last_error}"
        ) from last_error

    def generate_many(self, payloads: Sequence[Mapping[str, object]]) -> list[str]:
        """Generate a batch and return summaries in exactly input order."""

        if not payloads:
            return []
        workers = min(len(payloads), self.max_in_flight)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="target-server") as pool:
            futures = [pool.submit(self._generate_one, index, payload) for index, payload in enumerate(payloads)]
            return [future.result() for future in futures]


__all__ = ["OpenAICompatibleServerPool", "ServerPoolError"]
