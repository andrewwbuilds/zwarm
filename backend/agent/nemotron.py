"""Talks to NVIDIA's OpenAI-compatible chat endpoint.

Blocking and slow, so it never runs on the flight loop. It retries on 429 and 5xx, walks down a chain
of models, and gives up within a time budget so the operator is never left waiting on a hung model.
The API key comes from the environment or backend/.env and must never reach a log or an error message.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
PLANNER_MODELS = (
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-nano-3-30b-a3b",
)
VISION_MODELS = (
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
)
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

# Measured live: with thinking on, Ultra often asks instead of calling a tool and takes 2 to 8 s. With it
# off the same sentence gives the tool call in under 1 s. See agent/eval.py for the numbers.
TEMPLATE_KWARGS = {"nvidia/nemotron-3-ultra-550b-a55b": {"enable_thinking": False}}

TIMEOUT_S = 20.0  # one request
BUDGET_S = 40.0  # one complete() call, across every retry and model
RETRIES = 2  # extra tries on the same model after a 429 or 5xx
BACKOFF_S = 0.6
RATE_PER_MINUTE = 36  # NVIDIA allows about 40 a minute per model
DEAD_MODEL_S = 600.0  # a model that answered 404 or 403 isn't tried again for this long
OFFLINE_S = 20.0  # after every model failed, fail at once for this long so commands stay fast

_THINK = re.compile(r"<think>.*?(?:</think>|$)|^.*?</think>", re.DOTALL | re.IGNORECASE)
_INLINE_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class NemotronError(RuntimeError):
    """The model couldn't be reached. The message is safe to show: it never holds the key."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = "{}"
    problem: str | None = None  # the model's arguments weren't valid JSON


@dataclass
class Reply:
    model: str
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    elapsed_s: float = 0.0

    @property
    def text(self) -> str:
        """The answer, or for models that answer inside their reasoning, that."""
        return self.content or self.reasoning

    def assistant_message(self) -> dict:
        """This reply as it goes back into the conversation. Reasoning stays out."""
        message: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [{"id": c.id, "type": "function",
                                      "function": {"name": c.name, "arguments": c.raw_arguments}}
                                     for c in self.tool_calls]
        return message


def _flat(value) -> str:
    """Content can be a string, null, or a list of typed parts."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part.get("text") or "" if isinstance(part, dict) else str(part) for part in value)
    return str(value)


def _tool_call(raw: dict, index: int) -> ToolCall:
    function = raw.get("function") or {}
    name = str(function.get("name") or raw.get("name") or "")
    arguments = function.get("arguments", raw.get("arguments"))
    call_id = str(raw.get("id") or f"call-{index}")
    if isinstance(arguments, dict):
        return ToolCall(call_id, name, arguments, json.dumps(arguments))
    text = _flat(arguments).strip() or "{}"
    try:
        parsed = json.loads(text)
    except ValueError:
        return ToolCall(call_id, name, {}, "{}", "The arguments weren't valid JSON.")
    if not isinstance(parsed, dict):
        return ToolCall(call_id, name, {}, "{}", "The arguments must be a JSON object.")
    return ToolCall(call_id, name, parsed, text)


def parse_reply(data: dict, model: str, elapsed_s: float = 0.0) -> Reply:
    """Read one chat completion. Every field may be missing or null, so nothing here is assumed."""
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("no choices in the reply")
    message = choices[0].get("message") or {}
    reasoning = _flat(message.get("reasoning_content") or message.get("reasoning")).strip()
    content = _THINK.sub("", _flat(message.get("content"))).strip()
    calls = [_tool_call(raw, i) for i, raw in enumerate(message.get("tool_calls") or []) if isinstance(raw, dict)]
    if not calls and "<tool_call>" in content:
        # Some models write the call into the text instead of the tool_calls field.
        for i, found in enumerate(_INLINE_CALL.findall(content)):
            try:
                calls.append(_tool_call(json.loads(found), i))
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
        content = _INLINE_CALL.sub("", content).strip()
    return Reply(model, content, reasoning, calls, choices[0].get("finish_reason"), elapsed_s)


def _chain(env_name: str, defaults: tuple[str, ...]) -> list[str]:
    """A model named in the environment goes first; the rest stay as fallbacks."""
    named = [m.strip() for m in os.environ.get(env_name, "").split(",") if m.strip()]
    return list(dict.fromkeys([*named, *defaults]))


def _send(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(url, body, headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read()


class _Pace:
    """A sliding one-minute window per model, shared by every thread using this client."""

    def __init__(self, per_minute: int, clock):
        self.per_minute, self.clock = per_minute, clock
        self.lock = threading.Lock()
        self.sent: dict[str, deque] = {}

    def wait_s(self, model: str) -> float:
        """Take a slot and return 0, or return how long until one frees up."""
        with self.lock:
            now = self.clock()
            sent = self.sent.setdefault(model, deque())
            while sent and now - sent[0] >= 60.0:
                sent.popleft()
            if len(sent) < self.per_minute:
                sent.append(now)
                return 0.0
            return 60.0 - (now - sent[0]) + 0.05


class NemotronClient:
    def __init__(self, api_key: str | None = None, models=None, vision_models=None, url: str = URL,
                 timeout_s: float = TIMEOUT_S, budget_s: float = BUDGET_S, retries: int = RETRIES,
                 per_minute: int = RATE_PER_MINUTE, template_kwargs: dict | None = None, send=_send,
                 sleep=time.sleep, clock=time.monotonic):
        self._key = api_key
        self.models = list(models) if models else _chain("NEMOTRON_MODEL", PLANNER_MODELS)
        self.vision_models = list(vision_models) if vision_models else _chain("NEMOTRON_VISION_MODEL", VISION_MODELS)
        self.url, self.timeout_s, self.budget_s, self.retries = url, timeout_s, budget_s, retries
        self.template_kwargs = template_kwargs  # replaces TEMPLATE_KWARGS for every model when set
        self._send, self._sleep, self._clock = send, sleep, clock
        self._pace = _Pace(per_minute, clock)
        self._lock = threading.Lock()
        self._dead: dict[str, float] = {}
        self._offline_until = 0.0
        self.online: bool | None = None  # None until the first call
        self.last_model: str | None = None
        self.last_error: str | None = None
        self.requests = 0

    @property
    def model(self) -> str:
        """The model answering now, or the one that would be tried first."""
        return self.last_model or self.models[0]

    def _api_key(self) -> str:
        if self._key:
            return self._key
        key = os.environ.get("NVIDIA_API_KEY")
        if not key and ENV_FILE.exists():
            try:
                from dotenv import dotenv_values

                key = dotenv_values(ENV_FILE).get("NVIDIA_API_KEY")
            except Exception:
                logger.warning("[agent] couldn't read %s", ENV_FILE.name)
        if not key:
            raise NemotronError("NVIDIA_API_KEY isn't set.")
        self._key = key
        return key

    def _safe(self, text: str) -> str:
        """Scrub an error message, then cut it short."""
        if self._key:
            text = text.replace(self._key, "***")
        return text[:200]

    def complete(self, messages: list[dict], tools: list[dict] | None = None, models=None, tool_choice: str = "auto",
                 max_tokens: int = 1200, temperature: float = 0.1, budget_s: float | None = None) -> Reply:
        """One completion from the first model that gives one. Raises NemotronError if none does."""
        now = self._clock()
        if models is None and now < self._offline_until:
            raise NemotronError(f"The model is offline: {self.last_error or 'no answer'}")
        try:
            key = self._api_key()
        except NemotronError as error:
            with self._lock:
                self.online, self.last_error = False, str(error)
            raise
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"}
        deadline = now + (self.budget_s if budget_s is None else budget_s)
        problems = []
        for model in list(models or self.models):
            if self._clock() >= deadline:
                problems.append("out of time")
                break
            with self._lock:
                if self._clock() < self._dead.get(model, 0.0):
                    continue
            body: dict = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = tool_choice
            template = TEMPLATE_KWARGS.get(model) if self.template_kwargs is None else self.template_kwargs
            if template:
                body["chat_template_kwargs"] = template
            try:
                reply = self._try_model(model, headers, json.dumps(body).encode(), deadline)
            except _ModelFailed as failure:
                problems.append(f"{model.split('/')[-1]}: {failure}")
                continue
            with self._lock:
                self.online, self.last_model, self.last_error = True, model, None
            return reply
        message = self._safe("; ".join(problems) or "every model is unavailable")
        with self._lock:
            self.online, self.last_error = False, message
            if models is None:
                self._offline_until = self._clock() + OFFLINE_S
        logger.warning("[agent] no model answered: %s", message)
        raise NemotronError(f"No model answered: {message}")

    def _try_model(self, model: str, headers: dict, body: bytes, deadline: float) -> Reply:
        last = "no attempt"
        for attempt in range(self.retries + 1):
            wait = self._pace.wait_s(model)
            while wait > 0:
                if self._clock() + wait >= deadline:
                    raise _ModelFailed("rate limit, out of time")
                self._sleep(wait)
                wait = self._pace.wait_s(model)
            left = deadline - self._clock()
            if left <= 0:
                raise _ModelFailed("out of time")
            started = self._clock()
            self.requests += 1
            try:
                status, response_headers, raw = self._send(self.url, headers, body, min(self.timeout_s, left))
            except Exception as error:  # a timeout or a network failure: this model won't answer in time
                raise _ModelFailed(self._safe(type(error).__name__)) from None
            elapsed = self._clock() - started
            if 200 <= status < 300:
                try:
                    return parse_reply(json.loads(raw), model, elapsed)
                except ValueError as error:
                    raise _ModelFailed(self._safe(f"unreadable reply: {error}")) from None
            last = f"HTTP {status}"
            if status in (401, 403, 404, 410):
                with self._lock:
                    self._dead[model] = self._clock() + DEAD_MODEL_S
                raise _ModelFailed(last)
            if status != 429 and status < 500:
                raise _ModelFailed(last)
            if attempt < self.retries:
                pause = _retry_after(response_headers) or BACKOFF_S * (2 ** attempt)
                if self._clock() + pause >= deadline:
                    raise _ModelFailed(f"{last}, out of time")
                self._sleep(pause)
        raise _ModelFailed(last)


class _ModelFailed(Exception):
    pass


def _retry_after(headers: dict) -> float | None:
    for name, value in headers.items():
        if name.lower() == "retry-after":
            try:
                return min(max(float(value), 0.0), 10.0)
            except (TypeError, ValueError):
                return None
    return None
