"""A client for llama-server's `POST /v1/systemone` (a decision model answers typed questions about a text). Pure, stdlib only.

Shapes are those of a real llama-server run (tests/fixtures/systemone/, recorded with Julia-1 Q8_0) and of the server docs
(tools/server/README.md, "TypeSafe-compatible System One API"): the request is {"state", "questions": {id: {"type", "instructions",
"criteria"}}}, a `choice` answer is {"type": "choice", "choice", "probabilities", "confidence"}. No generation happens (one forward
pass), the data never leaves the machine: the server is a local llama-server.

The network is injected: `post(url, payload, timeout) -> (status, body text)`; it raises OSError (refused, timeout, ...) when no
answer came. Every failure is a JudgeError whose message names the URL.
"""

import json
import urllib.error
import urllib.request
from typing import NamedTuple

ENDPOINT = "/v1/systemone"
CHECK_QUESTION = "Is this text a greeting?"  # any noul question: it only has to be answered
QUESTION_ID = "q"


class JudgeError(Exception):
    """The judge server could not answer; the message says why and names the URL."""


class ChoiceAnswer(NamedTuple):
    choice: str
    probabilities: dict
    confidence: float

    @property
    def margin(self):
        """Top probability minus the second (the server's `confidence` is not documented, this is ours)."""
        top = sorted(self.probabilities.values(), reverse=True)
        return top[0] - top[1] if len(top) > 1 else top[0]


def _urllib_post(url, payload, timeout):
    request = urllib.request.Request(
        url, json.dumps(payload, ensure_ascii=False).encode("utf-8"), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:  # an answer with an error status
        return error.code, error.read().decode("utf-8", "replace")
    except urllib.error.URLError as error:  # no answer: refused, unreachable
        raise OSError(str(error.reason)) from None


class SystemOneClient:
    def __init__(self, base_url, model=None, timeout=60.0, post=None):
        self.url = base_url.rstrip("/") + ENDPOINT
        self.model = model
        self.timeout = timeout
        self._post = post or _urllib_post

    def _ask(self, state, question):
        """Send one question; returns (model, its answer dict)."""
        payload = {"state": state, "questions": {QUESTION_ID: question}}
        if self.model:
            payload["model"] = self.model
        try:
            status, text = self._post(self.url, payload, self.timeout)
        except ConnectionRefusedError:
            raise JudgeError(f"{self.url}: connection refused: is the server running on that URL?") from None
        except OSError as error:
            raise JudgeError(f"{self.url}: no answer ({error or type(error).__name__}): is the server running on that URL?") from None
        if status != 200:
            raise JudgeError(f"{self.url}: {self._status_message(status, text)}")
        try:
            body = json.loads(text)
            answer = body["answers"][QUESTION_ID]
        except (ValueError, KeyError, TypeError):
            raise JudgeError(f"{self.url}: the answer has no `answers.{QUESTION_ID}`: {text[:200]!r}") from None
        if not isinstance(answer, dict):
            raise JudgeError(f"{self.url}: the answer is not an object: {text[:200]!r}")
        return body.get("model"), answer

    @staticmethod
    def _status_message(status, text):
        try:
            detail = json.loads(text)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            detail = text[:200]
        if status == 404:
            return f"HTTP 404, this llama.cpp has no {ENDPOINT} (too old: `llama update`, then restart the server)"
        if status == 501:
            return f"HTTP 501, the loaded model is not a decision model ({detail})"
        return f"HTTP {status}: {detail}"

    def choice(self, state, instructions, options):
        """One `choice` question: which of `options` ({option: description}) fits `state` (a string or an object)?"""
        _, answer = self._ask(state, {"type": "choice", "instructions": instructions, "criteria": options})
        if answer.get("type") != "choice":
            raise JudgeError(f"{self.url}: expected a choice answer, got {answer.get('type')!r}")
        chosen, probabilities, confidence = answer.get("choice"), answer.get("probabilities"), answer.get("confidence")
        if chosen not in options:
            raise JudgeError(f"{self.url}: the choice {chosen!r} is not one of the options")
        if not isinstance(probabilities, dict) or not all(
            isinstance(p, (int, float)) and not isinstance(p, bool) for p in probabilities.values()
        ):
            raise JudgeError(f"{self.url}: `probabilities` is not a map of numbers")
        return ChoiceAnswer(chosen, probabilities, confidence if isinstance(confidence, (int, float)) else 0.0)

    def check_server(self):
        """Ask one trivial yes/no question; returns the model the server answers with."""
        model, _ = self._ask("hello", {"type": "noul", "instructions": CHECK_QUESTION})
        return model
