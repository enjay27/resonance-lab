"""A stand-in for llama-server's /v1/systemone, injected as `post` into judge_client.SystemOneClient (no network in tests)."""

import json

from judge_client import QUESTION_ID

MODEL = "models/Stub-Q8_0.gguf"


class StubJudge:
    """`answers` maps a message to {option: probability}; a message not in it gets the first option at 1.0.
    `fail` is a set of messages whose request fails (HTTP 500); `calls` records every request payload."""

    def __init__(self, answers=None, fail=(), model=MODEL, down=False):
        self.answers, self.fail, self.model, self.down, self.calls = answers or {}, set(fail), model, down, []

    def __call__(self, url, payload, timeout):
        if self.down:
            raise ConnectionRefusedError()
        self.calls.append(payload)
        question = payload["questions"][QUESTION_ID]
        if question["type"] == "noul":
            return 200, json.dumps({"model": self.model, "answers": {QUESTION_ID: {"type": "noul", "noul": 0.5}}})
        state = payload["state"]
        message = state["message"] if isinstance(state, dict) else state
        if message in self.fail:
            return 500, "boom"
        options = list(question["criteria"])
        probabilities = self.answers.get(message) or {options[0]: 1.0}
        probabilities = {option: probabilities.get(option, 0.0) for option in options}
        choice = max(probabilities, key=probabilities.get)
        top = sorted(probabilities.values(), reverse=True)
        answer = {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": top[0] - top[1]}
        return 200, json.dumps({"model": self.model, "answers": {QUESTION_ID: answer}})

    def states(self):
        """The `state` of every choice request, in order."""
        return [c["state"] for c in self.calls if c["questions"][QUESTION_ID]["type"] == "choice"]
