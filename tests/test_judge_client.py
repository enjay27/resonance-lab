import json
import os

import pytest

import judge_client
from judge_client import ChoiceAnswer, JudgeError, SystemOneClient

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "systemone")
URL = "http://127.0.0.1:8080"
OPTIONS = {
    "social": "greetings, thanks, small talk",
    "recruitment": "looking for party members, guild members, dungeon runs",
    "question": "asking for help or information",
    "other": "anything else",
}


def recorded(name):
    """A response recorded from a real llama-server (Julia-1 Q8_0): {"status", "request", "body"}."""
    with open(os.path.join(FIXTURES, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def answered(name):
    """The recorded response, its one answer under the id our client asks with (a real server echoes the id it was asked)."""
    body = recorded(name)["body"]
    (answer,) = body["answers"].values()
    body["answers"] = {judge_client.QUESTION_ID: answer}
    return body


class FakeServer:
    """The injected `post(url, payload, timeout) -> (status, body text)`; records the calls."""

    def __init__(self, status=200, body=None, raises=None):
        self.status, self.body, self.raises, self.calls = status, body, raises, []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        if self.raises:
            raise self.raises
        return self.status, self.body if isinstance(self.body, str) else json.dumps(self.body)


def client_for(server, **kwargs):
    return SystemOneClient(URL, post=server, **kwargs)


def test_choice_sends_one_choice_question_to_the_systemone_endpoint():
    server = FakeServer(body=answered("choice"))

    client_for(server).choice("2人募集", "Which category?", OPTIONS)

    (url, payload, _), = server.calls
    assert url == f"{URL}/v1/systemone"
    assert payload["state"] == "2人募集"
    assert list(payload["questions"].values()) == [{"type": "choice", "instructions": "Which category?", "criteria": OPTIONS}]
    assert "model" not in payload


def test_the_model_is_sent_only_when_given():
    server = FakeServer(body=answered("choice"))

    client_for(server, model="Julia-1").choice("x", "Which?", OPTIONS)

    assert server.calls[0][1]["model"] == "Julia-1"


def test_a_trailing_slash_on_the_base_url_is_tolerated():
    server = FakeServer(body=answered("choice"))

    SystemOneClient(URL + "/", post=server).choice("x", "Which?", OPTIONS)

    assert server.calls[0][0] == f"{URL}/v1/systemone"


def test_an_object_state_is_passed_through_for_the_channel_context():
    server = FakeServer(body=answered("choice_object_state"))

    client_for(server).choice({"channel": "party", "message": "2人募集"}, "Which?", OPTIONS)

    assert server.calls[0][1]["state"] == {"channel": "party", "message": "2人募集"}


def test_the_real_choice_response_is_parsed():
    body = answered("choice")
    server = FakeServer(body=body)

    answer = client_for(server).choice("x", "Which?", OPTIONS)

    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "recruitment"
    assert answer.probabilities == body["answers"][judge_client.QUESTION_ID]["probabilities"]
    assert answer.confidence == body["answers"][judge_client.QUESTION_ID]["confidence"]


def test_margin_is_the_top_probability_minus_the_second():
    answer = ChoiceAnswer("a", {"a": 0.6, "b": 0.3, "c": 0.1}, 0.4)

    assert answer.margin == pytest.approx(0.3)
    assert ChoiceAnswer("a", {"a": 1.0}, 1.0).margin == 1.0


def test_check_server_returns_the_model_name():
    server = FakeServer(body=answered("noul"))

    assert client_for(server).check_server() == "models/Julia-1-Q8_0.gguf"
    (_, payload, _), = server.calls
    assert [q["type"] for q in payload["questions"].values()] == ["noul"]


def error_of(server, call="choice"):
    client = client_for(server)
    with pytest.raises(JudgeError) as info:
        client.choice("x", "Which?", OPTIONS) if call == "choice" else client.check_server()
    return str(info.value)


def test_http_404_says_llama_cpp_is_too_old_and_names_the_url():
    message = error_of(FakeServer(404, {"error": {"message": "File Not Found", "type": "not_found_error", "code": 404}}))

    assert "llama update" in message and f"{URL}/v1/systemone" in message


def test_http_501_says_the_model_is_not_a_decision_model():
    message = error_of(FakeServer(501, {"error": {"message": "not a decision model", "code": 501}}))

    assert "decision model" in message and URL in message


def test_http_400_carries_the_servers_message():
    recorded_error = recorded("invalid_criteria")
    message = error_of(FakeServer(recorded_error["status"], recorded_error["body"]))

    assert "must be a non-empty object" in message and URL in message


def test_another_status_is_an_error_with_the_status():
    assert "500" in error_of(FakeServer(500, "boom"))


def test_a_refused_connection_asks_whether_the_server_runs():
    message = error_of(FakeServer(raises=ConnectionRefusedError()))

    assert "is the server running" in message and URL in message


def test_a_timeout_is_a_judge_error():
    assert URL in error_of(FakeServer(raises=TimeoutError()))


@pytest.mark.parametrize("body", [
    "not json",
    {"model": "m"},
    {"model": "m", "answers": {}},
    {"model": "m", "answers": {"other-id": {"type": "choice"}}},
    ["a", "list"],
])
def test_a_body_that_is_not_an_answer_is_a_judge_error(body):
    assert URL in error_of(FakeServer(body=body))


def test_a_choice_that_is_not_one_of_the_options_is_a_judge_error():
    body = answered("choice")
    body["answers"][judge_client.QUESTION_ID]["choice"] = "billing"

    assert "billing" in error_of(FakeServer(body=body))


def test_an_answer_of_another_type_is_a_judge_error():
    body = answered("choice")
    body["answers"][judge_client.QUESTION_ID] = {"type": "noul", "noul": 0.5}

    assert "choice" in error_of(FakeServer(body=body))


def test_probabilities_that_are_not_numbers_are_a_judge_error():
    body = answered("choice")
    body["answers"][judge_client.QUESTION_ID]["probabilities"] = {"social": "high"}

    assert URL in error_of(FakeServer(body=body))
