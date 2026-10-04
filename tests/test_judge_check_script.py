import json

import pytest

import judge_check
from config import JUDGE_URL

NOUL_BODY = {"model": "Julia-1-Q8_0.gguf", "answers": {"q": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": 18, "output_tokens": 0}}


def test_it_prints_the_model_the_server_answers_with(capsys):
    calls = []

    def post(url, payload, timeout):
        calls.append(url)
        return 200, json.dumps(NOUL_BODY)

    judge_check.main(["--url", "http://box:9000"], post=post)

    assert calls == ["http://box:9000/v1/systemone"]
    assert "Julia-1-Q8_0.gguf" in capsys.readouterr().out


def test_the_default_url_is_the_configs(capsys):
    calls = []
    judge_check.main([], post=lambda url, payload, timeout: calls.append(url) or (200, json.dumps(NOUL_BODY)))

    assert calls == [f"{JUDGE_URL}/v1/systemone"]


def test_an_unreachable_server_exits_non_zero_with_the_reason(capsys):
    def post(url, payload, timeout):
        raise ConnectionRefusedError()

    with pytest.raises(SystemExit) as info:
        judge_check.main(["--url", "http://box:9000"], post=post)

    assert info.value.code == 1
    assert "http://box:9000" in capsys.readouterr().err


def test_an_old_llama_cpp_exits_non_zero_with_the_update_hint(capsys):
    with pytest.raises(SystemExit):
        judge_check.main([], post=lambda url, payload, timeout: (404, "{}"))

    assert "llama update" in capsys.readouterr().err
