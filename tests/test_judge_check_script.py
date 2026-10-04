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


def test_local_checks_the_in_process_model_through_the_loader(capsys):
    import judge_local

    class Engine:
        @staticmethod
        def answer(request):
            return NOUL_BODY

    loads = []
    judge_check.main(["--local", "jaredpalmer/kev-0.8b", "--dtype", "fp32", "--device", "cpu"],
                     loader=lambda run, device, dtype: loads.append((run, device, dtype)) or Engine())

    assert loads == [("jaredpalmer/kev-0.8b", "cpu", "fp32")]
    assert judge_local.model_name("jaredpalmer/kev-0.8b", "fp32") in capsys.readouterr().out


def test_local_without_a_run_checks_the_configs_run(capsys):
    import config

    class Engine:
        @staticmethod
        def answer(request):
            return NOUL_BODY

    loads = []
    judge_check.main(["--local"], loader=lambda run, device, dtype: loads.append((run, device, dtype)) or Engine())

    assert loads == [(config.JUDGE_LOCAL_RUN, None, "bf16")]


def test_local_exits_non_zero_with_the_reason_when_the_model_does_not_load(capsys):
    def loader(run, device, dtype):
        raise OSError("no such file")

    with pytest.raises(SystemExit) as info:
        judge_check.main(["--local", "jaredpalmer/kev-0.8b"], loader=loader)

    assert info.value.code == 1
    assert "no such file" in capsys.readouterr().err


def test_local_and_url_exclude_each_other():
    with pytest.raises(SystemExit) as info:
        judge_check.main(["--local", "--url", "http://box:9000"])

    assert info.value.code == 2
