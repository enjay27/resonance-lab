import sys

import pytest

import judge_client
import judge_local
from judge_client import JudgeError, SystemOneClient
from test_judge_client import OPTIONS, answered

RUN = "jaredpalmer/kev-9b@v1.0"


class StubEngine:
    """What the loader returns: `answer(request dict) -> the /v1/systemone body`, like kev.serve.Server through its adapter."""

    def __init__(self, body=None, raises=None):
        self.body, self.raises, self.requests = body, raises, []

    def answer(self, request):
        self.requests.append(request)
        if self.raises:
            raise self.raises
        return self.body


class Loader:
    """The injected `loader(run, device, dtype) -> engine`; counts the loads."""

    def __init__(self, engine=None, raises=None):
        self.engine, self.raises, self.calls = engine, raises, []

    def __call__(self, run, device, dtype):
        self.calls.append((run, device, dtype))
        if self.raises:
            raise self.raises
        return self.engine


def client_for(engine=None, loader=None, **kwargs):
    return judge_local.LocalKevClient(RUN, loader=loader or Loader(engine), **kwargs)


def test_it_is_a_systemone_client_so_the_two_methods_answer_the_same_way():
    assert issubclass(judge_local.LocalKevClient, SystemOneClient)


def test_choice_asks_the_engine_one_choice_question_and_maps_the_answer():
    engine = StubEngine(answered("choice"))

    answer = client_for(engine).choice("2人募集", "Which category?", OPTIONS)

    (request,) = engine.requests
    assert request["state"] == "2人募集"
    assert list(request["questions"].values()) == [{"type": "choice", "instructions": "Which category?", "criteria": OPTIONS}]
    assert isinstance(answer, judge_client.ChoiceAnswer)
    assert answer.choice in OPTIONS
    assert answer.margin > 0


def test_an_object_state_goes_to_the_engine_as_it_is():
    engine = StubEngine(answered("choice_object_state"))
    state = {"channel": "party", "message": "2人募集"}

    client_for(engine).choice(state, "Which category?", OPTIONS)

    assert engine.requests[0]["state"] == state


def test_the_model_is_loaded_lazily_and_once():
    loader = Loader(StubEngine(answered("choice")))
    client = client_for(loader=loader, device="cpu", dtype="fp32")
    assert loader.calls == []

    client.choice("a", "Which?", OPTIONS)
    client.choice("b", "Which?", OPTIONS)

    assert loader.calls == [(RUN, "cpu", "fp32")]


def test_the_device_is_left_to_the_loader_and_the_dtype_is_bf16_by_default():
    loader = Loader(StubEngine(answered("choice")))

    client_for(loader=loader).choice("a", "Which?", OPTIONS)

    assert loader.calls == [(RUN, None, "bf16")]


def test_an_unknown_dtype_is_refused_before_anything_loads():
    loader = Loader()

    with pytest.raises(ValueError, match="int4"):
        client_for(loader=loader, dtype="int4")

    assert loader.calls == []


def test_check_server_loads_the_model_and_names_the_run_and_dtype():
    loader = Loader(StubEngine(answered("noul")))

    name = client_for(loader=loader, dtype="fp32").check_server()

    assert name == f"{RUN}+fp32"
    assert loader.calls


def test_a_failing_load_is_a_judge_error_naming_the_run():
    client = client_for(loader=Loader(raises=OSError("no such file")))

    with pytest.raises(JudgeError, match=r"jaredpalmer/kev-9b@v1\.0.*no such file"):
        client.choice("a", "Which?", OPTIONS)


def test_out_of_memory_says_so_and_names_the_smaller_models():
    class OutOfMemoryError(RuntimeError):  # torch.cuda.OutOfMemoryError, without importing torch
        pass

    client = client_for(loader=Loader(raises=OutOfMemoryError("CUDA out of memory")))

    with pytest.raises(JudgeError, match=r"out of memory.*kev-4b"):
        client.choice("a", "Which?", OPTIONS)


def test_a_failing_load_is_tried_again_by_the_next_call():
    loader = Loader(raises=OSError("disk"))
    client = client_for(loader=loader)

    for _ in range(2):
        with pytest.raises(JudgeError):
            client.choice("a", "Which?", OPTIONS)

    assert len(loader.calls) == 2


def test_an_invalid_request_is_a_judge_error_not_a_traceback():
    client = client_for(StubEngine(raises=ValueError("criteria must have 1..255 options")))

    with pytest.raises(JudgeError, match=r"jaredpalmer/kev-9b@v1\.0.*invalid request.*criteria must have"):
        client.choice("a", "Which?", OPTIONS)


def test_an_engine_that_fails_in_the_forward_pass_is_a_judge_error():
    client = client_for(StubEngine(raises=RuntimeError("CUDA error")))

    with pytest.raises(JudgeError, match=r"jaredpalmer/kev-9b@v1\.0.*CUDA error"):
        client.choice("a", "Which?", OPTIONS)


def test_a_choice_that_is_not_an_option_is_refused_like_the_servers_answer():
    body = answered("choice")
    body["answers"]["q"]["choice"] = "not-an-option"

    with pytest.raises(JudgeError, match="not one of the options"):
        client_for(StubEngine(body)).choice("a", "Which?", OPTIONS)


def test_without_kev_installed_the_default_loader_says_which_venv_to_use(monkeypatch):
    monkeypatch.setitem(sys.modules, "kev", None)  # `import kev` raises ImportError

    with pytest.raises(JudgeError, match=r"requirements-kev\.txt"):
        judge_local.LocalKevClient(RUN).check_server()


def test_the_default_run_is_the_pinned_kev_9b_in_the_config():
    import config

    assert config.JUDGE_LOCAL_RUN == "jaredpalmer/kev-9b@v1.0"
