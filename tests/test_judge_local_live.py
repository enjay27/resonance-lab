"""Against a REAL local Kev model; skipped unless RESONANCE_KEV_LOCAL names a run (CI and the cloud gate skip it).

    RESONANCE_KEV_LOCAL=jaredpalmer/kev-0.8b python -m pytest tests/test_judge_local_live.py -v     (in the .venv-kev virtualenv)

Shape checks only, like test_judge_live.py: which model is good enough is for `categorize.py --probe`.
"""

import os

import pytest

import gate_judge
from judge_local import LocalKevClient
from taxonomy import load_taxonomy

RUN = os.environ.get("RESONANCE_KEV_LOCAL")
pytestmark = pytest.mark.skipif(not RUN, reason="needs the kev model in this process: set RESONANCE_KEV_LOCAL (e.g. jaredpalmer/kev-0.8b)")

OPTIONS = gate_judge.choice_options(load_taxonomy())


@pytest.fixture(scope="module")
def client():
    return LocalKevClient(RUN, device=os.environ.get("RESONANCE_KEV_DEVICE"))


def test_the_model_loads_and_names_itself(client):
    assert client.check_server().startswith(RUN)


def test_a_japanese_line_gets_a_root_with_probabilities_that_sum_to_one(client):
    answer = client.choice("ID:12345 ダンジョン行きませんか？ 2人募集", gate_judge.INSTRUCTIONS, OPTIONS)

    assert answer.choice in OPTIONS
    assert set(answer.probabilities) == set(OPTIONS)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=0.01)


def test_a_chat_channel_object_state_is_accepted(client):
    answer = client.choice({"channel": "party", "message": "2人募集"}, gate_judge.INSTRUCTIONS, OPTIONS)

    assert answer.choice in OPTIONS
