import json
import os

import pytest

import gate_judge
import judge_prompts
from config import JUDGE_PROMPT_DIR
from judge_prompts import PromptError
from taxonomy import load_taxonomy

TAXONOMY = load_taxonomy()


def variant(**fields):
    return judge_prompts.parse_variant(fields)


# --- what a variant does to the question -------------------------------------------------------------------------------


def test_no_variant_is_todays_question_exactly_so_old_journals_stay_valid():
    instructions, options = judge_prompts.judge_question(TAXONOMY, None)

    assert instructions == gate_judge.INSTRUCTIONS
    assert options == gate_judge.choice_options(TAXONOMY)
    assert judge_prompts.judge_question(TAXONOMY, judge_prompts.load_variant("default")) == (instructions, options)


def test_a_description_replaces_the_text_of_its_root_and_keeps_the_order():
    _, options = judge_prompts.judge_question(TAXONOMY, variant(descriptions={"chat": "Talk, jokes and short reactions."}))

    assert options["chat"] == "Talk, jokes and short reactions."
    assert list(options) == list(gate_judge.choice_options(TAXONOMY))
    assert options["social"] == gate_judge.choice_options(TAXONOMY)["social"]


def test_dropping_a_root_leaves_it_out_of_the_question_and_keeps_the_order_of_the_rest():
    _, options = judge_prompts.judge_question(TAXONOMY, variant(drop=["other"]))

    assert "other" not in options
    assert list(options) == [name for name in gate_judge.choice_options(TAXONOMY) if name != "other"]


def test_the_instructions_can_be_replaced():
    instructions, _ = judge_prompts.judge_question(TAXONOMY, variant(instructions="Pick one."))

    assert instructions == "Pick one."


def test_every_change_gives_another_judge_id_so_journals_and_probes_never_mix():
    ids = set()
    for fields in ({}, {"descriptions": {"chat": "x"}}, {"drop": ["other"]}, {"instructions": "Pick one."}):
        instructions, options = judge_prompts.judge_question(TAXONOMY, variant(**fields))
        ids.add(gate_judge.judge_id("m", instructions, options))

    assert len(ids) == 4


# --- a variant that cannot be used ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("fields, message", [
    ({"descriptions": {"nonsense": "x"}}, "nonsense"),
    ({"descriptions": {"chat": ""}}, "chat"),
    ({"descriptions": {"chat": 5}}, "chat"),
    ({"drop": ["nonsense"]}, "nonsense"),
    ({"drop": "other"}, "drop"),
    ({"unknown_key": 1}, "unknown_key"),
    ({"instructions": ""}, "instructions"),
    ({"descriptions": {"other": "x"}, "drop": ["other"]}, "other"),
])
def test_a_malformed_variant_says_what_is_wrong(fields, message):
    with pytest.raises(PromptError, match=message):
        judge_prompts.judge_question(TAXONOMY, variant(**fields))


def test_a_question_needs_at_least_two_options():
    roots = list(TAXONOMY.roots)

    with pytest.raises(PromptError, match="two"):
        judge_prompts.judge_question(TAXONOMY, variant(drop=roots[:-1]))


# --- loading by name ------------------------------------------------------------------------------------------------------


def test_a_variant_is_a_json_file_read_as_utf8(tmp_path):
    (tmp_path / "mine.json").write_text(json.dumps({"description": "d", "descriptions": {"chat": "草, ww, すごい"}}, ensure_ascii=False), encoding="utf-8")

    loaded = judge_prompts.load_variant("mine", str(tmp_path))

    assert loaded.descriptions["chat"] == "草, ww, すごい"


def test_a_missing_variant_names_the_ones_there_are(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PromptError, match=r"nope.*a, default"):
        judge_prompts.load_variant("nope", str(tmp_path))


def test_a_damaged_variant_file_names_the_file(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(PromptError, match="bad.json"):
        judge_prompts.load_variant("bad", str(tmp_path))


@pytest.mark.parametrize("name", ["", "../x", "a b", "a/b", "é"])
def test_a_variant_name_is_a_plain_file_name(name):
    with pytest.raises(PromptError, match="name"):
        judge_prompts.load_variant(name)


# --- the variants the repo ships ---------------------------------------------------------------------------------------------


SHIPPED = judge_prompts.list_variants(JUDGE_PROMPT_DIR)


def test_the_repo_ships_the_variants_the_comparison_needs():
    assert {"default", "clean", "clean-no-other", "v2", "v2-no-other"} <= set(SHIPPED)


@pytest.mark.parametrize("name", SHIPPED)
def test_every_shipped_variant_builds_against_the_real_taxonomy(name):
    instructions, options = judge_prompts.judge_question(TAXONOMY, judge_prompts.load_variant(name))

    assert instructions and len(options) >= 2 and set(options) <= set(TAXONOMY.roots)
    assert all(isinstance(text, str) and text.strip() for text in options.values())


@pytest.mark.parametrize("name", [n for n in SHIPPED if n != "default"])
def test_a_shipped_variant_is_a_different_question_from_the_default(name):
    default = judge_prompts.judge_question(TAXONOMY, None)

    assert judge_prompts.judge_question(TAXONOMY, judge_prompts.load_variant(name)) != default


@pytest.mark.parametrize("name", [n for n in SHIPPED if n != "default"])
def test_the_training_note_never_reaches_the_judge(name):
    """"Never trained on." is a note about the translator's training data: in the option text it only confuses the judge."""
    _, options = judge_prompts.judge_question(TAXONOMY, judge_prompts.load_variant(name))

    assert not any("never trained on" in text.lower() for text in options.values())


def test_clean_changes_nothing_but_the_stray_note_of_other():
    _, default = judge_prompts.judge_question(TAXONOMY, None)
    _, clean = judge_prompts.judge_question(TAXONOMY, judge_prompts.load_variant("clean"))

    assert {name for name in default if default[name] != clean[name]} == {"other"}
    assert default["other"].startswith(clean["other"].rstrip(".")) and "never trained on" in default["other"].lower()


def test_the_no_other_variants_differ_from_their_twins_only_by_the_missing_root():
    for twin, name in (("clean", "clean-no-other"), ("v2", "v2-no-other")):
        _, with_other = judge_prompts.judge_question(TAXONOMY, judge_prompts.load_variant(twin))
        _, without = judge_prompts.judge_question(TAXONOMY, judge_prompts.load_variant(name))

        assert without == {k: v for k, v in with_other.items() if k != "other"}


def test_v2_teaches_the_party_slot_notation_the_judge_missed():
    _, options = judge_prompts.judge_question(TAXONOMY, judge_prompts.load_variant("v2"))

    for token in ("@T1", "@H1", "@D2", "↑"):
        assert token in options["recruitment"]
    assert "reaction" in options["chat"].lower() and "question" in options["game"].lower()


def test_the_shipped_files_are_utf8_json_with_a_description():
    for name in SHIPPED:
        if name == "default":
            continue
        with open(os.path.join(JUDGE_PROMPT_DIR, f"{name}.json"), encoding="utf-8") as f:
            assert json.load(f).get("description"), f"{name}: say what the variant is for"
