import re

import pytest

import glossary
import translation_assemble as assemble

NO_RULES = glossary.Glossary(season="T", doc_version="1.0.0", required=(), banned=(), fixes=())


def fix(when, old, new, unless=None):
    return glossary.Fix(re.compile(when), re.compile(unless) if unless else None, old, new, "why")


def rules(*fixes):
    return glossary.Glossary(season="T", doc_version="1.0.0", required=(), banned=(), fixes=tuple(fixes))


# --- choosing and batching the lines ---------------------------------------------------------------------------------------


LABELLED = [{"original": f"line{i}", "category": category, "channel": channel} for i, (category, channel) in enumerate([
    ("chat", "PARTY"), ("recruitment/guild", "WORLD"), ("non_japanese", "WORLD"), ("recruitment/party", "WORLD"), ("other/placeholder", "LOCAL")], 1)]


def test_the_lines_to_translate_skip_what_the_maintainer_does_not_want_and_keep_stable_ids():
    selected, skipped = assemble.select_lines(LABELLED)

    assert [r["i"] for r in selected] == [1, 4]  # the position in the labelled file, so the ids survive a change of the skip list
    assert selected[0] == {"i": 1, "ch": "P", "cat": "chat", "ja": "line1"}
    assert skipped == {"recruitment/guild": 1, "non_japanese": 1, "other/placeholder": 1}


def test_a_skip_list_can_be_changed_for_a_run():
    selected, skipped = assemble.select_lines(LABELLED, skip=())

    assert len(selected) == 5 and skipped == {}


def test_a_line_without_text_or_category_is_an_error():
    with pytest.raises(assemble.AssembleError, match="line 2"):
        assemble.select_lines([LABELLED[0], {"category": "chat"}])


def test_batches_are_even_in_size_and_keep_the_order():
    rows = [{"i": n} for n in range(1, 11)]

    batches = assemble.split_batches(rows, 4, "party")

    assert [(name, [r["i"] for r in part]) for name, part in batches] == [
        ("party1", [1, 2, 3, 4]), ("party2", [5, 6, 7]), ("party3", [8, 9, 10])]  # 10 lines, at most 4: three batches of 4, 3, 3


def test_a_batch_size_below_one_is_an_error():
    with pytest.raises(assemble.AssembleError, match="size"):
        assemble.split_batches([{"i": 1}], 0, "b")


# --- the corrections --------------------------------------------------------------------------------------------------------


def test_a_fix_replaces_the_rendering_in_the_text_and_in_the_terms():
    ko, terms, applied = assemble.apply_fixes("ワイプした", "와이프 했다", [["ワイプ", "와이프"]], rules(fix("ワイプ", "와이프", "전멸")))

    assert (ko, terms) == ("전멸 했다", [["ワイプ", "전멸"]]) and applied == ["why"]


def test_a_fix_does_nothing_when_its_condition_is_not_met():
    g = rules(fix("MN", "MN", "NM", unless="NM"))

    assert assemble.apply_fixes("NM MN", "MN", [], g)[0] == "MN"          # the Japanese has NM: unless
    assert assemble.apply_fixes("something", "MN", [], g)[0] == "MN"      # no `when`
    assert assemble.apply_fixes("MN募集", "MN", [], g)[0] == "NM"


def test_the_middle_dot_of_the_source_is_restored_unless_the_source_has_the_other_dots():
    assert assemble.apply_fixes("(・ω・)", "(･ω･)", [], NO_RULES)[0] == "(・ω・)"
    assert assemble.apply_fixes("(･ω･)", "(･ω･)", [], NO_RULES)[0] == "(･ω･)"
    assert assemble.apply_fixes("a・b", "a·b", [], NO_RULES)[0] == "a・b"


def test_an_arrow_of_the_source_stays_an_arrow_not_이상():
    assert assemble.apply_fixes("4500↑ @D2", "4500 이상 @D2", [], NO_RULES)[0] == "4500↑ @D2"
    assert assemble.apply_fixes("4500以上", "4500 이상", [], NO_RULES)[0] == "4500 이상"      # the source said 以上: stays
    assert assemble.apply_fixes("30k↑", "30k 이상", [], NO_RULES)[0] == "30k↑"


def test_the_slot_spacing_follows_the_majority_of_the_lines():
    rows = [{"translated": "@D 많이"}, {"translated": "@D1 많이"}, {"translated": "@T많이"}]

    changed = assemble.normalise_slot_spacing(rows)

    assert changed == 1 and [r["translated"] for r in rows] == ["@D 많이", "@D1 많이", "@T 많이"]


def test_the_slot_spacing_of_a_corpus_without_the_other_form_is_left_alone():
    rows = [{"translated": "@D 많이"}, {"translated": "@H1 많이"}]

    assert assemble.normalise_slot_spacing(rows) == 0


# --- putting the rounds together --------------------------------------------------------------------------------------------


INPUTS = {1: {"i": 1, "ch": "W", "cat": "recruitment/party", "ja": "ワイプ"}, 2: {"i": 2, "ch": "P", "cat": "chat", "ja": "草"}}


def test_the_latest_round_wins_and_the_fixes_run_once_at_the_end():
    round1 = [{"i": 1, "ko": "와이프", "terms": [["ワイプ", "와이프"]]}, {"i": 2, "ko": "ㅋㅋ", "terms": []}]
    round2 = [{"i": 1, "ko": "와이프 났다", "terms": [["ワイプ", "와이프"]], "flag": "check"}]

    rows, applied = assemble.assemble(INPUTS, [round1, round2], rules(fix("ワイプ", "와이프", "전멸")))

    assert [(r["i"], r["translated"]) for r in rows] == [(1, "전멸 났다"), (2, "ㅋㅋ")]
    assert rows[0]["flag"] == "check" and "flag" not in rows[1]
    assert rows[0]["original"] == "ワイプ" and rows[0]["channel"] == "W" and rows[0]["category"] == "recruitment/party"
    assert applied["why"] == 1


def test_a_line_no_round_translated_is_an_error_naming_it():
    with pytest.raises(assemble.AssembleError, match=r"not translated.*2"):
        assemble.assemble(INPUTS, [[{"i": 1, "ko": "전멸", "terms": []}]], NO_RULES)


# --- the terms --------------------------------------------------------------------------------------------------------------


ROWS = [{"terms": [["継", "계속"], ["火力", "딜"]]}, {"terms": [["継", "계속"], ["火力", "딜러"]]}, {"terms": [["継", "계속"]]}]


def test_the_term_table_counts_every_rendering_and_marks_the_inconsistent_ones():
    table = assemble.term_table(ROWS)

    assert table["継"] == {"계속": 3} and table["火力"] == {"딜": 1, "딜러": 1}
    assert assemble.multiple_renderings(table) == ["火力"]


def test_the_term_tsv_lists_the_most_used_term_first():
    text = assemble.format_terms_tsv(assemble.term_table(ROWS))

    lines = text.splitlines()
    assert lines[0] == "japanese\tuses\trenderings (count)\tconsistent"
    assert lines[1] == "継\t3\t계속 (3)\tyes" and lines[2].startswith("火力\t2\t") and lines[2].endswith("\tNO")


# --- the version guard -------------------------------------------------------------------------------------------------------


def test_batches_prepared_with_the_current_glossary_may_be_assembled():
    assemble.require_current({"doc_version": "1.2.0"}, "1.2.0")


@pytest.mark.parametrize("prepared, now", [("1.0.0", "1.1.0"), ("1.2.0", "1.10.0"), ("2.0.0", "1.9.0")])
def test_a_glossary_that_changed_since_the_batches_were_prepared_refuses_to_assemble(prepared, now):
    with pytest.raises(assemble.StaleBatches, match=rf"{prepared}.*{now}"):
        assemble.require_current({"doc_version": prepared}, now)


def test_the_run_record_names_what_the_batches_were_made_with():
    record = assemble.run_record("S1", "1.0.0", "abc123", batch_size=100, round_no=1, counts={"lines": 5}, batches=["a1"], git_sha="deadbeef")

    assert record["season"] == "S1" and record["doc_version"] == "1.0.0" and record["glossary_sha1"] == "abc123"
    assert record["rounds"] == {"1": ["a1"]} and record["git_sha"] == "deadbeef" and record["counts"] == {"lines": 5}


def test_a_later_round_is_added_to_the_record_and_keeps_the_first():
    first = assemble.run_record("S1", "1.0.0", "abc", batch_size=100, round_no=1, counts={"lines": 5}, batches=["a1"], git_sha="x")

    second = assemble.add_round(first, 2, ["rev1", "rev2"], "1.0.0")

    assert second["rounds"] == {"1": ["a1"], "2": ["rev1", "rev2"]}
    with pytest.raises(assemble.StaleBatches):
        assemble.add_round(first, 2, ["rev1"], "1.1.0")  # a round prepared with another glossary version than the run


# --- the brief the agents get ------------------------------------------------------------------------------------------------


DOC = """# T

**Version:** 1.0.0 · **Season:** S1

## 2. Style rules
Keep numbers.

## 3. Official
트윈 스트라이커

## Updating for a new season
steps

## Changelog
| 1.0.0 | 2026-10-05 | x |

## Observed terms, season 1 (generated, provisional)
| 継 | 218 | 계속 (218) |
"""


def test_the_brief_is_the_document_without_the_maintenance_and_observed_sections_plus_the_output_format():
    brief = assemble.build_brief(DOC, round_no=1)

    assert "트윈 스트라이커" in brief and "Keep numbers." in brief
    assert "Changelog" not in brief and "Observed terms" not in brief and "Updating for a new season" not in brief
    assert '"i"' in brief and '"ko"' in brief and '"terms"' in brief and '"flag"' in brief
    assert "prev" not in brief
    assert "version 1.0.0" in brief.lower()


def test_a_revision_brief_explains_prev():
    brief = assemble.build_brief(DOC, round_no=2)

    assert "prev" in brief and "revision" in brief.lower()
