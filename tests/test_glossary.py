import json
import os

import pytest

import glossary
from config import GLOSSARY_DIR, GLOSSARY_DOC
from glossary import GlossaryError

SEASONS = sorted(name[:-5] for name in os.listdir(GLOSSARY_DIR) if name.endswith(".json"))


def write(tmp_path, **fields):
    data = {"version": 1, "season": "T", "doc_version": "1.0.0", "required": [], "banned": [], "fixes": []}
    data.update(fields)
    path = tmp_path / "t.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(path)


# --- loading ---------------------------------------------------------------------------------------------------------------


def test_a_glossary_loads_its_rules_with_compiled_patterns(tmp_path):
    path = write(tmp_path, required=[{"ja": "継", "unless": "継続", "ko": "계속"}], banned=[{"ko": "계승", "why": "x"}],
                 fixes=[{"when": "ワイプ", "old": "와이프", "new": "전멸", "why": "wife"}])

    g = glossary.load_glossary(path)

    assert (g.season, g.doc_version) == ("T", "1.0.0")
    assert g.required[0].ko == "계속" and g.required[0].ja.search("継") and g.required[0].unless.search("継続")
    assert g.banned[0].when is None and g.fixes[0].when.search("ワイプ")


def test_a_missing_file_is_an_error_naming_it(tmp_path):
    with pytest.raises(GlossaryError, match="nowhere.json"):
        glossary.load_glossary(str(tmp_path / "nowhere.json"))


@pytest.mark.parametrize("fields, message", [
    ({"required": [{"ja": "(", "ko": "x"}]}, "required"),          # not a regex
    ({"required": [{"ja": "a"}]}, "ko"),
    ({"banned": [{"why": "x"}]}, "ko"),
    ({"fixes": [{"when": "a", "old": "b"}]}, "new"),
    ({"fixes": [{"when": "a", "old": "b", "new": "b"}]}, "same"),   # a fix that changes nothing
    ({"doc_version": "1.0"}, "doc_version"),
    ({"season": ""}, "season"),
])
def test_a_malformed_glossary_says_what_is_wrong(tmp_path, fields, message):
    with pytest.raises(GlossaryError, match=message):
        glossary.load_glossary(write(tmp_path, **fields))


# --- the document's version ------------------------------------------------------------------------------------------------


def test_the_version_is_read_from_the_header_of_the_document(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("# T\n\n**Version:** 1.12.3 · **Season:** S1 (2026-10)\n", encoding="utf-8")

    assert glossary.doc_version(str(doc)) == "1.12.3"


def test_a_document_without_a_version_is_an_error(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("# T\n", encoding="utf-8")

    with pytest.raises(GlossaryError, match="Version"):
        glossary.doc_version(str(doc))


@pytest.mark.parametrize("a, b, expected", [("1.0.0", "1.0.1", -1), ("1.10.0", "1.9.0", 1), ("2.0.0", "2.0.0", 0)])
def test_versions_compare_as_numbers_not_text(a, b, expected):
    assert glossary.compare_versions(a, b) == expected


def test_the_sha_of_a_glossary_file_changes_with_its_content(tmp_path):
    a = write(tmp_path, season="A")
    first = glossary.file_sha1(a)
    write(tmp_path, season="B")

    assert first != glossary.file_sha1(a) and len(first) == 40


# --- the shipped glossary and the document agree -----------------------------------------------------------------------------


@pytest.mark.parametrize("season", SEASONS)
def test_the_shipped_glossary_was_written_for_the_current_version_of_the_document(season):
    g = glossary.load_glossary(os.path.join(GLOSSARY_DIR, season + ".json"))

    assert g.doc_version == glossary.doc_version(GLOSSARY_DOC), (
        "docs/translation-glossary.md changed: update configs/glossary/<season>.json to match it, then set its doc_version")


@pytest.mark.parametrize("season", SEASONS)
def test_every_official_korean_name_of_the_rules_is_in_the_document(season):
    g = glossary.load_glossary(os.path.join(GLOSSARY_DIR, season + ".json"))
    with open(GLOSSARY_DOC, encoding="utf-8") as f:
        doc = f.read()

    missing = [rule.ko for rule in g.required if rule.ko not in doc]

    assert not missing, f"the rules require names the document never mentions: {missing}"
