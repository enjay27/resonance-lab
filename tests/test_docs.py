"""The two glossary documents in docs/ are versioned, and the labeling guide stays in step with the taxonomy.

The translation glossary and the labeling guide are read by people and given to translation / labelling agents; a new season edits them,
so each carries a version header and a changelog, and the guide names every category the taxonomy has (a category added there but
missing in the guide fails here).
"""

import os
import re

import pytest

from config import BASE_DIR
from taxonomy import load_taxonomy

DOCS = os.path.join(BASE_DIR, "docs")
GUIDES = ["translation-glossary.md", "labeling-guide.md"]


def read(name):
    with open(os.path.join(DOCS, name), encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("name", GUIDES)
def test_a_guide_has_a_version_a_season_and_a_changelog(name):
    text = read(name)

    assert re.search(r"^\*\*Version:\*\* \d+\.\d+\.\d+ · \*\*Season:\*\* \S+", text, re.M), "header: **Version:** X.Y.Z · **Season:** S1 (...)"
    assert re.search(r"^## Changelog$", text, re.M)
    assert re.search(r"^\| \d+\.\d+\.\d+ \| \d{4}-\d{2}-\d{2} \|", text, re.M), "the changelog table needs a row: | X.Y.Z | YYYY-MM-DD | what changed |"


def test_the_changelog_names_the_version_of_the_header():
    for name in GUIDES:
        text = read(name)
        version = re.search(r"\*\*Version:\*\* (\d+\.\d+\.\d+)", text).group(1)

        assert re.search(rf"^\| {re.escape(version)} \|", text, re.M), f"{name}: the header version {version} has no changelog row"


def test_the_labeling_guide_names_every_category_of_the_taxonomy():
    text = read("labeling-guide.md")

    missing = [path for path in load_taxonomy().paths if f"`{path}`" not in text]

    assert not missing, f"configs/category_taxonomy.json has categories the guide does not describe: {missing}"


def test_the_translation_glossary_holds_the_official_class_names():
    text = read("translation-glossary.md")

    for korean in ("트윈 스트라이커", "스톰 블레이드", "윈드 나이트", "프로스트 메이지", "디바인 아처", "헤비 가디언", "실드 나이트", "실반 오라클", "비트 퍼포머"):
        assert korean in text


def test_the_guides_hold_no_chat_lines_only_terms():
    """No data in git: a glossary cell is a term or a short example, never a pasted message (a chat line is a long run of Japanese)."""
    longest = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]{25,}")
    for name in GUIDES:
        for number, line in enumerate(read(name).splitlines(), 1):
            assert not longest.search(line), f"{name}:{number}: 25+ Japanese characters in a row look like a chat line"
