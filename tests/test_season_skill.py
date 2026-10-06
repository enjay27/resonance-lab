"""The runbook for a season's labelling and translation (.claude/skills/season-data/SKILL.md) names the real commands and the real guards."""

import os
import re

import pytest

import label_lines
import translate_agents
from config import BASE_DIR

SKILL = os.path.join(BASE_DIR, ".claude", "skills", "season-data", "SKILL.md")


def text():
    with open(SKILL, encoding="utf-8") as f:
        return f.read()


def test_the_skill_has_a_name_and_a_description_that_says_when_to_use_it():
    head = text().split("---")[1]

    assert re.search(r"^name: season-data$", head, re.M)
    assert re.search(r"^description: .{40,}", head, re.M | re.S)


@pytest.mark.parametrize("script, module", [("translate_agents.py", translate_agents), ("label_lines.py", label_lines)])
def test_every_command_of_the_scripts_is_in_the_runbook(script, module):
    body = text()

    for command in module.COMMANDS:
        assert f"scripts/{script} {command}" in body, f"the runbook never shows `scripts/{script} {command}`"


def test_the_runbook_names_the_documents_the_agents_are_briefed_with_and_the_version_guard():
    body = text()

    for needle in ("docs/labeling-guide.md", "docs/translation-glossary.md", "configs/glossary/", "configs/label_map.json", "STALE", "data/translation", "data/labeling"):
        assert needle in body


def test_the_runbook_keeps_the_rules_that_cost_a_wrong_turn():
    body = text()

    for needle in ("ワイプ", "継", "迷妄", "no data in git", "NOT VERIFIED"):
        assert needle.lower() in body.lower()
