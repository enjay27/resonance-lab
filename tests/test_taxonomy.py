import json

import pytest

import config
import taxonomy
from taxonomy import Taxonomy, TaxonomyError, coverage, load_taxonomy, root_of

TAX = Taxonomy(roots=("chat", "game", "other"), paths={"chat": "talk", "game": "play", "game/combat": "fight", "other": "rest"})


def sample(category=None):
    row = {"original": "a", "translated": "b"}
    if category is not None:
        row["category"] = category
    return row


def test_the_repo_taxonomy_loads_with_roots_children_and_descriptions():
    loaded = load_taxonomy()

    assert "game" in loaded.roots and "other" in loaded.roots and "trade" not in loaded.roots
    assert "game/market" in loaded.paths and loaded.paths["game"]
    assert loaded.roots == tuple(path for path in loaded.paths if "/" not in path)


def test_a_root_is_the_part_before_the_first_slash():
    assert root_of("game/combat") == root_of("game") == "game"
    assert root_of("game/combat/skills") == "game"
    assert root_of("unknown") == "unknown"


def test_a_missing_or_broken_taxonomy_is_an_error_that_names_the_file(tmp_path):
    with pytest.raises(TaxonomyError, match="missing.json"):
        load_taxonomy(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"categories": []}), encoding="utf-8")
    with pytest.raises(TaxonomyError, match="bad.json"):
        load_taxonomy(str(bad))


def test_coverage_counts_lines_per_root_and_a_child_counts_for_its_root():
    found = coverage([sample("chat"), sample("game"), sample("game/combat"), sample("game/combat")], TAX)

    assert found["counts"] == {"chat": 1, "game": 3, "other": 0}  # every root is listed, also the empty ones
    assert found["unknown"] == {} and found["unlabeled"] == 0


def test_coverage_lists_unlabeled_lines_and_labels_the_taxonomy_does_not_know():
    found = coverage([sample(), sample(""), sample("Chat"), sample("Chat"), sample("game/swim"), sample("chat")], TAX)

    assert found["unlabeled"] == 2
    assert found["unknown"] == {"Chat": 2, "game/swim": 1}
    assert found["counts"] == {"chat": 1, "game": 0, "other": 0}  # an unknown label is not counted under a root


def test_the_taxonomy_module_reads_the_configured_file():
    assert taxonomy.CATEGORY_TAXONOMY == config.CATEGORY_TAXONOMY
