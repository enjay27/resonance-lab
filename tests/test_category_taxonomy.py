"""configs/category_taxonomy.json (the categories a message can be in, with the description Gate 1 will show a judge) and the
shipped dataset recipes (configs/datasets/balanced-v1*.json) must agree: a recipe that names a category nobody can produce
would silently select nothing."""

import json
import os

import pytest

import config
from dataset_recipe import UNCATEGORIZED, is_category, load_recipe, recipe_key, recipe_path

SHIPPED = ("balanced-v1", "balanced-v1-detailed")
NOT_TRAINED_ON = "other"  # the catch-all: unreadable, fragments, symbols only


def _flatten(categories, prefix=""):
    flat = {}
    for name, node in categories.items():
        path = f"{prefix}{name}"
        flat[path] = node
        flat.update(_flatten(node.get("children", {}), prefix=path + "/"))
    return flat


@pytest.fixture(scope="module")
def taxonomy():
    with open(config.CATEGORY_TAXONOMY, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def flat(taxonomy):
    return _flatten(taxonomy["categories"])


def _weights(name):
    recipe, _ = load_recipe(recipe_path(name))
    return recipe.weights


def test_every_category_is_a_valid_path_with_a_description(flat):
    assert flat
    for path, node in flat.items():
        assert is_category(path), path
        assert isinstance(node.get("description"), str) and len(node["description"]) > 20, path


def test_market_talk_is_a_game_category_because_players_cannot_trade_with_each_other(flat):
    # Star Resonance has no player-to-player trade: players list items on the open market, so chat is about the market
    # (prices, what to list, what to buy), not an exchange between two players.
    assert "trade" not in flat and not [path for path in flat if path.startswith("trade/")]
    assert "game/market" in flat and "open market" in flat["game/market"]["description"].lower()
    assert "open market" in flat["question"]["description"].lower()  # "what should I sell?" is a question, not market talk


def test_the_taxonomy_has_a_catch_all_that_is_not_the_uncategorized_marker(flat):
    assert NOT_TRAINED_ON in flat and UNCATEGORIZED not in flat


def test_the_roots_are_the_choices_of_the_first_gate(taxonomy):
    roots = list(taxonomy["categories"])

    assert len(roots) >= 8 and len(roots) == len(set(roots))
    assert {"social", "chat", "game", "question", "coordination", "recruitment", "bot", "spam"} <= set(roots)


@pytest.mark.parametrize("name", SHIPPED)
def test_a_shipped_recipe_names_only_categories_of_the_taxonomy(name, flat):
    unknown = [key for key in _weights(name) if key not in flat]

    assert not unknown


@pytest.mark.parametrize("name", SHIPPED)
def test_a_shipped_recipe_never_trains_on_the_catch_all(name):
    assert not [key for key in _weights(name) if key == NOT_TRAINED_ON or key.startswith(NOT_TRAINED_ON + "/")]


@pytest.mark.parametrize("name", SHIPPED)
def test_the_keys_of_a_shipped_recipe_do_not_overlap(name):
    keys = list(_weights(name))

    for key in keys:
        assert recipe_key(key, [other for other in keys if other != key]) is None, key


def test_the_root_recipe_covers_every_root_except_the_catch_all(taxonomy):
    roots = set(taxonomy["categories"]) - {NOT_TRAINED_ON}

    assert set(_weights("balanced-v1")) == roots


def test_the_detailed_recipe_splits_the_root_weights_without_changing_them():
    roots, detailed = _weights("balanced-v1"), _weights("balanced-v1-detailed")

    for root, weight in roots.items():
        inside = [key for key in detailed if key == root or key.startswith(root + "/")]
        assert inside, root
        assert sum(detailed[key] for key in inside) == weight, root
    assert all(recipe_key(key, roots) for key in detailed)


def test_the_shipped_recipes_let_recruitment_walls_through_in_a_small_share():
    for name in SHIPPED:
        recipe, _ = load_recipe(recipe_path(name))
        assert recipe.keep == ("recruitment spam",)
        assert recipe.shares["spam"] <= 0.05 if "spam" in recipe.shares else True


def test_the_market_has_its_own_weight_in_the_detailed_recipe_and_trade_is_gone_from_both():
    roots, detailed = _weights("balanced-v1"), _weights("balanced-v1-detailed")

    assert "trade" not in roots and "trade" not in detailed
    assert detailed["game/market"] > 0 and roots["game"] == sum(w for key, w in detailed.items() if key.startswith("game/"))


def test_the_example_recipe_is_not_one_of_the_shipped_ones():
    assert "example" not in SHIPPED and os.path.exists(recipe_path("example"))


def _pool(categories):
    from dataset_recipe import line_key

    return [(line_key(f"{category} line {i}"), category) for category, n in categories.items() for i in range(n)]


def test_the_root_recipe_gives_every_root_its_weight_as_a_percent_when_the_pool_is_big_enough():
    from dataset_recipe import select_lines

    recipe, _ = load_recipe(recipe_path("balanced-v1"))
    roots = {root: 12 * int(weight) for root, weight in recipe.weights.items()}  # every root has 12 lines per percent

    selection = select_lines(recipe, _pool({**roots, "other": 500}))

    assert selection.allocation.targets == {root: 12 * int(weight) for root, weight in recipe.weights.items()}
    biggest = max(recipe.weights, key=recipe.weights.get)
    assert selection.allocation.total == 1200 and selection.allocation.limited_by == biggest  # the biggest weight runs out first
    assert not any(key for key, category in _pool({"other": 500}) if key in selection.keys)


def test_the_detailed_recipe_picks_a_root_only_line_up_only_through_a_key_that_covers_it():
    from dataset_recipe import select_lines

    recipe, _ = load_recipe(recipe_path("balanced-v1-detailed"))
    pool = _pool({"chat": 500, "chat/casual": 500, "chat/reaction": 500, "social": 500, "social/greeting": 500})

    selection = select_lines(recipe, pool)

    assert selection.available["chat/casual"] == 500 and selection.available["chat/reaction"] == 500
    assert selection.available["social"] == 1000  # social/greeting is under the 'social' key; the root-only 'chat' lines are under none
