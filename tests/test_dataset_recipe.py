import json
import os
import random
from fractions import Fraction

import pytest

import config
from dataset_recipe import (
    UNCATEGORIZED, RecipeError, allocate, category_of, line_key, line_rank, load_recipe, parse_recipe,
    read_categories, recipe_key, recipe_path, select_lines,
)
from valsplit import bucket


def recipe(**changes):
    data = {"seed": 7, "categories": {"greetings": {"weight": 1}, "chat": {"weight": 3}}, **changes}
    return parse_recipe(data)


# --- the recipe file --------------------------------------------------------------------------------------------


def test_a_recipe_has_a_seed_a_total_and_weights_whose_share_is_weight_over_total_weight():
    parsed = recipe(total=1000)

    assert parsed.seed == 7
    assert parsed.total == 1000
    assert parsed.weights == {"greetings": Fraction(1), "chat": Fraction(3)}
    assert parsed.shares == {"greetings": Fraction(1, 4), "chat": Fraction(3, 4)}
    assert parsed.keep == ()


def test_the_defaults_are_a_fixed_seed_no_total_and_nothing_kept():
    parsed = parse_recipe({"categories": {"chat": {"weight": 1}}})

    assert (parsed.seed, parsed.total, parsed.keep) == (42, None, ())


def test_weights_are_relative_so_they_need_not_add_up_to_anything():
    parsed = parse_recipe({"categories": {"a": {"weight": 1}, "b": {"weight": 1}, "c": {"weight": 2}}})

    assert parsed.shares == {"a": Fraction(1, 4), "b": Fraction(1, 4), "c": Fraction(1, 2)}


def test_decimal_weights_give_the_same_shares_as_whole_ones():
    decimals = parse_recipe({"categories": {"a": {"weight": 0.1}, "b": {"weight": 0.2}, "c": {"weight": 0.7}}})
    wholes = parse_recipe({"categories": {"a": {"weight": 1}, "b": {"weight": 2}, "c": {"weight": 7}}})

    assert decimals.shares == wholes.shares == {"a": Fraction(1, 10), "b": Fraction(1, 5), "c": Fraction(7, 10)}


def test_a_single_category_takes_the_whole_dataset_whatever_its_weight():
    assert parse_recipe({"categories": {"a": {"weight": 5}}}).shares == {"a": Fraction(1)}


@pytest.mark.parametrize("key", ["share", "fraction"])
def test_only_weight_is_supported_and_share_or_fraction_says_so(key):
    with pytest.raises(RecipeError, match="only 'weight'"):
        parse_recipe({"categories": {"a": {key: 0.5}}})


@pytest.mark.parametrize("weight", [0, -0.1, True, "2", None])
def test_a_weight_must_be_a_number_above_0(weight):
    with pytest.raises(RecipeError, match="weight"):
        parse_recipe({"categories": {"a": {"weight": weight}}})


def test_an_unknown_key_is_refused_so_a_typo_cannot_silently_do_nothing():
    with pytest.raises(RecipeError, match="shares"):
        parse_recipe({"categories": {"a": {"weight": 1}}, "shares": {}})
    with pytest.raises(RecipeError, match="max"):
        parse_recipe({"categories": {"a": {"weight": 1, "max": 3}}})


def test_a_recipe_without_categories_is_refused():
    with pytest.raises(RecipeError, match="categories"):
        parse_recipe({"seed": 1})
    with pytest.raises(RecipeError, match="categories"):
        parse_recipe({"categories": {}})


@pytest.mark.parametrize("name", ["", "/chat", "chat/", "a//b", "Chat", "a b", "chat/Party"])
def test_a_category_is_a_lower_case_path(name):
    with pytest.raises(RecipeError, match="category"):
        parse_recipe({"categories": {name: {"weight": 1}}})


@pytest.mark.parametrize("total", [0, -5, 1.5, True, "10"])
def test_a_total_is_a_positive_whole_number_or_null(total):
    with pytest.raises(RecipeError, match="total"):
        recipe(total=total)


def test_only_a_known_filter_can_be_kept():
    assert recipe(keep=["recruitment spam"]).keep == ("recruitment spam",)
    with pytest.raises(RecipeError, match="hallucination"):
        recipe(keep=["hallucination"])
    with pytest.raises(RecipeError, match="keep"):
        recipe(keep="recruitment spam")


def test_a_description_is_allowed_and_kept():
    assert recipe(description="for the sweep").description == "for the sweep"


def test_a_recipe_file_is_loaded_with_a_hash_that_ignores_formatting(tmp_path):
    data = {"seed": 1, "categories": {"a": {"weight": 1}, "b": {"weight": 1}}}
    one, two, three = tmp_path / "one.json", tmp_path / "two.json", tmp_path / "three.json"
    one.write_text(json.dumps(data), encoding="utf-8")
    two.write_text(json.dumps(data, indent=4, sort_keys=True), encoding="utf-8")
    three.write_text(json.dumps({**data, "seed": 2}), encoding="utf-8")

    parsed, digest = load_recipe(str(one))

    assert parsed.seed == 1
    assert digest == load_recipe(str(two))[1]
    assert digest != load_recipe(str(three))[1]
    assert len(digest) == 64


def test_a_broken_recipe_file_names_the_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    with pytest.raises(RecipeError, match="bad.json"):
        load_recipe(str(bad))
    with pytest.raises(RecipeError, match="missing.json"):
        load_recipe(str(tmp_path / "missing.json"))


def test_a_recipe_is_found_by_name_in_the_recipe_folder():
    assert recipe_path("balanced") == os.path.join(config.RECIPE_DIR, "balanced.json")
    assert recipe_path("balanced", directory="/x") == os.path.join("/x", "balanced.json")


def test_every_recipe_in_the_repo_loads():
    for name in os.listdir(config.RECIPE_DIR):
        if name.endswith(".json"):
            load_recipe(os.path.join(config.RECIPE_DIR, name))


# --- lines and their categories ---------------------------------------------------------------------------------


def test_a_line_key_is_the_sha1_the_validation_split_uses_so_variants_share_a_category():
    assert line_key("おやすみ") == line_key("おやすみ！") == line_key(" おやすみ\n")
    key = line_key("おやすみ")
    assert len(key) == 40 and int(key, 16) >= 0
    assert int(key[:8], 16) / 0x100000000 == pytest.approx(bucket("おやすみ"))


def test_the_categories_file_maps_line_keys_to_categories(tmp_path):
    path = tmp_path / "categories.jsonl"
    rows = [{"key": line_key("おやすみ"), "category": "greetings", "confidence": 0.9, "by": "regex"},
            {"key": line_key("募集"), "category": "recruitment/party"}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")

    categories = read_categories(str(path))

    assert categories == {line_key("おやすみ"): "greetings", line_key("募集"): "recruitment/party"}
    assert category_of("おやすみ！", categories) == "greetings"
    assert category_of("ぜんぜん別の行", categories) == UNCATEGORIZED


def test_a_bad_categories_file_names_the_line(tmp_path):
    path = tmp_path / "categories.jsonl"
    good = json.dumps({"key": line_key("a"), "category": "chat"})
    for bad, message in [("{oops", "line 2"), (json.dumps({"key": "x"}), "line 2"),
                         (json.dumps({"key": line_key("b"), "category": "Bad Name"}), "line 2")]:
        path.write_text(good + "\n" + bad + "\n", encoding="utf-8")
        with pytest.raises(RecipeError, match=message):
            read_categories(str(path))


def test_one_line_in_two_categories_is_refused_but_the_same_twice_is_fine(tmp_path):
    path = tmp_path / "categories.jsonl"
    row = {"key": line_key("a"), "category": "chat"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    assert read_categories(str(path)) == {line_key("a"): "chat"}

    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "category": "bot"}) + "\n", encoding="utf-8")
    with pytest.raises(RecipeError, match="chat.*bot|bot.*chat"):
        read_categories(str(path))


def test_a_key_of_the_recipe_covers_its_children_and_the_most_specific_one_wins():
    keys = ["recruitment", "recruitment/guild", "chat"]

    assert recipe_key("recruitment", keys) == "recruitment"
    assert recipe_key("recruitment/party", keys) == "recruitment"
    assert recipe_key("recruitment/guild", keys) == "recruitment/guild"
    assert recipe_key("recruitment/guild/walls", keys) == "recruitment/guild"
    assert recipe_key("chat", keys) == "chat"


def test_a_category_that_no_key_covers_has_no_recipe_key():
    keys = ["recruitment", "chat"]

    assert recipe_key("recruitmentx", keys) is None  # a prefix of the text is not a parent in the path
    assert recipe_key("trade", keys) is None
    assert recipe_key(UNCATEGORIZED, keys) is None
    assert recipe_key(UNCATEGORIZED, [UNCATEGORIZED, "chat"]) == UNCATEGORIZED


# --- how many lines each key gets -------------------------------------------------------------------------------


def shares(**named):
    return {name: Fraction(str(value)) for name, value in named.items()}


def test_the_largest_dataset_the_shares_allow_ends_where_the_first_category_runs_out():
    result = allocate(shares(a=0.5, b=0.5), {"a": 10, "b": 100})

    assert result.targets == {"a": 10, "b": 10}
    assert result.total == 20
    assert result.limited_by == "a"


def test_a_total_is_split_by_the_shares():
    result = allocate(shares(a=0.2, b=0.8), {"a": 1000, "b": 1000}, total=100)

    assert result.targets == {"a": 20, "b": 80}
    assert result.total == 100
    assert result.limited_by is None


def test_a_total_the_shares_cannot_reach_is_capped_and_says_which_category_limits_it():
    result = allocate(shares(a=0.5, b=0.5), {"a": 10, "b": 100}, total=500)

    assert result.total == 20
    assert result.requested == 500
    assert result.limited_by == "a"


def test_a_category_without_lines_makes_an_empty_dataset_and_is_named():
    result = allocate(shares(a=0.5, b=0.5), {"a": 0, "b": 100})

    assert result.total == 0
    assert result.limited_by == "a"
    assert result.targets == {"a": 0, "b": 0}


def test_the_targets_add_up_to_the_total_and_follow_the_shares_within_one_line():
    share = shares(a=0.1, b=0.3, c=0.6)
    for total in range(1, 200):
        result = allocate(share, {"a": 1000, "b": 1000, "c": 1000}, total=total)

        assert sum(result.targets.values()) == total
        for name, fraction in share.items():
            assert abs(result.targets[name] - total * fraction) < 1


def test_a_bigger_total_never_takes_a_line_away_from_a_category():
    # Plain largest-remainder rounding can (the Alabama paradox); the nested learning curve needs this to hold.
    share = shares(a=0.1, b=0.3, c=0.6)
    previous = {"a": 0, "b": 0, "c": 0}
    for total in range(1, 300):
        result = allocate(share, {"a": 1000, "b": 1000, "c": 1000}, total=total).targets

        assert all(result[name] >= previous[name] for name in share)
        previous = result


# --- which lines ------------------------------------------------------------------------------------------------


def lines(counts):
    """(line key, category) pairs: counts is {category: how many}."""
    return [(line_key(f"{category} line {i}"), category) for category, n in counts.items() for i in range(n)]


def test_the_selection_has_the_planned_number_of_lines_per_key():
    parsed = recipe(total=40)

    selection = select_lines(parsed, lines({"greetings": 100, "chat": 100, "trade": 50}))

    assert selection.allocation.targets == {"greetings": 10, "chat": 30}
    assert selection.available == {"greetings": 100, "chat": 100}
    chosen = {key for key, category in lines({"greetings": 100, "chat": 100, "trade": 50}) if key in selection.keys}
    assert len(selection.keys) == 40 == len(chosen)


def test_a_category_the_recipe_does_not_cover_is_never_selected():
    parsed = recipe()
    pool = lines({"greetings": 50, "chat": 150, "trade": 500})

    selection = select_lines(parsed, pool)

    assert not any(key in selection.keys for key, category in pool if category == "trade")


def test_children_count_towards_their_parent_key():
    parsed = parse_recipe({"categories": {"recruitment": {"weight": 1}, "chat": {"weight": 1}}})
    pool = lines({"recruitment/party": 30, "recruitment/guild": 30, "chat": 500})

    selection = select_lines(parsed, pool, )

    assert selection.available == {"recruitment": 60, "chat": 500}
    assert selection.allocation.targets == {"recruitment": 60, "chat": 60}


def test_uncategorized_lines_can_get_a_weight_like_any_category():
    parsed = parse_recipe({"categories": {UNCATEGORIZED: {"weight": 1}, "chat": {"weight": 1}}})
    pool = lines({UNCATEGORIZED: 10, "chat": 40})

    assert select_lines(parsed, pool).allocation.targets == {UNCATEGORIZED: 10, "chat": 10}


def test_the_same_inputs_give_the_same_lines_whatever_the_order():
    parsed = recipe(total=30)
    pool = lines({"greetings": 100, "chat": 100})
    shuffled = pool[:]
    random.Random(1).shuffle(shuffled)

    assert select_lines(parsed, pool).keys == select_lines(parsed, shuffled).keys


def test_another_seed_draws_other_lines():
    pool = lines({"greetings": 200, "chat": 200})

    assert select_lines(recipe(seed=1, total=40), pool).keys != select_lines(recipe(seed=2, total=40), pool).keys


def test_a_smaller_total_is_inside_a_bigger_one_so_a_learning_curve_is_nested():
    pool = lines({"greetings": 300, "chat": 300})
    sets = [select_lines(recipe(total=total), pool).keys for total in (40, 120, 400)]

    assert sets[0] < sets[1] < sets[2]


def test_the_draw_does_not_follow_the_validation_hash():
    # Training lines are the ones with a high validation bucket. A rank equal to that bucket would pick
    # "the lowest of the high ones" every time; the salted rank must be unrelated to it.
    texts = [f"line number {i}" for i in range(3000)]
    ranks = [line_rank(7, line_key(text)) for text in texts]
    buckets = [bucket(text) for text in texts]
    mean_r, mean_b = sum(ranks) / len(ranks), sum(buckets) / len(buckets)
    covariance = sum((r - mean_r) * (b - mean_b) for r, b in zip(ranks, buckets))
    spread = (sum((r - mean_r) ** 2 for r in ranks) * sum((b - mean_b) ** 2 for b in buckets)) ** 0.5

    assert abs(covariance / spread) < 0.1


def test_a_rank_is_a_number_below_one_that_depends_on_the_seed_and_the_line():
    key = line_key("おやすみ")

    assert 0 <= line_rank(1, key) < 1
    assert line_rank(1, key) == line_rank(1, key)
    assert line_rank(1, key) != line_rank(2, key)


def test_when_a_key_runs_out_the_lines_entering_at_the_same_time_stay_out_too():
    # 'chat' sorts before 'recruitment' and would otherwise take the one extra line of the tie.
    result = allocate(shares(chat=0.5, recruitment=0.5), {"chat": 500, "recruitment": 60})

    assert result.targets == {"chat": 60, "recruitment": 60}
    assert result.limited_by == "recruitment"


def test_only_the_ratio_of_the_sizes_matters_to_the_allocation():
    pool = {"a": 1000, "b": 1000}

    assert allocate({"a": Fraction(1), "b": Fraction(3)}, pool, total=40) == allocate(shares(a=0.25, b=0.75), pool, total=40)
