import pytest

from valsplit import bucket, is_validation


def test_bucket_is_pinned_so_the_split_never_moves_between_runs_or_machines():
    # sha1-based on purpose: Python's hash() is salted per process and would reshuffle the validation set every run.
    assert bucket("スカイ") == pytest.approx(bucket("スカイ！"))  # same line after normalising
    assert bucket("スカイ") == pytest.approx(0.67955, abs=1e-5)  # golden value of the sha1 scheme: change it and every validation set changes


def test_variants_of_one_line_land_on_the_same_side():
    for line in ("おやすみ", "遺跡1F", "杖@2募集", "ウルト溜まった"):
        assert is_validation(line, 0.5) == is_validation(line + "！", 0.5) == is_validation(" " + line.upper() + "\n", 0.5)


def test_the_fraction_decides_roughly_how_many_lines_are_validation():
    lines = [f"line number {i} 募集" for i in range(4000)]

    share = sum(is_validation(line, 0.05) for line in lines) / len(lines)

    assert 0.035 < share < 0.065


def test_a_line_keeps_its_side_when_the_dataset_grows():
    lines = [f"line {i}" for i in range(500)]
    before = {line: is_validation(line, 0.05) for line in lines}

    after = {line: is_validation(line, 0.05) for line in lines + [f"new {i}" for i in range(500)]}

    assert all(after[line] == side for line, side in before.items())


def test_zero_fraction_means_no_validation_and_one_means_all():
    assert not is_validation("line", 0)
    assert is_validation("line", 1)


def test_lines_with_only_punctuation_still_get_a_side():
    assert is_validation("！！！", 1)
    assert not is_validation("！！！", 0)
