import pytest

import prompts

# The TranslateGemma instruction exactly as the shipped model's training file has it (maintainer's sample,
# 2026-10-01); the line follows after one newline.
TG_JA_KO = (
    "You are a professional Japanese (ja) to Korean (ko) translator. Your goal is to accurately convey the meaning "
    "and nuances of the original Japanese text while adhering to Korean grammar, vocabulary, and cultural sensitivities.\n"
    "Produce only the Korean translation, without any additional explanations or commentary. "
    "Please translate the following Japanese text into Korean:\n"
)
TG_KO_JA = (
    "You are a professional Korean (ko) to Japanese (ja) translator. Your goal is to accurately convey the meaning "
    "and nuances of the original Korean text while adhering to Japanese grammar, vocabulary, and cultural sensitivities.\n"
    "Produce only the Japanese translation, without any additional explanations or commentary. "
    "Please translate the following Korean text into Japanese:\n"
)


def test_translategemma_prompt_is_the_one_the_shipped_model_trained_on():
    assert prompts.build_prompt("translategemma", "ja-ko", "杖@2募集") == TG_JA_KO + "杖@2募集"
    assert prompts.build_prompt("translategemma", "ko-ja", "법사@2 모집") == TG_KO_JA + "법사@2 모집"


def test_hy_prompt_is_hys_documented_default_translation_prompt():
    assert prompts.build_prompt("hy", "ja-ko", "杖@2募集") == (
        "Translate the following text into Korean. Note that you should only output the translated "
        "result without any additional explanation:\n\n杖@2募集"
    )
    assert prompts.build_prompt("hy", "ko-ja", "법사").startswith("Translate the following text into Japanese.")


def test_no_style_leaves_the_line_alone():
    assert prompts.build_prompt(None, "ja-ko", "杖@2募集") == "杖@2募集"


def test_unknown_style_or_direction_is_refused_with_the_known_ones():
    with pytest.raises(ValueError, match="translategemma"):
        prompts.build_prompt("nope", "ja-ko", "x")
    with pytest.raises(ValueError, match="ja-ko"):
        prompts.build_prompt("hy", "en-ko", "x")


def test_the_style_follows_the_llamafactory_template():
    assert prompts.style_for_template("gemma3") == "translategemma"
    assert prompts.style_for_template("hy_dense_1_8b") == "hy"
    assert prompts.style_for_template("hy_dense_7b") == "hy"
    with pytest.raises(ValueError, match="gemma3"):
        prompts.style_for_template("nope")
