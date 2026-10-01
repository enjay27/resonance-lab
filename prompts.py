"""The instruction text put in front of each line, per model family and direction. Pure, no torch.

The prompt is part of what a model is trained on: preprocess.py writes it into the training rows and
lf_tools.chat_messages uses it for eval, so both read it from here. resonance-stream must send the same
text (stream-contract.md, section 1).
"""

# TranslateGemma: the instruction its own chat template builds from language codes, newline, the line.
# Copied from the shipped model's training file (maintainer's sample, 2026-10-01).
_TG = (
    "You are a professional {src} ({s}) to {dst} ({d}) translator. Your goal is to accurately convey the meaning "
    "and nuances of the original {src} text while adhering to {dst} grammar, vocabulary, and cultural sensitivities.\n"
    "Produce only the {dst} translation, without any additional explanations or commentary. "
    "Please translate the following {src} text into {dst}:\n{{text}}"
)

# Hy-MT2: its documented "Default Translation" prompt (README); the ko-ja variant only changes the language name.
_HY = (
    "Translate the following text into {dst}. Note that you should only output the translated "
    "result without any additional explanation:\n\n{{text}}"
)

STYLES = {
    "translategemma": {
        "ja-ko": _TG.format(src="Japanese", s="ja", dst="Korean", d="ko"),
        "ko-ja": _TG.format(src="Korean", s="ko", dst="Japanese", d="ja"),
    },
    "hy": {
        "ja-ko": _HY.format(dst="Korean"),
        "ko-ja": _HY.format(dst="Japanese"),
    },
}

# Which style a LLaMA-Factory template is trained with.
TEMPLATE_STYLES = {"gemma3": "translategemma", "hy_dense_1_8b": "hy", "hy_dense_7b": "hy"}


def build_prompt(style, direction, text):
    """`text` with the instruction of `style` for `direction` ("ja-ko" / "ko-ja"); style None = the line alone."""
    if style is None:
        return text
    if style not in STYLES:
        raise ValueError(f"unknown prompt style {style!r}; choose one of: {', '.join(sorted(STYLES))}")
    if direction not in STYLES[style]:
        raise ValueError(f"unknown direction {direction!r}; choose one of: {', '.join(sorted(STYLES[style]))}")
    return STYLES[style][direction].format(text=text)


def style_for_template(template):
    if template not in TEMPLATE_STYLES:
        raise ValueError(f"no prompt style for template {template!r}; known: {', '.join(sorted(TEMPLATE_STYLES))}")
    return TEMPLATE_STYLES[template]
