<!-- Research by the maintainer, 2026-10-01 (uploaded in session). Kept verbatim as the plan of record for the model change. -->

Resonance Stream · translator model · research

# Translator Model Shortlist

Which base model should replace the TranslateGemma‑4B fine-tune for Japanese → Korean game chat, for a 4B default tier and a \~9B high tier chosen in the installer.

2026-10-01 shipped: TranslateGemma-4B + LoRA, Q4_K_M server: llama.cpp b8157 Vulkan status: desk research, not yet run on Kade's eval set

Recommendation

## Three model families go to Kade's eval set

No published benchmark measures Japanese → Korean chat for these models, so this page doesn't name one winner. It picks families: each gives a matching default and high-tier model, so one prompt format and one LoRA recipe serve both installer choices. Kade's `eval.py` (chrF, COMET, term accuracy, JP leakage) on the real Blue Protocol eval set should decide.

A · lead challenger

### Hy-MT2 (Tencent)

default

Hy-MT2-1.8B · Q4_K_M 1.08 GB

high

Hy-MT2-7B · Q4_K_M \~4.6 GB

A translation-only model, released 2026-05-21 under Apache 2.0. In the only game-text test found, the 1.8B beat TranslateGemma-4B on Japanese↔English (chrF 54.1 vs 46.8) at about half the VRAM. Our shipped llama.cpp build already loads its `hunyuan-dense` architecture. Its official README lists LLaMA-Factory fine-tuning.

open The 1.8B is smaller than the 4B tier you asked for. Its handling of Japanese slang and its Korean output are unmeasured.

B · baseline, lowest cost

### TranslateGemma (Google)

default

TranslateGemma-4B (today)

high

TranslateGemma-12B · \~7 GB est.

The current pipeline and prompt carry over unchanged. On WMT24++ en→ko the 12B scores MetricX 2.97 against 3.93 for the 4B (lower is better). The 4B should first be retrained on the prompt the app actually sends, which is a quality fix on its own (see below).

open 12B is larger than \~9B. Google's own human evaluation found a ja→en regression from mistranslated names.

C · newest general model

### Gemma 4 (Google)

default

Gemma 4 E4B · Q4_K_M \~5.0 GB

high

Gemma 4 12B (June 2026)

Apache 2.0. In the same game-text test, the smaller E2B scored highest of all models (chrF 61.6). That makes E4B a serious contender.

cost The llama.cpp server needs an upgrade (≥ April 2026, and ≥ June 2026 for the 12B). It also needs a new chat template with thinking turned off. The E4B file is about twice the size of today's model.

Found on the way

## The shipped model sees a prompt it was never trained on

In `resonance-lab` (`experiment/translategemma`), training uses LLaMA-Factory's `gemma3` template on dataset `bp_translation_nosystem`: the user turn is the raw Japanese line and nothing else, cut off at 128 tokens. At runtime, `translation_prompt` in `crates/core/src/text.rs` adds a long English instruction, a `[P0]` placeholder rule and a literal `<bos>`. The training lines never contain placeholders.

```
trained on
<start_of_turn>user
遺跡1Fから　29k↑　＠T1<end_of_turn>
<start_of_turn>model
```

```
app sends
<bos><start_of_turn>user
You are a professional Japanese (ja)
to Korean (ko) translator. ...
The input may contain placeholders
such as [P0], [P1] ...
遺跡1Fから　29k↑　＠T1<end_of_turn>
<start_of_turn>model
```

Whichever family wins, train on the exact prompt the app sends, with masked placeholders in some training lines. This also answers open item A4: LLaMA-Factory adds BOS once, so the literal `<bos>` very likely gives a double BOS that training never had. Kade should confirm before the app changes.

Default tier · runs on the Low tier and on CPU

## 4B-class candidates

| Model | License | Q4_K_M | Quality evidence | Speed evidence | llama.cpp b8157 | Fine-tune path |
| --- | --- | --- | --- | --- | --- | --- |
| TranslateGemma-4Bshipped · Jan 2026 · 34 layers | Gemma Terms | 2.6 GB | WMT24++ MetricX en→ko 3.93, en→ja 4.44. Game text ja↔en chrF 46.8. | 130 ms/line 3.64 GB VRAM | runs today | proven LLaMA-Factory, current recipe |
| Hy-MT2-1.8BMay 2026 · hunyuan-dense | Apache 2.0 | 1.08 GB | FLORES-200 all-pairs XCOMET 79.77 (Tencent). Game text ja↔en chrF 54.1, ja→en 62.8. Wrong-language output on low-resource pairs (not ja/ko). | 85 ms/line 1.77 GB VRAM | supported since ≤ b6076; skip the 1.25-bit file | documented LLaMA-Factory, LoRA |
| Gemma 4 E4BApr 2026 · 42 layers | Apache 2.0 | \~5.0 GB | No ja/ko text-translation score. Its smaller sibling E2B scored game-text chrF 61.6 (best of six). | E2B: 96 ms/line E4B not measured | upgrade needs Apr 2026+ | Unsloth guide LLaMA-Factory not checked |
| Qwen3.5-4BMar 2026 · DeltaNet hybrid | Apache 2.0 | 2.54 GiB | General model, 201 languages. No translation score found. Kade's earlier Qwen3 tries were replaced by TranslateGemma. | 22–47 tok/s on Vulkan | upgrade Vulkan kernels late; open Vulkan bugs | likely not checked |

The ms/line and VRAM figures come from one third-party test (Playto: RTX 4070 Ti, llama.cpp b8724, all layers on the GPU, Q4_K_M, Japanese↔English game lines). They show relative speed, not what users will see. Korean was not part of that test. Qwen3.5 Vulkan figures come from llama.cpp discussions on other hardware.

High tier · chosen in the installer

## \~9B-class candidates

| Model | License | Q4_K_M | Quality evidence | llama.cpp b8157 | Fine-tune path |
| --- | --- | --- | --- | --- | --- |
| Hy-MT2-7BMay 2026 · hunyuan-dense | Apache 2.0 | \~4.6 GB | FLORES-200 all-pairs XCOMET 86.89, on par with Gemma 4 31B (86.84) per Tencent. COMET-22 FLORES 0.8747 vs TranslateGemma-12B 0.8732. Behind it on an LLM-judged WMT26 score (60.51 vs 71.19). The last two are from a competitor's model card. | supported | documented same recipe as 1.8B |
| TranslateGemma-12BJan 2026 | Gemma Terms | \~7 GB est. | WMT24++ MetricX en→ko 2.97, en→ja 3.82; beats base Gemma 3 27B on both. Human MQM en→ko 4.6. | supported | proven same recipe as 4B |
| Gemma 4 12BJun 2026 · gemma4_unified | Apache 2.0 | not checked | No published text-translation score. Community reports favour Gemma 4 over Qwen3.5 for non-English text, mostly for other languages and larger sizes. | upgrade needs Jun 2026+ | not checked |
| Qwen3.5-9BMar 2026 | Apache 2.0 | \~6 GB | MMMLU 81.2 (general multilingual knowledge, not translation). No translation score found. | upgrade | not checked |

Sizes marked "est." or "\~" come from community builds or from the same model's earlier version, not from the exact file we would ship. The current GPU tiers (-ngl 12 / 24 / 32 / all) assume TranslateGemma-4B's 34 layers. A new model needs them re-mapped.

Hard filters

## Dropped, and why

**HY-MT1.5 1.8B / 7B, Hunyuan-MT-7B**license The Tencent HY Community License excludes South Korea, where our users are. Hy-MT2 was relicensed to Apache 2.0, so it is unaffected.

**Seed-X-PPO-7B**quality ByteDance says its quantized builds are unstable and recommends beam search, which llama-server lacks.

**EXAONE 4.x / 4.5**license Non-commercial only. The small 1.2B doesn't cover Japanese.

**Kanana-2, Kanana Nano**size · license Kanana-2 is a 30B-A3B MoE. Nano is CC-BY-NC.

**HyperCLOVA X SEED 0.5–3B**no evidence Strong in Korean, but no Japanese support or translation results found.

**Qwen3.6 / Qwen3.8**size Only 27B and larger. "Qwen3.8-9B" is a community distill, not an official release.

Next step

## How to choose: Kade's eval set decides

1. **Zero-shot round.** Run `scripts/eval.py` on `bp-eval-dataset.jsonl` for the shipped fine-tune, Hy-MT2-1.8B and 7B, Gemma 4 E4B and TranslateGemma-12B, each with its own chat template. Add placeholder lines and a ms/line column. A model that is close to the shipped fine-tune without training is a strong base.
2. **Fine-tune the top family.** Train with the app's exact prompt, including placeholders, at both sizes. Retrain TranslateGemma-4B the same way as the control, so the comparison is fair.
3. **Speed check on Windows.** Run Q4_K_M through llama-server Vulkan on a low-end GPU and on CPU only, at our `-c 1536 -b 64` settings.
4. **Then the app change, in its own PR:** prompt template in `crates/core`, GPU tier mapping, a second model in the gist plus a model choice in the installer, and a server upgrade if family C wins.

Sources

Read during this research on 2026-10-01. huggingface.co and arxiv.org were blocked from this session, so model-card and paper figures come through search summaries and should be checked against the originals before a final decision.

1. [enjay27/resonance-lab](https://github.com/enjay27/resonance-lab), branches `main`, `experiment/qwen3.5`, `experiment/translategemma`: training recipe and `eval.py`
2. [TranslateGemma Technical Report](https://arxiv.org/pdf/2601.09012) (arXiv 2601.09012): Table 4 MetricX per language, MQM; [Google announcement](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/)
3. [Tencent-Hunyuan/Hy-MT2](https://github.com/Tencent-Hunyuan/Hy-MT2) (README, LICENSE.txt: Apache 2.0); [Hy-MT2 report](https://arxiv.org/html/2605.22064v2) (arXiv 2605.22064); [Tencent relicensing post](https://x.com/TencentHunyuan/status/2059249996256711150)
4. [Index-Translate model card](https://huggingface.co/IndexTeam/Index-Translate-35B-A3B-preview): COMET-22 and WMT26 judge figures (competitor source)
5. [Playto: six local models measured on game text](https://playto.dev/blog/best-local-llm-for-translation/); [Hy-MT2 1.8B in practice](https://playto.dev/blog/hy-mt2-in-practice/)
6. [HY-MT1.5 License.txt](https://huggingface.co/tencent/HY-MT1.5-1.8B/blob/main/License.txt): territory excludes EU, UK, South Korea
7. [llama.cpp PR #22836](https://github.com/ggml-org/llama.cpp/pull/22836) (STQ1_0, open, ARM only); [Hunyuan dense GGUFs built with b6076](https://huggingface.co/bartowski/tencent_Hunyuan-7B-Instruct-GGUF)
8. [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4); [Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4); [E4B config.json](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json)
9. [Qwen3.5 small models](https://venturebeat.com/technology/alibabas-small-open-source-qwen3-5-9b-beats-openais-gpt-oss-120b-and-can-run); [llama.cpp Vulkan performance thread](https://github.com/ggml-org/llama.cpp/discussions/10879); [Vulkan Qwen3.5 issue #27237](https://github.com/ggml-org/llama.cpp/issues/27237)
10. [Seed-X-PPO-7B GGUF notes](https://huggingface.co/mradermacher/Seed-X-PPO-7B-GGUF); [EXAONE 4.5](https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B); [Kanana-2](https://github.com/kakao/kanana-2/); [HyperCLOVA X SEED](https://huggingface.co/naver-hyperclovax)

Resonance Stream · research only, no app code changed.