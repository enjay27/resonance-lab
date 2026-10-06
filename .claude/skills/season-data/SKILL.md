---
name: season-data
description: Label a season's raw Blue Protocol chat lines and translate them Japanese -> Korean with parallel agents, following docs/labeling-guide.md and docs/translation-glossary.md. Use when new chat logs arrive (a new season, a dataset refresh), when the guides or the glossary change, or when asked to (re)label or (re)translate lines or to build the Gate 1 judge sample.
---

# season-data

The runbook of the work done for season 1 (2026-10): 2,200 distinct lines labelled, 2,169 translated. The mechanics are tested repo code; **the judgement is in the two
versioned guides** and in the agents that read them. Nothing here trains a model; a change to the training data is a feature with its own PR (CLAUDE.md).

## Rules that hold the whole way

- **Plan first** (workflow-control): the first turn of a season is a plan, not edits. Anything that edits a guide, the taxonomy, the label map, the glossary JSON or code is a PR.
- **No data in git.** The chat lines, labels and translations live in `data/labeling/<season>/` and `data/translation/<season>/` (gitignored). Only terms and rules (the guides) are committed; examples in them are made up.
- **The guides are the single source.** Agents are briefed with `docs/labeling-guide.md` / `docs/translation-glossary.md` (the tools copy them into `round<N>/brief.md`); the rules the checks run are `configs/label_map.json` / `configs/glossary/<season>.json`. Change a guide first, bump its version, then the data file; tests keep them in step.
- **STALE batches are refused.** `assemble` fails when the guide is not the version the batches were prepared with. Fix = prepare again (or restore the guide). Never edit `run.json` to get past it.
- **Report what ran.** Agents and checks that did not run are `NOT VERIFIED`; say so, never imply a review that was not done (a sample review is not a full review).

## 0. Start

1. `git checkout -B claude/<season>-data origin/main`. Read both guides' version headers and open-questions tables: answer what you can with the maintainer before starting, not after 2,000 lines.
2. Pick the season name (`S2`, ...). If the game's names changed (new class, dungeon, raid stage), update the glossary document (section 3/4), `configs/glossary/<season>.json` (copy the last season's, set `season` and `doc_version`) and the label guide first, in their own PR, and let it merge.
3. The raw logs: `data/raw/raw_translated_logs.jsonl` after `python scripts/fetch_data.py`, or the app's `dataset_<CHANNEL>.jsonl` files.

## 1. Label

```
python scripts/label_lines.py prepare  --season S2 --raw <log.jsonl> [more logs] [--size 300]
   -> data/labeling/S2/lines.jsonl (distinct lines), round1/in/<batch>.jsonl, round1/brief.md, run.json
```
Start **one agent per batch file, all in parallel** (Agent tool, general-purpose, background). Prompt (fill the paths):

> You label chat lines of the online game Blue Protocol: Star Resonance (Japanese server); the chat is from that game and mostly about its content (raids, dungeons, party recruitment, boss calls).
> Read `<dir>/round1/brief.md` completely first: it is the labeling guide and the output format. Label every row of `<dir>/round1/in/<batch>.jsonl` and write
> `<dir>/round1/out/<batch>.jsonl`. When finished run `python scripts/label_lines.py check --season S2 --batch <batch>` and fix every problem it lists. Report only: done, and how many lines you marked unsure.

```
python scripts/label_lines.py check    --season S2                 # every batch; exit 1 names what is missing or wrong
python scripts/label_lines.py assemble --season S2                 # labels.jsonl, judge-sample.jsonl, labels.meta.json + the counts table
```
Then: a **human reads the `unsure` lines** (`labels.jsonl`); corrections are edits of `labels.jsonl`, and `python scripts/label_lines.py export --season S2` rebuilds the judge sample; `python scripts/label_lines.py report --season S2` prints the counts table again.
Paste the counts table into section 4 of the labeling guide (PR, version bump). The judge sample is for `categorize.py --probe` / `compare_judges.py`: copy it to `data/eval/gate1-sample.jsonl` yourself; never overwrite a hand-labelled sample. A label the taxonomy lacks goes through `configs/label_map.json` (guide section 2 says "proposed"); making it real is a taxonomy PR that also removes it from the map.

## 2. Translate

```
python scripts/translate_agents.py prepare  --season S2 --labels data/labeling/S2/labels.jsonl [--size 200] [--include-guild]
   -> data/translation/S2/round1/in/<batch>.jsonl, brief.md, run.json (guild adverts, non_japanese and placeholders are skipped by default)
```
One agent per batch, in parallel. Prompt:

> You translate chat lines of the online game Blue Protocol: Star Resonance (Japanese server) into Korean for Korean players of the same game. The chat is from that game and is mostly about its content
> (raids, dungeons, party recruitment with slot notation, boss and channel calls, raid mechanics). Read `<dir>/round1/brief.md` completely first: it is the glossary and the rules; use its Korean names exactly.
> Translate every row of `<dir>/round1/in/<batch>.jsonl` and write `<dir>/round1/out/<batch>.jsonl`. When finished run `python scripts/translate_agents.py check --season S2 --round 1 --batch <batch>`
> and fix every problem it lists. Report only: done, and the lines you flagged.

```
python scripts/translate_agents.py check    --season S2           # the latest round; --round N, --batch B
python scripts/translate_agents.py assemble --season S2           # final.jsonl, terms.tsv, final.meta.json; applies the glossary's fixes; exit 1 lists glossary problems
python scripts/translate_agents.py revise   --season S2           # round N+1: the lines that break the glossary, with `prev`; --all = every line
python scripts/translate_agents.py report   --season S2           # counts, flagged lines, terms with more than one rendering
```
Loop `revise` -> agents (same prompt, the new `round<N>/brief.md`) -> `check` -> `assemble` until `assemble` exits 0. The latest round wins per line.

**Independent review, always:** a separate agent that did not translate reads ~10% of `final.jsonl` (spread over categories, all flagged lines) and reports ok / minor / wrong with the reason. Fix the **pattern** it finds everywhere
(a glossary `fixes` entry or `required`/`banned` rule + a revision round), not only the sampled lines. Season 1: 205 ok / 14 minor / 1 wrong of 220. A full review of every line is a separate, optional pass.

**If the glossary changes mid-season** (the maintainer answers an open question): edit the document (version bump), the JSON (`doc_version`), then `prepare` the revision again -- `assemble` refuses the old batches until you do.

## 3. Deliver and close the season

1. Send the maintainer `final.jsonl` (`original, translated, category, channel, flag, terms`) and `terms.tsv`; say how many lines, how many were revised, flagged, the terms with more than one rendering, and the open questions. Do not send guild or chat lines anywhere else.
2. Docs PR: replace the glossary's *Observed terms* table (do not append), answer/replace the open-questions table, bump both guides (patch / minor / major per their *Updating* sections), add changelog rows, set `doc_version` in `configs/glossary/<season>.json` and `configs/label_map.json`.
3. `.memory/sessions/<date>-<topic>.md` with what went wrong this time; `MEMORY.md` *Now*; then `just check` and the PR (one at a time, CLAUDE.md).
4. Using the translations as training data is **not part of this runbook**: it changes the model (a feature, its own plan and eval run).

## 4. Wrong turns of season 1 (do not repeat)

- **ワイプ became 와이프** (wife) in the first batches: the glossary says 전멸; `fixes` and `banned` now catch it. Check a new term list for words that mean something else in Korean.
- **継 was guessed 계승**; the official name is 계속 (stage of 幻夢レイド). Guessing a name is fine only if it is flagged and replaced when the official one arrives: keep `terms` honest so a replacement is one table lookup.
- **迷妄の森 vs 森 (the healer subclass)** and **ティナ the dungeon vs the character**, **火力 role vs damage**: context rules in the glossary; the checks can only test the dungeon forms, so the reviewer must read these.
- **Kana punctuation counted as Japanese:** `・` (U+30FB) and `ー` are kaomoji, not Japanese; agents swapped them for other dots. The check allows them; `assemble` restores the source's `・`.
- **A rule of the maintainer that was not applied:** "guild lines need no translation" -- the first batches translated them anyway, and the docs claimed otherwise. `prepare` now skips them by default; read `run.json` `counts.skipped` after prepare.
- **Docs and data drift:** the glossary document, its JSON and the label map each carry the guide version they were written for; a test fails when they differ. Update all three together.
- **Do not claim a gate passed that did not run** (CLAUDE.md): this runbook's checks are code checks plus an agent review of a sample, not a proof of translation quality.

## Files

| what | where |
|---|---|
| guides (versioned, human) | `docs/labeling-guide.md`, `docs/translation-glossary.md` |
| rules as data | `configs/label_map.json`, `configs/glossary/<season>.json`, `configs/category_taxonomy.json` |
| logic (pure, tested) | `labeling_tools.py`, `glossary.py`, `translation_check.py`, `translation_assemble.py` |
| CLIs | `scripts/label_lines.py`, `scripts/translate_agents.py` |
| a season's work (gitignored) | `data/labeling/<season>/`, `data/translation/<season>/` (`run.json` = what the batches were made with; `*.meta.json` = what the result was made with) |
