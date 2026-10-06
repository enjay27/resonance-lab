# 2026-10-06 — the season-1 labelling and translation become a versioned workflow

Runbook: `.claude/skills/season-data/SKILL.md`. Guides: `docs/labeling-guide.md`, `docs/translation-glossary.md` (both 1.0.1). Tools: PRs #74 (guides), #75 (translation), #76 (labeling), this one (skill).

## What happened (season 1, 2026-10-05)
- The maintainer's five chat logs (2026-09-30 … 10-05) gave 2,200 distinct lines; an agent labelled all of them (133 unsure) into the taxonomy plus four proposed labels
  (`coordination/boss_call`, `recruitment/closed`, `other/placeholder`, `non_japanese`); 2,171 are a judge sample (the 29 non-Japanese lines are out).
- 2,169 lines were translated by 11 agents (batches of ~200), reviewed by an independent agent on 220 of them (205 ok / 14 minor / 1 wrong), then 745 lines were revised by 8 agents
  when the maintainer supplied the OFFICIAL Korean class and dungeon names; the strict glossary check passes on all 2,169 lines. 406 distinct game terms, 14 with more than one rendering.
- The scratchpad scripts (validators, merge, final) became `translation_check.py`, `translation_assemble.py`, `scripts/translate_agents.py`; assembling the real batches with the new tool
  reproduces the old final file exactly. Labelling became `labeling_tools.py`, `scripts/label_lines.py`; exporting the real labels reproduces the counts and the judge sample's roots.

## Decisions of the maintainer
- Glossary rules are data (`configs/glossary/<season>.json`), the guide is the human version. `assemble` REFUSES when the glossary document changed since `prepare` (same guard for the labeling guide).
- Three PRs in order: translation tooling, labeling tooling, skill + note.
- Guild adverts need no translation (the guild is Korean). Season 1 translated them anyway (60 lines): the tool now skips them by default (`--include-guild`).

## Wrong turns (also in the skill)
- ワイプ -> 와이프 (wife); 継 -> 계승 instead of the official 계속; 森 healer vs 迷妄の森; ティナ dungeon vs character; `・` counted as Japanese by the first validator;
  guild lines translated against the maintainer's rule, and the docs said otherwise until the tools were written (a claim made from memory, not from the data: count the data before reporting it).
- The first dev/test split of the judge sample was seeded 70/30 (1,534/637); the tool's is a hash of the line (1,499/672) so a line keeps its side as the sample grows. Probes made on the old sample need re-running.

## Still open (nothing started)
- The four proposed categories are not in `configs/category_taxonomy.json` (a taxonomy + recipes PR; the label map then drops them).
- The translations are not training data (a feature: its own plan and an eval run). A full independent review of all 2,169 lines is optional.
- Glossary open questions (ギター, a bare 遺跡, 開拓局, ...) are in the glossary's section 7; Kev fine-tuning not planned in detail; the maintainer's GPU is still unknown.
- NOT VERIFIED on the maintainer's machine: `scripts/translate_agents.py` and `scripts/label_lines.py` ran only here (Linux), on made-up lines in tests and on the real season-1 files.
