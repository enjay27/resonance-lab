# Labeling guide — which category is a chat line

**Version:** 1.0.1 · **Season:** S1 (2026-10)

How a Japanese *Blue Protocol: Star Resonance* chat line gets its category. Used by whoever labels the Gate 1 sample by hand, by labelling agents, and as the
reference when the judge's answers are read. The categories themselves live in `configs/category_taxonomy.json` (what the judge is shown); this guide says
how to apply them and records the season-1 decisions. Terms and rules only — no chat lines (no data in git, `tests/test_docs.py`).

Companion: [`translation-glossary.md`](translation-glossary.md) (how a line is translated). Season procedure: [Updating for a new season](#updating-for-a-new-season).

## 1. The rules of labelling

1. **One label per line**, the most specific path that is sure (`recruitment/party`), else its root (`recruitment`). A deeper gate may split a root later; a line is never
   given two labels.
2. **Label what the line does, not which words it contains.** `ティナ` in a recruitment line is recruitment; in a question it is a question.
3. **Overlaps are settled in this order:** automated (`bot`) → not readable (`other`) → recruiting (`recruitment`) → asking (`question`) → calling out while playing
   (`coordination`) → statement about the game (`game`) → polite formula (`social`) → the rest (`chat`).
4. **A question about the game is `question`; a statement about it is `game`.** A laughing reaction inside a conversation is `chat/reaction`.
5. **Unsure?** Label your best guess and mark the line `unsure` — never leave it unlabelled, never drop it. Unsure lines are reported separately and are the first ones to re-read.
6. **The channel is a hint, not a rule.** In season 1 every recruitment, guild, closed-recruitment and boss-call line came from `WORLD`; `PARTY` lines were chat, coordination,
   social, questions and game talk. A `WORLD` line that looks like conversation is still `chat`.
7. **The eval set is labelled by hand and no classifier may label it** (Gate 1's accuracy is measured on it).

## 2. The categories

Paths are the ones of `configs/category_taxonomy.json`. "Proposed" marks a season-1 category that is **not** in the taxonomy yet; for the judge it maps to the path in the last column.

| Category | What it is | Typical forms (made up) | Judge sees |
|---|---|---|---|
| `social` | Polite formulas: greetings, farewells, thanks, apologies, congratulations | おはよう, お疲れ様でした, ありがとうございました, おめでとう | `social` |
| `social/greeting` | Hello and goodbye | こんばんは, お先に失礼します, また明日 | |
| `social/thanks` | Thanks, apologies, congratulations | ありがとう, ごめんなさい, 乙 | |
| `chat` | Everyday conversation that is none of the others | what someone is doing, jokes, opinions, small talk | `chat` |
| `chat/casual` | Ordinary talk | | |
| `chat/reaction` | Very short emotional reaction or laughter | 草, ww, すごい, ナイス, えぐ | |
| `game` | A statement about how the game works or what to do in it | a boss mechanic explained, a build opinion, a patch remark | `game` |
| `game/combat` | Classes, skills, builds, damage, rotations, stats | | |
| `game/progress` | Quests, levels, maps, dungeons, bosses, gear goals, farming, events | | |
| `game/system` | Maintenance, patches, bugs, lag, servers, login | | |
| `game/market` | The open market (there is **no** trade between two players): prices, what sells, that something was listed or bought | | |
| `question` | The player asks for information or help and expects an answer | どうやって〜？, 〜はどこですか, 教えてください | `question` |
| `question/game` | A question about the game | | |
| `question/other` | Any other question, incl. to a person and rhetorical ones | | |
| `coordination` | Real-time call-outs while playing together: ready checks, waiting, positions, asking for a heal or revive, starting a run | 準備OK, ちょっと待って, 集合, 回復お願い | `coordination` |
| `coordination/boss_call` (proposed) | A world-boss or channel call: a boss name with a channel number and often a timer or an HP percentage | boss name + `20ch` | `coordination` |
| `recruitment` | Someone looks for people | | `recruitment` |
| `recruitment/party` | Members for a party, dungeon, raid or boss run, often with slot notation and requirements | `墓M6 @T1H1D2`, `継NM @H1`, 〆 lines are the next row | |
| `recruitment/closed` (proposed) | A recruitment that is over: `〆`, "full", an apology to those who did not fit | 〆です, 溢れた方すみません | `recruitment` |
| `recruitment/guild` | A guild advertises itself, or a player looks for a guild | ギルメン募集, 初心者歓迎 | `recruitment` |
| `bot` | Automated messages, not a person typing | | `bot` |
| `bot/system` | Notices produced by the game or a tool: loot, queue pops, timers, status lines | | |
| `bot/ad` | Automated advertisement posts | | |
| `spam` | Floods and walls | | `spam` |
| `spam/wall` | A long pasted wall, typically a recruitment or advertisement text with several IDs | | |
| `spam/flood` | Repeated characters or the same short text posted over and over | | |
| `other` | Does not fit, or cannot be read: symbols only, emoticons only, cut-off fragments, a lone character | `P`, `？`, `123` | `other` |
| `other/placeholder` (proposed) | The app's own stand-in for content it could not send | `[이모지]`, `[스티커]` | `other` |
| `non_japanese` (proposed) | A line in another language (in season 1 almost always Korean); **not translated**, not judge-ready | | left out of the judge sample |

Notes on the proposed categories: they are not in `configs/category_taxonomy.json` or any recipe yet. Adding one there needs a PR of its own (taxonomy + recipes + `tests/test_category_taxonomy.py`),
and the version of this guide rises with it.

## 3. Decisions of season 1

- **Guild lines are labelled `recruitment/guild` and, by default, not translated** (the maintainer's guild members are Korean; the translation tooling skips them). Season 1 translated them anyway: 60 lines.
- **Long advertisement walls** were labelled by what they recruit (guild or party), not `spam`; no line was labelled `spam` in season 1. Whether a wall should be `spam/wall` is open.
- **`bot` lines are real chat lines of the app's log** (timekeeper bots announcing the hour, weekend-reminder bots): a category, not a drop.
- **Slot notation is recruitment:** `@T1`, `D3H1`, `T1H3D12`, `↑` (minimum score) and a dungeon/raid name. The judge missed this until the wording of its question taught it (`configs/judge_prompts/v2.json`).
- **Short remarks inside a raid** (`PARTY`): a reaction is `chat/reaction`, an instruction to someone is `coordination`, an explanation of a mechanic is `game`, a request for a fact is `question`.
- **Boss calls are not recruitment:** a line that only names a boss and a channel tells where it is; it recruits nobody.
- **`other` is never trained on**; so a line in doubt between `other` and `chat` is labelled `chat` with `unsure` unless it is really a lone symbol or fragment.

## 4. What season 1 produced

Labels written by the labelling agent for the five logs of 2026-09-30 … 2026-10-05, 2,200 distinct lines (a human reviews the unsure ones):

| Category | Lines | | Category | Lines |
|---|---|---|---|---|
| `recruitment/party` | 1,296 | | `recruitment/guild` | 60 |
| `chat` | 356 | | `coordination/boss_call` | 43 |
| `coordination` | 188 | | `question` | 42 |
| `social` | 111 | | `game` | 35 |
| `non_japanese` | 29 | | `recruitment/closed` | 19 |
| `other` | 15 | | `bot` | 4 |
| `other/placeholder` | 2 | | | |

133 lines are `unsure`. For the judge, 2,171 lines are ready (the `non_japanese` ones left out; proposed categories mapped to the *Judge sees* column; 70 / 30 split into dev / test).
Not a measurement of the judge: it is the labelled set the judges are scored on (`scripts/compare_judges.py`).

What the first measurements showed, to read a judge's score against: the rule baseline (`categorizer.py`) puts a recruitment line in `recruitment` with about 100 % precision but its `game`
rule is about 10 % precise; Kev-0.8B's argmax accuracy rose from 29 % to 60 % when the wording of its question changed (Kev-9B not measured yet).

## Updating for a new season

1. Branch from `main`. New chat logs → label a fresh sample (do not reuse the lines a judge was tuned on).
2. A new kind of line → add a category to `configs/category_taxonomy.json` **and** to section 2 here; `tests/test_docs.py` fails until the guide names every taxonomy path.
3. Replace section 4 with the new season's counts; keep the decisions of section 3 that still hold and date the ones that change.
4. Bump the version: patch for wording, minor for a new category or rule, major for a new season. Add a changelog row.
5. `just check`.

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.1 | 2026-10-06 | Corrected the guild decision: season 1 translated the guild lines (the guide said it did not); the tooling skips them by default. |
| 1.0.0 | 2026-10-05 | First versioned guide: the taxonomy's 25 categories plus the four proposed ones (`coordination/boss_call`, `recruitment/closed`, `other/placeholder`, `non_japanese`); season-1 decisions and counts. |
