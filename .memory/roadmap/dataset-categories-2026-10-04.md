# Message categories and the first recommended dataset mix — 2026-10-04

**Status: a FIRST GUESS, set without data (maintainer, 2026-10-04: "this is logic and pipeline design, real data is not needed yet").** Nothing has been categorised or trained.
Files: `configs/category_taxonomy.json` (the categories + the description a classifier will be shown), `configs/datasets/balanced-v1.json` (weights by root, needs only Gate 1),
`configs/datasets/balanced-v1-detailed.json` (roots split into children, needs Gate 1-A). `tests/test_category_taxonomy.py` pins that the recipes and the taxonomy agree.

## The categories (roots = Gate 1's choice; children = Gate 1-A's; a deeper gate can add a level)
| root | children | what it is |
|---|---|---|
| `social` | greeting, thanks | polite formulas: おはよう, ありがとう, 乙 |
| `chat` | casual, reaction | everyday talk; reaction = 草, ww, すごい |
| `game` | combat, progress, system, market | statements about how the game works (classes, quests, bosses, patches, bugs, the open market) |
| `question` | game, other | a player asks for information or help |
| `coordination` | | real-time call-outs while playing: 準備OK, 待って, 集合, 回復お願い |
| `recruitment` | party, guild | looking for party / raid members, or guild members |
| `bot` | system, ad | automated messages (maintainer: translated, so a category, not a drop) |
| `spam` | wall, flood | pasted walls, floods, repetition (recruitment walls included) |
| `other` | | unreadable, symbols only, fragments: **never trained on** (in no recipe) |

Single label per message; overlaps are settled by the descriptions: a *question* about the game is `question`, a *statement* about it is `game`; a laughing reaction inside a conversation is `chat/reaction`.

## The weights (percent, a root's share = weight / total) and why
social 8 · chat 26 (casual 18, reaction 8) · game 27 (combat 9, progress 8, system 4, market 6) · question 12 · coordination 10 · recruitment 12 (party 8, guild 4) · bot 3 · spam 2.
- **Not the natural mix.** Recruitment is the most templated, most repetitive text: it is held to 12% (a big share teaches memorisation, and `dedup` leaves little unique anyway).
- **Lifted:** `game` (27%): game terms are the weakest metric (term accuracy 1/21 on the first TG eval); `question` and `coordination`: short, varied, imperative or interrogative phrasing the translator meets constantly in a party.
- **`chat/reaction` 8%**: the failures `草` -> `고블린` and `ww` left as `ww` are reactions; enough to learn them, small because there are few distinct ones.
- **`social` 8%**: formulaic, saturates fast. **`game/market` 6%**: price and item phrasing, needs the item terms; the first weight to lower if the data shows little market talk.
- **There is no `trade` category (maintainer, 2026-10-04):** Star Resonance has no trade between players, only the open market where players list items. Chat about it is market talk (`game/market`, a statement) or a question (`question`: "what should I sell?").
- **`bot` 3%, `spam` 2%** with `keep: ["recruitment spam"]` (maintainer: allow walls, trusting Gate 1, in a small share). `cutoff_len` is 256: a wall's Korean can be cut mid-sentence; check `inspect_pair.py`.

## What to know before tuning
- **Hierarchy:** a recipe key covers its descendants, never its ancestors. A line categorised only as `chat` (Gate 1 without 1-A) is NOT picked by `chat/casual`: use `balanced-v1` until Gate 1-A exists. A test pins that the detailed recipe keeps each root's weight.
- **The bottleneck line of the `preprocess.py` report names the category that limits the dataset size** (e.g. `reaction` runs out long before `chat/casual`): lower its weight or accept a smaller dataset, per run. Nothing here is a measurement.
- Compare recipes in the notebook / MLflow by the `recipe` column at a fixed learning rate; the validation file is the same for every recipe, so eval loss is comparable. A data change is a feature: retrain and run `eval.py`.
- **Not decided:** whether the eval set's `category` field should use these root names (then `eval.py` reports chrF / term accuracy per category for free); whether `bot` lines even reach the app's log.
