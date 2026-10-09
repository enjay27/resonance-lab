# .memory — which file takes what

State (what is next, what is written but not yet verified) lives in the organization
Project "Resonance", as issues with `cmd:` labels; read it as `STATUS.md` on the `status`
branch. There is no `MEMORY.md`. Everything else lives here, one folder per kind of note.

| folder | takes | rule |
|---|---|---|
| `active-issues/` | things **not to trust yet**: code written but never run on the GPU machine, docs that contradict the code, known bugs, mismatches with resonance-stream | one file per topic; **keep each under ~150 lines** — close items by deleting them, the history is in `sessions/` and git |
| `roadmap/` | what is next, with the check that proves it done; facts with a number | one file per workstream |
| `reference/` | verified facts to look up, not to act on: formats, schemas, what a tool prints. Placeholders only — no player data | one file per topic; the tested code or test it mirrors is named in the file |
| `sessions/` | dated write-ups `YYYY-MM-DD-<topic>.md`, **including wrong turns** | append-only; flat folder |

Write the note in the same commit as the change it describes.
