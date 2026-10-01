# Gates for resonance-lab. `just check` runs every gate that runs without a GPU.
# See CLAUDE.md for which gate applies to which part of the repo.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

default: check

# Every gate that runs on any OS. The model part (train/eval/fix_metadata)
# needs CUDA and the full requirements.txt; it has no automated gate.
check: lint data-check

# Lint (pyflakes + syntax), whole repo. Formatting is not enforced yet.
lint:
    ruff check .

# Formatting: not part of `check` yet (CLAUDE.md). Shows what would change.
fmt-check:
    ruff format --check .

# data part: raw-log validation, preprocessing, train/val split. CPU only.
data-check:
    pytest

# Gate tooling, CPU only (no torch).
setup-dev:
    pip install -r requirements-dev.txt
