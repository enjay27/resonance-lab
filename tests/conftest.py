import json

import pytest


@pytest.fixture
def write_jsonl(tmp_path):
    """Writes rows (dicts, or raw strings for malformed lines) to a JSONL file and returns its path."""

    def write(name, rows):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write((row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)) + "\n")
        return str(path)

    return write


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
