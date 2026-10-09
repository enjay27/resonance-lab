"""What the fake tools check before they write placeholder output: the inputs the real tool would need.

Each failed check prints `[MOCK CONTRACT] <tool>: <what is wrong>` and exits 2, which stops the pipeline at that
stage, so a stage that stops leaving what the next real tool reads fails the mock run.
"""

import os
import sys

import yaml

GGUF_MAGIC = b"GGUF"


def fail(tool, message):
    print(f"[MOCK CONTRACT] {tool}: {message}", file=sys.stderr)
    sys.exit(2)


def require(tool, condition, message):
    if not condition:
        fail(tool, message)


def load_yaml(tool, path, required_keys):
    require(tool, os.path.isfile(path), f"config {path} does not exist (paths are relative to the repo root)")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    require(tool, isinstance(config, dict), f"{path} is not a YAML mapping")
    missing = [key for key in required_keys if key not in config]
    require(tool, not missing, f"{path} lacks {', '.join(missing)}")
    return config


def parse_overrides(tool, arguments):
    """`key=value` command-line overrides (OmegaConf style) as a dict."""
    overrides = {}
    for argument in arguments:
        require(tool, "=" in argument, f"override {argument!r} is not key=value")
        key, value = argument.split("=", 1)
        overrides[key] = value
    return overrides


def require_repo_relative(tool, key, value):
    """A directory override must be relative to the repo root, forward slashes, and stay inside it."""
    require(tool, not os.path.isabs(value) and "\\" not in value and ".." not in value.split("/"),
            f"{key}={value!r} must be a repo-relative path with forward slashes")


def write(path, data=b"mock\n"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
