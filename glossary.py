"""The translation glossary as data (configs/glossary/<season>.json) and the version of its human document.

docs/translation-glossary.md is what people read and what translation agents are briefed with; the JSON holds the same rules in a form the
checks can run (`required`, `banned`) and the deterministic corrections `translation_assemble.py` applies (`fixes`). Two things tie them together:
the JSON names the document version it was written for (`doc_version`), and every export records the version its batches were prepared with, so a
glossary edited in between is noticed (`translation_assemble.StaleBatches`). Pure: files and regular expressions only.
"""

import hashlib
import json
import re
from typing import NamedTuple, Optional, Tuple

VERSION_RX = re.compile(r"^\*\*Version:\*\* (\d+\.\d+\.\d+)", re.M)
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class GlossaryError(Exception):
    """The glossary (or its document) cannot be used; the message names the file or the rule."""


class Required(NamedTuple):
    ja: "re.Pattern"
    unless: Optional["re.Pattern"]
    ko: str


class Banned(NamedTuple):
    ko: str
    when: Optional["re.Pattern"]
    why: str


class Fix(NamedTuple):
    when: "re.Pattern"
    unless: Optional["re.Pattern"]
    old: str
    new: str
    why: str


class Glossary(NamedTuple):
    season: str
    doc_version: str
    required: Tuple[Required, ...]
    banned: Tuple[Banned, ...]
    fixes: Tuple[Fix, ...]


def _pattern(text, where):
    try:
        return re.compile(text)
    except (re.error, TypeError) as error:
        raise GlossaryError(f"{where}: {text!r} is not a regular expression ({error})") from None


def _optional(entry, key, where):
    return _pattern(entry[key], where) if entry.get(key) else None


def _text(entry, key, where):
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise GlossaryError(f"{where}: needs a non-empty \"{key}\"")
    return value


def load_glossary(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise GlossaryError(f"{path}: glossary not found") from None
    except (OSError, ValueError) as error:
        raise GlossaryError(f"{path}: not a readable JSON file ({error})") from None
    if not isinstance(data, dict):
        raise GlossaryError(f"{path}: needs a JSON object")
    season = _text(data, "season", path)
    doc_version = data.get("doc_version")
    if not isinstance(doc_version, str) or not SEMVER.match(doc_version):
        raise GlossaryError(f"{path}: doc_version must look like 1.0.0")
    required, banned, fixes = [], [], []
    for n, entry in enumerate(data.get("required", [])):
        where = f"{path}: required[{n}]"
        required.append(Required(_pattern(_text(entry, "ja", where), where), _optional(entry, "unless", where), _text(entry, "ko", where)))
    for n, entry in enumerate(data.get("banned", [])):
        where = f"{path}: banned[{n}]"
        banned.append(Banned(_text(entry, "ko", where), _optional(entry, "when", where), entry.get("why", "")))
    for n, entry in enumerate(data.get("fixes", [])):
        where = f"{path}: fixes[{n}]"
        old, new = _text(entry, "old", where), _text(entry, "new", where)
        if old == new:
            raise GlossaryError(f"{where}: old and new are the same, the fix changes nothing")
        fixes.append(Fix(_pattern(_text(entry, "when", where), where), _optional(entry, "unless", where), old, new, entry.get("why", "")))
    return Glossary(season, doc_version, tuple(required), tuple(banned), tuple(fixes))


def doc_version(path):
    """The `**Version:** X.Y.Z` of a guide's header."""
    try:
        with open(path, encoding="utf-8") as f:
            match = VERSION_RX.search(f.read())
    except OSError as error:
        raise GlossaryError(f"{path}: cannot be read ({error})") from None
    if not match:
        raise GlossaryError(f"{path}: no \"**Version:** X.Y.Z\" in the header")
    return match.group(1)


def compare_versions(a, b):
    """-1, 0 or 1 as numbers (1.10.0 is newer than 1.9.0)."""
    left, right = [tuple(int(part) for part in v.split(".")) for v in (a, b)]
    return (left > right) - (left < right)


def file_sha1(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()
