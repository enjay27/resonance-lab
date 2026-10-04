"""The committed notebooks hold no outputs: eval lines are players' chat (CLAUDE.md: no data in git)."""

import ast
import glob
import json
import os

import pytest

from config import BASE_DIR

NOTEBOOKS = sorted(glob.glob(os.path.join(BASE_DIR, "notebooks", "*.ipynb")))
COMMITTED = [path for path in NOTEBOOKS if ".executed." not in os.path.basename(path)]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_the_parameter_test_notebook_exists():
    assert os.path.join(BASE_DIR, "notebooks", "parameter_test.ipynb") in COMMITTED


@pytest.mark.parametrize("path", COMMITTED, ids=os.path.basename)
def test_a_notebook_has_no_outputs_and_no_execution_counts(path):
    for index, cell in enumerate(load(path)["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [], f"cell {index} of {os.path.basename(path)} has saved output: strip it before committing"
        assert cell["execution_count"] is None, f"cell {index} of {os.path.basename(path)} has an execution count"


@pytest.mark.parametrize("path", COMMITTED, ids=os.path.basename)
def test_a_notebook_is_well_formed_and_its_code_parses(path):
    notebook = load(path)
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids)), "cell ids must be unique"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{os.path.basename(path)} cell {index}")


@pytest.mark.parametrize("path", COMMITTED, ids=os.path.basename)
def test_a_notebook_carries_no_machine_specific_metadata(path):
    metadata = load(path)["metadata"]
    assert set(metadata) <= {"kernelspec", "language_info"}
    assert "path" not in json.dumps(metadata).lower().replace("pathlib", "")


def source(path):
    return ["".join(cell["source"]) for cell in load(path)["cells"]]


def test_the_notebook_is_for_jupyter_not_pycharm():
    notebook = os.path.join(BASE_DIR, "notebooks", "parameter_test.ipynb")
    text = "\n".join(source(notebook))
    assert "pycharm" not in text.lower()
    assert "jupyter lab" in text
    with open(os.path.join(BASE_DIR, "requirements-notebook.txt"), encoding="utf-8") as f:
        requirements = [line.split("#")[0].strip().lower() for line in f]
    assert "jupyterlab" in requirements


def test_run_all_does_not_start_the_sweep():
    """The sweep trains, merges and evaluates every parameter set (hours of GPU): it needs a switch that is off."""
    notebook = os.path.join(BASE_DIR, "notebooks", "parameter_test.ipynb")
    (cell,) = [text for text in source(notebook) if "pt.run_sweep(" in text]
    assert "RUN_SWEEP = False" in cell
    call = next(line for line in cell.splitlines() if "pt.run_sweep(" in line)
    assert call.startswith(" ") and "if RUN_SWEEP:" in cell.split(call)[0]
