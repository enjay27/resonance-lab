"""Runs the real llamafactory pipeline (run_pipeline.py) with only the GPU parts faked.

The code is copied into a temp folder, because every path in config.py derives from the folder the code
lives in: all outputs land there and the repo is never touched. What is faked, and nothing else:

  - `llamafactory-cli train` / `export`, `convert_hf_to_gguf.py`, `llama-quantize`: executables here that check
    what the real tool would need and write placeholder files (see contract.py);
  - `torch`, `tqdm`, `transformers`: stub packages (site/) so eval.py's own code runs; the "model" echoes the
    reference translation.

No pytest import here, so the harness also runs as a plain script. Linux / macOS only (the fakes are scripts).
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FIXTURES = os.path.join(REPO, "tests", "fixtures", "mock_pipeline")

RAW_LOG = "data/raw/mock_raw.jsonl"  # RESONANCE_RAW_LOGS, relative to the repo root: Fetch Data then fetches nothing
EVAL_SET = "data/eval/bp-eval-dataset.jsonl"
PROFILE = "hy-mt2-1.8b"


def copy_repo(dst):
    """The code of the repo (root *.py, scripts/, configs/) plus the fixtures as raw log and eval set, in `dst`."""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(REPO):
        if name.endswith(".py"):
            shutil.copy2(os.path.join(REPO, name), dst)
    for folder in ("scripts", "configs"):
        shutil.copytree(os.path.join(REPO, folder), os.path.join(dst, folder), ignore=shutil.ignore_patterns("__pycache__"))
    for source, target in (("raw.jsonl", RAW_LOG), ("eval.jsonl", EVAL_SET)):
        os.makedirs(os.path.dirname(os.path.join(dst, target)), exist_ok=True)
        shutil.copy2(os.path.join(FIXTURES, source), os.path.join(dst, target))
    return dst


def install_fakes(dst):
    """The fake llama.cpp checkout in `dst`; returns the directory with the fake `llamafactory-cli` (for PATH)."""
    llama_cpp = os.path.join(dst, "llama.cpp")
    shutil.copytree(os.path.join(HERE, "llama.cpp"), llama_cpp)
    for path in (os.path.join(llama_cpp, "build", "bin", "llama-quantize"), os.path.join(llama_cpp, "convert_hf_to_gguf.py")):
        if os.path.isfile(path):
            os.chmod(path, 0o755)
    return os.path.join(HERE, "bin")


def mock_env(bin_dir, extra_pythonpath=(), flags=None):
    env = dict(os.environ)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "site"), *extra_pythonpath, *filter(None, [env.get("PYTHONPATH")])])
    env["MOCK_GPU_LIB"] = HERE
    env["RESONANCE_RAW_LOGS"] = RAW_LOG
    env["RESONANCE_LF_PROFILE"] = PROFILE
    env["RESONANCE_MLFLOW"] = "0"  # tracking off: no run touches MLflow
    for name in ("RESONANCE_RECIPE", "MLFLOW_TRACKING_URI"):
        env.pop(name, None)
    env.update(flags or {})
    return env


def run_pipeline(dst, extra_pythonpath=(), flags=None, args=(), mutate=None):
    """Copy the repo into `dst`, install the fakes and run `python run_pipeline.py [args]` there; the CompletedProcess.

    `mutate(dst)`, when given, edits the copy first (a config drifting out of step with the stages, say)."""
    copy_repo(dst)
    if mutate:
        mutate(dst)
    bin_dir = install_fakes(dst)
    return subprocess.run([sys.executable, "run_pipeline.py", *args], cwd=dst, env=mock_env(bin_dir, extra_pythonpath, flags),
                          capture_output=True, text=True, encoding="utf-8", timeout=300)


if __name__ == "__main__":
    import tempfile

    target = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="mock-pipeline-")
    result = run_pipeline(target, extra_pythonpath=sys.argv[2:])
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    print(f"exit {result.returncode}; workdir {target}")
    sys.exit(result.returncode)
