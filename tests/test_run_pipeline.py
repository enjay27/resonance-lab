import sys

import pytest

import run_pipeline


def script(tmp_path, body):
    path = tmp_path / "stage.py"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_check_system_copes_without_torch(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` raises ImportError

    run_pipeline.check_system()

    assert "torch not installed" in capsys.readouterr().out


def test_run_step_passes_the_stage_arguments_to_the_script(tmp_path):
    out = tmp_path / "args.txt"
    path = script(tmp_path, f"import sys; open({str(out)!r}, 'w').write(' '.join(sys.argv[1:]))")

    assert run_pipeline.run_step("Stage", path, ("--format", "pair")) is True
    assert out.read_text() == "--format pair"


def test_run_step_reports_a_failing_script(tmp_path):
    assert run_pipeline.run_step("Stage", script(tmp_path, "raise SystemExit(1)")) is False


def test_run_step_reports_a_missing_script(tmp_path):
    assert run_pipeline.run_step("Stage", str(tmp_path / "nope.py")) is False


def test_main_halts_at_the_first_failing_stage(tmp_path, monkeypatch):
    ran = []
    stages = [
        run_pipeline.pipelines.Stage("one", "1.py"),
        run_pipeline.pipelines.Stage("two", "2.py"),
        run_pipeline.pipelines.Stage("three", "3.py"),
    ]
    monkeypatch.setattr(run_pipeline.pipelines, "stages", lambda name: stages)
    monkeypatch.setattr(run_pipeline, "check_system", lambda: None)
    monkeypatch.setattr(run_pipeline, "run_step", lambda name, path, args=(), env=None: ran.append(name) or name != "two")
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"])

    with pytest.raises(SystemExit) as exc:
        run_pipeline.main()

    assert exc.value.code == 1
    assert ran == ["one", "two"]


def test_run_step_passes_the_environment_to_the_script(tmp_path):
    out = tmp_path / "env.txt"
    path = script(tmp_path, f"import os; open({str(out)!r}, 'w').write(os.environ['RESONANCE_LF_PROFILE'])")

    assert run_pipeline.run_step("Stage", path, env={"RESONANCE_LF_PROFILE": "hy-mt2-1.8b"}) is True
    assert out.read_text() == "hy-mt2-1.8b"


def test_stage_env_puts_the_model_parameter_over_the_environment():
    base = {"RESONANCE_LF_PROFILE": "hy-mt2-7b", "PATH": "x"}

    assert run_pipeline.stage_env("hy-mt2-1.8b", base) == {"RESONANCE_LF_PROFILE": "hy-mt2-1.8b", "PATH": "x"}
    assert run_pipeline.stage_env(None, base) == base  # no parameter: the environment stands


def test_main_runs_every_stage_with_the_model_parameter_in_the_environment(monkeypatch):
    envs = []
    monkeypatch.setattr(run_pipeline.pipelines, "stages", lambda name: [run_pipeline.pipelines.Stage("one", "1.py")])
    monkeypatch.setattr(run_pipeline, "check_system", lambda: None)
    monkeypatch.setattr(run_pipeline, "run_step", lambda name, path, args=(), env=None: envs.append(env) or True)
    monkeypatch.setenv("RESONANCE_LF_PROFILE", "hy-mt2-7b")
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--model", "hy-mt2-1.8b"])

    run_pipeline.main()

    assert envs[0]["RESONANCE_LF_PROFILE"] == "hy-mt2-1.8b"
