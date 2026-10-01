import json
import os
import sys

import pytest
import yaml

import config
import lf_tools


@pytest.fixture
def profile():
    return lf_tools.load_profile(config.LF_PROFILE)


def read_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- profile: one folder = one model, train.yaml + merge.yaml ---------------------------------


def test_default_profile_files_exist(profile):
    assert os.path.isfile(profile.train_yaml)
    assert os.path.isfile(profile.merge_yaml)


def test_merge_reads_the_adapter_train_writes(profile):
    train, merge = read_yaml(profile.train_yaml), read_yaml(profile.merge_yaml)
    assert train["output_dir"] == merge["adapter_name_or_path"]
    assert profile.adapter_dir == os.path.join(config.BASE_DIR, train["output_dir"])


def test_train_and_merge_use_the_same_base_model_and_template(profile):
    train, merge = read_yaml(profile.train_yaml), read_yaml(profile.merge_yaml)
    assert train["model_name_or_path"] == merge["model_name_or_path"] == profile.base_model
    assert train["template"] == merge["template"] == profile.template


def test_train_reads_the_dataset_update_dataset_info_writes(profile):
    assert read_yaml(profile.train_yaml)["dataset"] == config.LF_DATASET_NAME == profile.dataset


def test_merged_model_dir_comes_from_the_merge_yaml(profile):
    assert profile.merged_dir == os.path.join(config.BASE_DIR, read_yaml(profile.merge_yaml)["export_dir"])


def test_unknown_profile_lists_the_available_ones():
    with pytest.raises(ValueError, match="translategemma-4b"):
        lf_tools.load_profile("nope")


def test_available_profiles_are_the_folders_with_both_yaml_files():
    assert "translategemma-4b" in lf_tools.available_profiles()


# --- dataset_info.json -------------------------------------------------------------------------


def test_dataset_info_maps_the_pair_columns():
    info = lf_tools.dataset_info("processed/x.jsonl")

    assert info == {
        config.LF_DATASET_NAME: {
            "file_name": "processed/x.jsonl",
            "columns": {"prompt": "original", "response": "translated"},
        }
    }


def test_dataset_info_file_is_what_preprocess_writes():
    # file_name is relative to the dataset dir, and must land on preprocess's output file.
    info = lf_tools.dataset_info_for_processed_logs()
    file_name = info[config.LF_DATASET_NAME]["file_name"]
    assert os.path.normpath(os.path.join(config.LF_DATASET_DIR, file_name)) == os.path.normpath(config.PROCESSED_LOGS)
    assert "\\" not in file_name  # always forward slashes in the json, whatever the OS


# --- commands: argument lists, never a shell string --------------------------------------------


def test_train_and_merge_commands():
    assert lf_tools.train_command("t.yaml") == ["llamafactory-cli", "train", "t.yaml"]
    assert lf_tools.merge_command("m.yaml") == ["llamafactory-cli", "export", "m.yaml"]


def test_convert_command_uses_this_python_and_the_llama_cpp_script():
    cmd = lf_tools.convert_command("/l", "/merged", "/out.gguf")
    assert cmd == [sys.executable, os.path.join("/l", "convert_hf_to_gguf.py"), "/merged", "--outfile", "/out.gguf", "--outtype", "f16"]


def test_quantize_command():
    assert lf_tools.quantize_command("/bin/q", "/a.gguf", "/b.gguf") == ["/bin/q", "/a.gguf", "/b.gguf", "Q4_K_M"]


@pytest.mark.parametrize(
    "relative",
    [
        os.path.join("build", "bin", "Release", "llama-quantize.exe"),  # Windows, MSVC
        os.path.join("build", "bin", "llama-quantize"),  # Linux / macOS
        os.path.join("build", "bin", "llama-quantize.exe"),  # Windows, Ninja / MinGW
    ],
)
def test_quantize_binary_finds_each_build_layout(tmp_path, relative):
    binary = tmp_path / relative
    binary.parent.mkdir(parents=True)
    binary.write_text("")

    assert lf_tools.quantize_binary(str(tmp_path)) == str(binary)


def test_quantize_binary_missing_names_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="llama-quantize"):
        lf_tools.quantize_binary(str(tmp_path))


def test_gguf_names_follow_the_profile(tmp_path):
    f16, q4 = lf_tools.gguf_paths("translategemma-4b", str(tmp_path))
    assert f16 == str(tmp_path / "bp-translategemma-4b-f16.gguf")
    assert q4 == str(tmp_path / "bp-translategemma-4b-q4_k_m.gguf")


# --- training log ------------------------------------------------------------------------------


def test_tail_returns_the_last_lines(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("\n".join(f"line {i}" for i in range(100)), encoding="utf-8")

    assert lf_tools.tail(str(log), 3) == ["line 97", "line 98", "line 99"]


def test_tail_of_a_missing_log_is_empty(tmp_path):
    assert lf_tools.tail(str(tmp_path / "nope.log"), 3) == []


def test_dataset_info_serialises_to_json():
    json.dumps(lf_tools.dataset_info("processed/x.jsonl"))


# --- run ---------------------------------------------------------------------------------------


def test_run_returns_when_the_command_succeeds(capsys):
    lf_tools.run([sys.executable, "-c", "pass"], "a good step")
    assert "a good step" in capsys.readouterr().out


def test_run_exits_one_when_the_command_fails():
    with pytest.raises(SystemExit) as exc:
        lf_tools.run([sys.executable, "-c", "raise SystemExit(3)"], "a bad step")
    assert exc.value.code == 1


def test_run_exits_one_when_the_program_is_missing(capsys):
    with pytest.raises(SystemExit) as exc:
        lf_tools.run(["definitely-not-a-program-xyz"], "a missing program")
    assert exc.value.code == 1


def test_run_starts_in_the_repo_root(tmp_path):
    out = tmp_path / "cwd.txt"
    lf_tools.run([sys.executable, "-c", f"import os; open({str(out)!r}, 'w').write(os.getcwd())"], "cwd")
    assert out.read_text() == config.BASE_DIR


# --- run_logged: the training run, output captured to a log -------------------------------------


def test_run_logged_writes_the_output_to_the_log(tmp_path):
    log = tmp_path / "logs" / "train.log"

    lf_tools.run_logged([sys.executable, "-c", "print('hello'); import sys; print('oops', file=sys.stderr)"], str(log), "train")

    text = log.read_text(encoding="utf-8")
    assert "hello" in text and "oops" in text


def test_run_logged_failure_shows_the_log_tail_and_exits_one(tmp_path, capsys):
    log = tmp_path / "train.log"
    code = "\n".join(f"print('step {i}')" for i in range(60)) + "\nraise SystemExit(2)"

    with pytest.raises(SystemExit) as exc:
        lf_tools.run_logged([sys.executable, "-c", code], str(log), "train", tail_lines=5)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "step 59" in out and "step 54" not in out  # only the last 5 lines


def test_run_logged_missing_program_exits_one_without_a_traceback(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        lf_tools.run_logged(["definitely-not-a-program-xyz"], str(tmp_path / "t.log"), "train")

    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().out.lower()


# --- eval prompts --------------------------------------------------------------------------------


def test_training_prompt_is_the_raw_line_in_the_models_turn_format():
    # What the model saw in training (template gemma3, no system prompt); the tokenizer adds <bos>.
    text = lf_tools.training_prompt("gemma3", "遺跡1Fから　29k↑　＠T1")

    assert text == "<start_of_turn>user\n遺跡1Fから　29k↑　＠T1<end_of_turn>\n<start_of_turn>model\n"
    assert "<bos>" not in text


def test_training_prompt_of_an_unknown_template_lists_the_known_ones():
    with pytest.raises(ValueError, match="gemma3"):
        lf_tools.training_prompt("nope", "x")


def test_the_default_profiles_template_has_a_training_prompt(profile):
    assert lf_tools.training_prompt(profile.template, "x")


def test_translategemma_messages_carry_the_language_codes():
    assert lf_tools.translategemma_messages("こんにちは") == [
        {"role": "user", "content": [{"type": "text", "source_lang_code": "ja", "target_lang_code": "ko", "text": "こんにちは"}]}
    ]
