import json
import os
import sys

import pytest
import yaml

import config
import lf_tools
import runs


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


@pytest.mark.parametrize("name", lf_tools.available_profiles())
def test_no_profile_lets_the_trainer_report_to_a_tracker_on_its_own(name):
    """transformers' default is report_to=all: with mlflow-skinny installed (for tracker.py) its MLflow callback starts a
    run against a local sqlite store the skinny client cannot open, and the training dies. Tracking is ours (tracker.py)."""
    assert read_yaml(lf_tools.load_profile(name).train_yaml)["report_to"] == "none"


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


# --- Hy-MT2 profiles (Tencent; templates registered upstream in LLaMA-Factory) -------------------

HY_PROFILES = ["hy-mt2-1.8b", "hy-mt2-7b"]


@pytest.mark.parametrize("name", HY_PROFILES)
def test_hy_profiles_are_complete_and_agree(name):
    assert name in lf_tools.available_profiles()
    p = lf_tools.load_profile(name)
    train, merge = read_yaml(p.train_yaml), read_yaml(p.merge_yaml)
    assert train["model_name_or_path"] == merge["model_name_or_path"] == p.base_model
    assert train["template"] == merge["template"] == p.template
    assert p.dataset == config.LF_DATASET_NAME
    assert merge["adapter_name_or_path"] == train["output_dir"]
    assert p.template in lf_tools.TRAINING_PROMPTS


def test_hy_profiles_use_their_own_template_and_output_dirs():
    small, big = (lf_tools.load_profile(n) for n in HY_PROFILES)
    assert (small.template, big.template) == ("hy_dense_1_8b", "hy_dense_7b")
    assert small.base_model == "tencent/Hy-MT2-1.8B" and big.base_model == "tencent/Hy-MT2-7B"
    assert len({small.adapter_dir, big.adapter_dir, small.merged_dir, big.merged_dir}) == 4


def test_hy_training_prompts_match_the_upstream_templates():
    # LLaMA-Factory's hy_dense_* templates, no system prompt; the BOS is added by the template prefix.
    assert lf_tools.training_prompt("hy_dense_1_8b", "こんにちは") == "<｜hy_User｜>こんにちは<｜hy_Assistant｜>"
    assert lf_tools.training_prompt("hy_dense_7b", "こんにちは") == "こんにちは<|extra_0|>"


def test_chat_messages_of_hy_use_the_official_english_translate_prompt():
    for template in ("hy_dense_1_8b", "hy_dense_7b"):
        [msg] = lf_tools.chat_messages(template, "こんにちは")
        assert msg["role"] == "user"
        assert msg["content"] == (
            "Translate the following text into Korean. Note that you should only output the translated "
            "result without any additional explanation:\n\nこんにちは"
        )


def test_chat_messages_of_gemma3_are_translategemmas():
    assert lf_tools.chat_messages("gemma3", "x") == lf_tools.translategemma_messages("x")


def test_chat_messages_of_an_unknown_template_lists_the_known_ones():
    with pytest.raises(ValueError, match="hy_dense_7b"):
        lf_tools.chat_messages("nope", "x")


# --- BOS in the training-format eval prompt ------------------------------------------------------


def test_with_bos_prepends_it_when_the_tokenizer_did_not():
    # Hy-MT2's tokenizer does not add BOS by itself (checked locally); training's template prefix did.
    assert lf_tools.with_bos([5, 6], bos_id=1) == [1, 5, 6]


def test_with_bos_leaves_ids_that_already_start_with_it():
    assert lf_tools.with_bos([1, 5, 6], bos_id=1) == [1, 5, 6]  # gemma3: the tokenizer adds <bos>


def test_with_bos_is_a_noop_without_a_bos_token():
    assert lf_tools.with_bos([5, 6], bos_id=None) == [5, 6]


# --- generate() inputs ---------------------------------------------------------------------------


def test_generate_inputs_keep_only_what_generate_accepts():
    # Hy-MT2's tokenizer also returns token_type_ids; model.generate rejects it (transformers 4.57).
    encoding = {"input_ids": [[1, 2]], "attention_mask": [[1, 1]], "token_type_ids": [[0, 0]]}

    assert lf_tools.generate_inputs(encoding) == {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}


def test_generate_inputs_work_without_an_attention_mask():
    assert lf_tools.generate_inputs({"input_ids": [[1]]}) == {"input_ids": [[1]]}


# --- --model: parameter > environment > default ----------------------------------------------------


def test_model_name_prefers_the_parameter_over_the_environment():
    env = {"RESONANCE_LF_PROFILE": "hy-mt2-7b"}

    assert lf_tools.model_name("hy-mt2-1.8b", env) == "hy-mt2-1.8b"


def test_model_name_falls_back_to_the_environment_then_the_default():
    assert lf_tools.model_name(None, {"RESONANCE_LF_PROFILE": "hy-mt2-7b"}) == "hy-mt2-7b"
    assert lf_tools.model_name(None, {}) == config.LF_PROFILE_DEFAULT
    assert lf_tools.model_name(None, {"RESONANCE_LF_PROFILE": ""}) == config.LF_PROFILE_DEFAULT  # an empty variable is unset


def test_profile_from_args_reads_the_model_parameter(monkeypatch):
    monkeypatch.setenv("RESONANCE_LF_PROFILE", "hy-mt2-7b")

    profile, rest = lf_tools.profile_from_args(["--model", "hy-mt2-1.8b", "--prompt", "x"], "demo")

    assert profile.name == "hy-mt2-1.8b"
    assert rest == ["--prompt", "x"]  # the caller's own arguments are left for its parser


def test_profile_from_args_uses_the_environment_without_the_parameter(monkeypatch):
    monkeypatch.setenv("RESONANCE_LF_PROFILE", "hy-mt2-7b")

    profile, rest = lf_tools.profile_from_args([], "demo")

    assert profile.name == "hy-mt2-7b" and rest == []


def test_an_unknown_model_lists_the_profiles():
    with pytest.raises(ValueError, match="hy-mt2-1.8b"):
        lf_tools.profile_from_args(["--model", "nope"], "demo")


def test_merge_stage_uses_the_model_parameter_not_the_environment(monkeypatch, capsys):
    import merge

    monkeypatch.setenv("RESONANCE_LF_PROFILE", "translategemma-4b")
    with pytest.raises(SystemExit):
        merge.merge(["--model", "hy-mt2-1.8b"])  # no adapter in the test tree: the error names the one asked for

    assert "hy-mt2-1.8b_lora" in capsys.readouterr().out


def test_train_stage_trains_the_model_parameters_yaml(monkeypatch, tmp_path):
    import importlib.util

    # scripts/unsloth/train.py has the same module name: load the llamafactory one by path.
    spec = importlib.util.spec_from_file_location(
        "lf_train", os.path.join(config.BASE_DIR, "scripts", "llamafactory", "train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    seen = []
    monkeypatch.setattr(train, "run_logged", lambda cmd, log, what: seen.append(cmd))
    monkeypatch.setattr(train, "check_training_data", lambda profile: None)  # the data check has its own tests
    monkeypatch.setattr(train, "start_run", lambda base, name: runs.start_run(str(tmp_path), name))

    train.train(["--model", "hy-mt2-7b"])

    assert seen[0][2].replace("\\", "/").endswith("configs/llamafactory/hy-mt2-7b/train.yaml")


def test_no_llamafactory_script_reads_the_profile_from_config_directly():
    # Every script picks its model through --model (parameter > env > default), never config.LF_PROFILE.
    folder = os.path.join(config.BASE_DIR, "scripts", "llamafactory")
    for name in os.listdir(folder):
        if name.endswith(".py") and name != "lf_tools.py":
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                assert "LF_PROFILE" not in f.read(), name


# --- "-fast" profiles: the same model and recipe, only the speed settings differ --------------------

FAST_PAIRS = [("translategemma-4b", "translategemma-4b-fast"), ("hy-mt2-1.8b", "hy-mt2-1.8b-fast"), ("hy-mt2-7b", "hy-mt2-7b-fast")]
# What a fast profile may change: where it writes, and throughput (batch shape, packing, kernels, attention).
FAST_MAY_DIFFER = {
    "output_dir", "per_device_train_batch_size", "gradient_accumulation_steps", "gradient_checkpointing",
    "packing", "neat_packing", "enable_liger_kernel", "flash_attn", "eval_steps", "save_steps",
}


@pytest.mark.parametrize("base,fast", FAST_PAIRS)
def test_fast_profile_changes_only_speed_settings(base, fast):
    b, f = (read_yaml(lf_tools.load_profile(n).train_yaml) for n in (base, fast))

    changed = {k for k in b.keys() | f.keys() if b.get(k) != f.get(k)}

    assert changed <= FAST_MAY_DIFFER, changed - FAST_MAY_DIFFER
    assert f["packing"] is True and f["enable_liger_kernel"] is True
    for key in ("model_name_or_path", "template", "dataset", "cutoff_len", "learning_rate", "num_train_epochs", "lora_rank"):
        assert f[key] == b[key]


@pytest.mark.parametrize("base,fast", FAST_PAIRS)
def test_fast_profile_keeps_the_effective_batch_and_has_its_own_dirs(base, fast):
    pb, pf = lf_tools.load_profile(base), lf_tools.load_profile(fast)
    b, f = read_yaml(pb.train_yaml), read_yaml(pf.train_yaml)

    eff = lambda y: y["per_device_train_batch_size"] * y["gradient_accumulation_steps"]  # noqa: E731
    assert eff(f) >= eff(b)
    assert pf.base_model == pb.base_model and pf.template == pb.template
    assert pf.adapter_dir != pb.adapter_dir and pf.merged_dir != pb.merged_dir
    assert read_yaml(pf.merge_yaml)["adapter_name_or_path"] == f["output_dir"]


@pytest.mark.parametrize("base,fast", FAST_PAIRS)
def test_fast_profile_evaluates_often_enough_to_have_a_validation_curve(base, fast):
    """A fast run is ~90 steps (packed, effective batch 32, 3 epochs). With eval_steps 100 it never evaluated: no
    eval-loss curve in MLflow and no best checkpoint for load_best_model_at_end. Evaluate and save every few steps."""
    f = read_yaml(lf_tools.load_profile(fast).train_yaml)

    assert f["eval_strategy"] == f["save_strategy"] == "steps" and f["load_best_model_at_end"] is True
    assert f["eval_steps"] <= 20
    assert f["save_steps"] % f["eval_steps"] == 0  # the trainer refuses a best-model save that is not on an eval step


@pytest.mark.parametrize("base,fast", FAST_PAIRS)
def test_the_base_profile_keeps_its_own_eval_cadence(base, fast):
    b = read_yaml(lf_tools.load_profile(base).train_yaml)

    assert b["eval_steps"] == b["save_steps"] == 100  # ~1600 steps: eval every 100 (the fast profiles' scaling is their own)


# --- a variant of a profile: the same recipe with one thing changed ---------------------------------

# (base, variant, what the train.yaml may differ in besides the output folder)
VARIANTS = [("translategemma-4b", "translategemma-4b-lr1e-4", {"learning_rate"})]


@pytest.mark.parametrize("base,variant,changes", VARIANTS)
def test_a_variant_profile_changes_only_what_it_is_for(base, variant, changes):
    b, v = (lf_tools.load_profile(n) for n in (base, variant))
    bt, vt = read_yaml(b.train_yaml), read_yaml(v.train_yaml)

    assert {k for k in bt.keys() | vt.keys() if bt.get(k) != vt.get(k)} == changes | {"output_dir"}
    assert v.base_model == b.base_model and v.template == b.template and v.dataset == b.dataset
    bm, vm = read_yaml(b.merge_yaml), read_yaml(v.merge_yaml)
    assert {k for k in bm.keys() | vm.keys() if bm.get(k) != vm.get(k)} == {"adapter_name_or_path", "export_dir"}
    assert vm["adapter_name_or_path"] == vt["output_dir"]
    assert v.adapter_dir != b.adapter_dir and v.merged_dir != b.merged_dir  # a run of one never overwrites the other


def test_the_lr1e4_variant_has_the_learning_rate_it_is_named_for():
    assert read_yaml(lf_tools.load_profile("translategemma-4b-lr1e-4").train_yaml)["learning_rate"] == 1.0e-4


@pytest.mark.parametrize("name", lf_tools.available_profiles())
def test_every_profiles_model_folders_are_gitignored(name):
    """The *.safetensors are ignored, but a merged model's folder also holds tokenizer.json (tens of MB) and configs:
    the whole folder of every profile (merged model, adapters) must be ignored, never one `git add -A` from a commit."""
    import subprocess

    p = lf_tools.load_profile(name)
    for folder in (p.adapter_dir, p.merged_dir):
        path = os.path.join(os.path.relpath(folder, config.BASE_DIR), "tokenizer.json")
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=config.BASE_DIR).returncode == 0, path


# --- reading training pairs for inspection ---------------------------------------------------------


def test_first_pairs_reads_original_and_translated_in_order(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"original": "a", "translated": "가"}\n\n{"original": "b", "translated": "나"}\n{"original": "c", "translated": "다"}\n',
        encoding="utf-8",
    )

    assert lf_tools.first_pairs(str(path), 2) == [("a", "가"), ("b", "나")]


def test_first_pairs_says_when_the_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess"):
        lf_tools.first_pairs(str(tmp_path / "nope.jsonl"), 1)


def test_first_pairs_rejects_rows_without_the_pair_columns(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"instruction": "x", "input": "a", "output": "b"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="--format pair"):
        lf_tools.first_pairs(str(path), 1)


# --- the full training example per template (verified against inspect_pair.py output, 2026-10-01) ---
# Made-up line pair; the strings are what LLaMA-Factory builds: BOS + prompt masked, response trained.

ORIGINAL, TRANSLATED = "杖@2募集", "법사@2 모집"


def test_training_example_of_gemma3_ends_the_turn_in_the_response():
    masked, trained = lf_tools.training_example("gemma3", ORIGINAL, TRANSLATED)

    assert masked == "<bos><start_of_turn>user\n杖@2募集<end_of_turn>\n<start_of_turn>model\n"
    assert trained == "법사@2 모집<end_of_turn>\n"


def test_training_example_of_hy_1_8b_trains_the_assistant_token_and_the_eos():
    # The 1.8B template puts <｜hy_Assistant｜> in the response; LLaMA-Factory appends the eos (efficient_eos).
    masked, trained = lf_tools.training_example("hy_dense_1_8b", ORIGINAL, TRANSLATED)

    assert masked == "<｜hy_begin▁of▁sentence｜><｜hy_User｜>杖@2募集"
    assert trained == "<｜hy_Assistant｜>법사@2 모집<｜hy_place▁holder▁no▁2｜>"


def test_training_example_of_hy_7b_has_the_user_marker_in_the_prompt():
    masked, trained = lf_tools.training_example("hy_dense_7b", ORIGINAL, TRANSLATED)

    assert masked == "<|startoftext|>杖@2募集<|extra_0|>"
    assert trained == "법사@2 모집<|eos|>"


def test_training_example_is_the_inference_prompt_plus_the_answer():
    # What the model is asked at inference (training_prompt) is the training text up to the answer.
    for template in ("gemma3", "hy_dense_1_8b", "hy_dense_7b"):
        masked, trained = lf_tools.training_example(template, ORIGINAL, TRANSLATED)
        bos = lf_tools.TRAINING_BOS[template]

        assert (masked + trained).startswith(bos + lf_tools.training_prompt(template, ORIGINAL))


def test_training_example_of_an_unknown_template_lists_the_known_ones():
    with pytest.raises(ValueError, match="hy_dense_7b"):
        lf_tools.training_example("nope", "a", "b")


# --- the user text of a training example follows the template's prompt style -------------------------


def test_training_user_text_is_the_instruction_and_the_line_of_the_templates_style():
    import prompts

    for template in ("gemma3", "hy_dense_1_8b", "hy_dense_7b"):
        assert lf_tools.training_user_text(template, "杖@2募集") == prompts.build_prompt(
            prompts.style_for_template(template), "ja-ko", "杖@2募集"
        )


def test_hy_chat_messages_use_the_shared_prompt_module():
    import prompts

    [msg] = lf_tools.chat_messages("hy_dense_7b", "杖@2募集")

    assert msg["content"] == prompts.build_prompt("hy", "ja-ko", "杖@2募集")


# --- the training file must have been made for the profile's prompt style -----------------------------


def _made_for(tmp_path, style, **kwargs):
    import manifest

    data = tmp_path / "lora_train_data.jsonl"
    data.write_text('{"original": "a", "translated": "b"}\n', encoding="utf-8")
    manifest.val_path(str(data))
    with open(manifest.val_path(str(data)), "w", encoding="utf-8") as f:
        f.write('{"original": "v", "translated": "w"}\n')
    manifest.write_manifest(str(data), fmt="pair", style=style, reverse=False, raw_path=str(data), counts={}, eval_set=None,
                            eval_lines=0, val_fraction=0.05)
    return str(data)


def test_check_training_data_accepts_a_file_made_for_the_profiles_template(tmp_path):
    data = _made_for(tmp_path, "hy")

    meta = lf_tools.check_training_data(lf_tools.load_profile("hy-mt2-1.8b"), data)

    assert meta["style"] == "hy"


def test_check_training_data_refuses_another_models_prompt_style(tmp_path):
    from manifest import ManifestError

    data = _made_for(tmp_path, "translategemma")

    with pytest.raises(ManifestError, match="hy-mt2-1.8b"):
        lf_tools.check_training_data(lf_tools.load_profile("hy-mt2-1.8b"), data)


def test_check_training_data_without_a_manifest_names_the_profile_and_the_fix(tmp_path):
    from manifest import ManifestError

    with pytest.raises(ManifestError, match="--model translategemma-4b"):
        lf_tools.check_training_data(lf_tools.load_profile("translategemma-4b"), str(tmp_path / "missing.jsonl"))


def test_the_train_stage_stops_when_the_data_does_not_match(monkeypatch, capsys):
    import importlib.util

    from manifest import ManifestError

    spec = importlib.util.spec_from_file_location(
        "lf_train2", os.path.join(config.BASE_DIR, "scripts", "llamafactory", "train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    ran = []
    monkeypatch.setattr(train, "run_logged", lambda *a: ran.append(a))

    def refuse(profile):
        raise ManifestError("wrong style")

    monkeypatch.setattr(train, "check_training_data", refuse)

    with pytest.raises(SystemExit) as exc:
        train.train(["--model", "hy-mt2-7b"])

    assert exc.value.code == 1 and not ran
    assert "wrong style" in capsys.readouterr().out


def test_update_dataset_info_stops_when_the_data_does_not_match(monkeypatch, capsys):
    import update_dataset_info
    from manifest import ManifestError

    def refuse(profile):
        raise ManifestError("wrong style")

    monkeypatch.setattr(update_dataset_info, "check_training_data", refuse)
    monkeypatch.setattr(update_dataset_info, "LF_DATASET_INFO_PATH", "/nonexistent/dataset_info.json")

    with pytest.raises(SystemExit) as exc:
        update_dataset_info.main(["--model", "hy-mt2-1.8b"])

    assert exc.value.code == 1
    assert "wrong style" in capsys.readouterr().out


# --- LLaMA-Factory reads the validation rows from their own dataset -----------------------------------------


def test_dataset_info_has_a_training_and_a_validation_entry():
    info = lf_tools.dataset_info_for_processed_logs()

    assert set(info) == {config.LF_DATASET_NAME, config.LF_VAL_DATASET_NAME}
    assert info[config.LF_DATASET_NAME]["file_name"].endswith("lora_train_data.jsonl")
    assert info[config.LF_VAL_DATASET_NAME]["file_name"].endswith("lora_train_data.val.jsonl")
    assert info[config.LF_VAL_DATASET_NAME]["columns"] == info[config.LF_DATASET_NAME]["columns"]


@pytest.mark.parametrize("name", lf_tools.available_profiles())
def test_every_profile_validates_on_the_validation_dataset_not_a_random_row_split(name):
    train = read_yaml(lf_tools.load_profile(name).train_yaml)

    assert train["eval_dataset"] == config.LF_VAL_DATASET_NAME
    assert "val_size" not in train  # LLaMA-Factory refuses both together (hparams/data_args.py)


# --- every training is its own run: its own adapter dir, log and status ------------------------------------


def test_train_command_overrides_the_output_dir_with_a_repo_relative_path():
    cmd = lf_tools.train_command("configs/x/train.yaml", os.path.join(config.BASE_DIR, "outputs", "hy_lora", "20261001-163005"))

    assert cmd == ["llamafactory-cli", "train", "configs/x/train.yaml", "output_dir=outputs/hy_lora/20261001-163005"]


def test_merge_command_overrides_the_adapter_with_a_repo_relative_path():
    cmd = lf_tools.merge_command("configs/x/merge.yaml", os.path.join(config.BASE_DIR, "outputs", "hy_lora", "r1"))

    assert cmd == ["llamafactory-cli", "export", "configs/x/merge.yaml", "adapter_name_or_path=outputs/hy_lora/r1"]


def test_the_commands_without_a_directory_are_the_plain_yaml_ones():
    assert lf_tools.train_command("t.yaml") == ["llamafactory-cli", "train", "t.yaml"]
    assert lf_tools.merge_command("m.yaml") == ["llamafactory-cli", "export", "m.yaml"]


def _load_stage(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"lf_{name}_x", os.path.join(config.BASE_DIR, "scripts", "llamafactory", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_stage_runs_in_its_own_dir_with_its_own_log_and_records_success(monkeypatch, tmp_path):
    train = _load_stage("train")
    monkeypatch.setattr(train, "check_training_data", lambda profile: None)
    monkeypatch.setattr(train, "start_run", lambda base, name: runs.start_run(str(tmp_path), name))
    seen = []
    monkeypatch.setattr(train, "run_logged", lambda cmd, log, what: seen.append((cmd, log)))

    train.train(["--model", "hy-mt2-1.8b"])

    cmd, log = seen[0]
    run_dir = runs.latest_run(str(tmp_path))
    assert cmd[-1] == "output_dir=" + os.path.relpath(run_dir, config.BASE_DIR).replace(os.sep, "/")
    assert log == os.path.join(run_dir, runs.TRAIN_LOG_NAME)
    assert runs.read_run(run_dir)["status"] == "complete"


def test_train_stage_records_a_failed_run_and_still_stops_the_pipeline(monkeypatch, tmp_path):
    train = _load_stage("train")
    monkeypatch.setattr(train, "check_training_data", lambda profile: None)
    monkeypatch.setattr(train, "start_run", lambda base, name: runs.start_run(str(tmp_path), name))

    def fail(cmd, log, what):
        raise SystemExit(1)

    monkeypatch.setattr(train, "run_logged", fail)

    with pytest.raises(SystemExit) as exc:
        train.train(["--model", "hy-mt2-1.8b"])

    assert exc.value.code == 1
    assert runs.read_run(runs.latest_run(str(tmp_path)))["status"] == "failed"


def _use_dirs(monkeypatch, merge, profile, adapter_base, merged_dir):
    """The merge stage on a temp tree: the profile's adapter base dir and merged dir are replaced."""
    real = merge.profile_from_args

    def moved(argv, description):
        found, rest = real(argv, description)  # the real parsing: --model is consumed, --run stays in `rest`
        return found._replace(adapter_dir=adapter_base, merged_dir=merged_dir), rest

    monkeypatch.setattr(merge, "profile_from_args", moved)


def test_merge_stage_merges_the_latest_complete_run_and_notes_where_it_came_from(monkeypatch, tmp_path):
    merge = _load_stage("merge")
    profile = lf_tools.load_profile("hy-mt2-1.8b")
    run = runs.start_run(str(tmp_path / "lora"), profile.name)
    runs.finish_run(run.dir, "complete")
    merged = tmp_path / "merged"
    merged.mkdir()
    _use_dirs(monkeypatch, merge, profile, str(tmp_path / "lora"), str(merged))
    seen = []
    monkeypatch.setattr(merge, "run", lambda cmd, what: seen.append(cmd))

    merge.merge(["--model", "hy-mt2-1.8b"])

    assert seen[0][-1] == "adapter_name_or_path=" + os.path.relpath(run.dir, config.BASE_DIR).replace(os.sep, "/")
    assert json.loads((merged / "resonance_run.json").read_text(encoding="utf-8"))["run"] == run.id


def test_merge_stage_refuses_a_run_that_did_not_finish(monkeypatch, tmp_path, capsys):
    merge = _load_stage("merge")
    run = runs.start_run(str(tmp_path / "lora"), "hy-mt2-1.8b")
    runs.finish_run(run.dir, "failed")
    _use_dirs(monkeypatch, merge, lf_tools.load_profile("hy-mt2-1.8b"), str(tmp_path / "lora"), str(tmp_path / "merged"))
    ran = []
    monkeypatch.setattr(merge, "run", lambda cmd, what: ran.append(cmd))

    with pytest.raises(SystemExit) as exc:
        merge.merge(["--model", "hy-mt2-1.8b"])

    assert exc.value.code == 1 and not ran and "failed" in capsys.readouterr().out


def test_merge_stage_takes_a_named_run(monkeypatch, tmp_path):
    merge = _load_stage("merge")
    base = str(tmp_path / "lora")
    first = runs.start_run(base, "hy-mt2-1.8b")
    runs.finish_run(first.dir, "complete")
    second = runs.start_run(base, "hy-mt2-1.8b", now=datetime_after(first))
    runs.finish_run(second.dir, "complete")
    merged = tmp_path / "merged"
    merged.mkdir()
    _use_dirs(monkeypatch, merge, lf_tools.load_profile("hy-mt2-1.8b"), base, str(merged))
    seen = []
    monkeypatch.setattr(merge, "run", lambda cmd, what: seen.append(cmd))

    merge.merge(["--model", "hy-mt2-1.8b", "--run", first.id])

    assert seen[0][-1].endswith(first.id)


def datetime_after(run):
    from datetime import datetime, timedelta, timezone

    return datetime.now(timezone.utc) + timedelta(days=1)
