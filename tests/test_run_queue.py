import json
import os

import pytest

import run_queue
from run_queue import RunQueue


@pytest.fixture
def queue(tmp_path):
    return RunQueue(str(tmp_path / ".run.result.backup.json"), str(tmp_path / ".run.result.backup.files"))


def _new(queue, name="hy-20261001", now=1_000):
    return queue.start_run("resonance-lab", name, now_ms=now)


def test_a_started_run_is_on_disk_at_once_with_a_fresh_id(queue, tmp_path):
    first, second = _new(queue), _new(queue)

    assert first != second and len(first) >= 32
    on_disk = json.loads((tmp_path / ".run.result.backup.json").read_text(encoding="utf-8"))
    assert {doc["local_run_id"] for doc in on_disk["runs"].values()} == {first, second}


def test_a_run_remembers_its_experiment_name_and_start(queue):
    run = queue.get(_new(queue, "my-run", now=1234))

    assert run["experiment"] == "resonance-lab" and run["run_name"] == "my-run" and run["start_time_ms"] == 1234
    assert run["remote_id"] is None and run["events"] == []


def test_a_given_local_id_is_kept_so_later_stages_can_resume_the_run(queue):
    local = queue.start_run("e", "n", local_run_id="abc", now_ms=1)

    assert local == "abc" and queue.has_run("abc") and not queue.has_run("zzz")
    assert queue.start_run("e", "n", local_run_id="abc", now_ms=2) == "abc" and len(queue.pending()) == 1  # not duplicated


def test_what_a_stage_records_is_appended_in_order_and_unsent(queue):
    local = _new(queue)

    queue.add_params(local, {"lr": "1e-05"})
    queue.add_tags(local, {"git.commit": "abc"})
    queue.add_metrics(local, {"eval.chrf": 66.7}, ts_ms=5000)
    queue.finish(local, "FINISHED", now_ms=9000)

    events = queue.get(local)["events"]
    assert [e["type"] for e in events] == ["params", "tags", "metrics", "status"]
    assert all(e["sent"] is False for e in events)
    assert events[0]["params"] == {"lr": "1e-05"} and events[2]["ts_ms"] == 5000 and events[3]["status"] == "FINISHED"


def test_get_hands_out_a_copy_not_the_stored_document(queue):
    local = _new(queue)

    queue.add_tags(local, {"a": "b"})
    doc = queue.get(local)
    doc["events"][0]["something_new"] = {"nested": [1, 2]}

    assert queue.get(local)["events"][0].get("something_new") is None  # get() hands out a copy, not the stored document


def test_files_are_copied_into_the_queue_so_the_originals_may_go(queue, tmp_path):
    source = tmp_path / "report.txt"
    source.write_text("chrF 66.7", encoding="utf-8")
    log = tmp_path / "trainer_log.jsonl"
    log.write_text('{"current_steps": 1}\n', encoding="utf-8")
    local = _new(queue)

    queue.add_artifact(local, str(source), artifact_path="eval")
    queue.add_step_log(local, str(log))
    source.unlink()
    log.unlink()

    artifact, step_log = queue.get(local)["events"]
    assert artifact["type"] == "artifact" and artifact["artifact_path"] == "eval"
    assert open(queue.file_path(local, artifact["file"]), encoding="utf-8").read() == "chrF 66.7"
    assert step_log["type"] == "step_log" and open(queue.file_path(local, step_log["file"]), encoding="utf-8").read() == '{"current_steps": 1}\n'


def test_two_files_with_the_same_name_do_not_overwrite_each_other(queue, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "report.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b" / "report.txt").write_text("B", encoding="utf-8")
    local = _new(queue)

    queue.add_artifact(local, str(tmp_path / "a" / "report.txt"))
    queue.add_artifact(local, str(tmp_path / "b" / "report.txt"))

    first, second = queue.get(local)["events"]
    assert open(queue.file_path(local, first["file"]), encoding="utf-8").read() == "A"
    assert open(queue.file_path(local, second["file"]), encoding="utf-8").read() == "B"
    assert first["artifact_path"] is None and first["file"] != second["file"]


def test_pending_lists_runs_oldest_first_and_only_those_with_work_left(queue):
    old, new = _new(queue, "old", now=1), _new(queue, "new", now=2)
    queue.add_params(old, {"a": "1"})
    queue.add_params(new, {"b": "2"})

    assert [r["run_name"] for r in queue.pending()] == ["old", "new"]

    queue.set_remote_id(old, "remote-old")
    queue.mark_sent(old, 0)
    assert [r["run_name"] for r in queue.pending()] == ["new"]  # old is created remotely and has nothing unsent


def test_a_run_not_yet_created_on_the_server_is_pending_even_without_events(queue):
    local = _new(queue)

    assert [r["local_run_id"] for r in queue.pending()] == [local]

    queue.set_remote_id(local, "r1")
    assert queue.pending() == []


def test_the_queue_survives_a_restart_with_what_was_sent_and_what_was_not(tmp_path):
    path, files = str(tmp_path / "q.json"), str(tmp_path / "q.files")
    first = RunQueue(path, files)
    local = _new(first)
    first.add_params(local, {"a": "1"})
    first.add_tags(local, {"b": "2"})
    first.set_remote_id(local, "r1")
    first.mark_sent(local, 0)

    again = RunQueue(path, files).get(local)

    assert again["remote_id"] == "r1" and [e["sent"] for e in again["events"]] == [True, False]


def test_a_crash_while_writing_leaves_the_previous_file_intact(queue, tmp_path, monkeypatch):
    local = _new(queue)
    path = tmp_path / ".run.result.backup.json"
    before = path.read_text(encoding="utf-8")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(run_queue.os, "replace", boom)
    with pytest.raises(OSError):
        queue.add_params(local, {"a": "1"})

    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []  # no stray temp file


def test_a_corrupt_queue_file_is_set_aside_not_lost_and_not_fatal(tmp_path):
    path = tmp_path / "q.json"
    path.write_text('{"runs": {"1": {"local_run_id": "x", ', encoding="utf-8")  # cut off mid-write

    queue = RunQueue(str(path), str(tmp_path / "q.files"))

    assert queue.pending() == [] and queue.warning and "corrupt" in queue.warning
    kept = [p for p in tmp_path.iterdir() if ".corrupt-" in p.name]
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8").startswith('{"runs"')
    queue.start_run("e", "n")  # and it works again
    assert len(queue.pending()) == 1


def test_a_failing_optional_event_is_counted_and_given_up_after_the_limit(queue, tmp_path):
    local = _new(queue, "r", now=1)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    queue.add_artifact(local, str(tmp_path / "f.txt"))

    assert queue.fail_event(local, 0, give_up_after=3) == 1 and queue.get(local)["events"][0]["sent"] is False
    assert queue.fail_event(local, 0, give_up_after=3) == 2
    assert queue.fail_event(local, 0, give_up_after=3) == 3
    event = queue.get(local)["events"][0]
    assert event["sent"] is True and event["gave_up"] is True and event["attempts"] == 3


def test_a_run_that_later_stages_continued_is_pruned_once_everything_is_sent(queue):
    """Merge, GGUF and eval add events after the training closed the run, so the last event is no longer its status."""
    done = _new(queue, "continued", now=1)
    queue.finish(done, "FINISHED")
    queue.add_tags(done, {"stage.merge": "done"})
    queue.set_remote_id(done, "r1")
    for i in range(len(queue.get(done)["events"])):
        queue.mark_sent(done, i)
    unfinished = _new(queue, "never closed", now=2)
    queue.add_params(unfinished, {"a": "1"})
    queue.set_remote_id(unfinished, "r2")
    queue.mark_sent(unfinished, 0)

    assert queue.prune(keep=0) == 1
    assert not queue.has_run(done) and queue.has_run(unfinished)  # a run no stage closed is kept


def test_synced_runs_are_pruned_to_the_newest_few_with_their_files(queue, tmp_path):
    ids = []
    for n in range(5):
        local = _new(queue, f"run{n}", now=n)
        queue.add_params(local, {"a": "1"})
        (tmp_path / f"f{n}.txt").write_text("x", encoding="utf-8")
        queue.add_artifact(local, str(tmp_path / f"f{n}.txt"))
        queue.finish(local, "FINISHED")
        queue.set_remote_id(local, f"r{n}")
        for i in range(len(queue.get(local)["events"])):
            queue.mark_sent(local, i)
        ids.append(local)
    waiting = _new(queue, "unsent", now=99)
    queue.add_params(waiting, {"a": "1"})

    removed = queue.prune(keep=2)

    assert removed == 3
    assert not queue.has_run(ids[0]) and not queue.has_run(ids[2]) and queue.has_run(ids[3]) and queue.has_run(ids[4])
    assert queue.has_run(waiting)  # never prune what has not been sent
    assert not os.path.exists(os.path.join(str(tmp_path / ".run.result.backup.files"), ids[0]))
    assert os.path.isdir(os.path.join(str(tmp_path / ".run.result.backup.files"), ids[4]))


def test_the_default_locations_are_the_gitignored_ones():
    import config

    assert config.RUN_QUEUE_PATH == os.path.join(config.BASE_DIR, ".run.result.backup.json")
    assert config.RUN_QUEUE_FILES == os.path.join(config.BASE_DIR, ".run.result.backup.files")


@pytest.mark.parametrize("bad", ["a'b", "x y", "a;b", ""])
def test_a_given_local_id_must_be_safe_to_put_in_a_search_filter(queue, bad):
    with pytest.raises(ValueError, match="local_run_id"):
        queue.start_run("e", "n", local_run_id=bad)
