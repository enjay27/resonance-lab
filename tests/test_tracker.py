import json

import pytest

import tracker
from run_queue import RunQueue
from tracker import NullTracker, Tracker, sync


class FakeClient:
    """The MLflow server as the sync sees it: records every call; `down` / `fail` make it misbehave."""

    def __init__(self):
        self.down = False
        self.fail_on_call = {}  # method name -> the call number (1-based) that raises
        self.counts = {}
        self.calls = []
        self.remote_runs = {}  # remote id -> local id (a run that already exists on the server)

    def _enter(self, name, *args):
        if self.down:
            raise ConnectionError("NAS is off")
        self.counts[name] = self.counts.get(name, 0) + 1
        if self.fail_on_call.get(name) == self.counts[name]:
            raise RuntimeError(f"{name} failed")
        self.calls.append((name, *args))

    def ping(self):
        self._enter("ping")

    def find_run(self, experiment, local_id):
        self._enter("find_run", local_id)
        return next((remote for remote, local in self.remote_runs.items() if local == local_id), None)

    def create_run(self, experiment, run_name, start_ms, local_id):
        self._enter("create_run", run_name, local_id)
        remote = f"remote-{len(self.remote_runs) + 1}"
        self.remote_runs[remote] = local_id
        return remote

    def log_batch(self, run_id, params=None, tags=None, metrics=None):
        self._enter("log_batch", run_id, dict(params or {}), dict(tags or {}), list(metrics or []))

    def log_artifact(self, run_id, path, artifact_path):
        self._enter("log_artifact", run_id, open(path, encoding="utf-8").read(), artifact_path)

    def set_terminated(self, run_id, status, end_ms):
        self._enter("set_terminated", run_id, status, end_ms)

    def named(self, name):
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def queue(tmp_path):
    return RunQueue(str(tmp_path / "q.json"), str(tmp_path / "q.files"))


@pytest.fixture
def client():
    return FakeClient()


def _run(queue, name="hy", now=1_000_000, **events):
    local = queue.start_run("resonance-lab", name, now_ms=now)
    if "params" in events:
        queue.add_params(local, events["params"])
    if "tags" in events:
        queue.add_tags(local, events["tags"])
    return local


# --- replaying the queue ----------------------------------------------------------------------------------------


def test_a_pending_run_is_created_with_its_local_id_then_filled_and_closed(queue, client):
    local = _run(queue, params={"lr": "1e-05"}, tags={"git.commit": "abc"})
    queue.add_metrics(local, {"eval.chrf": 66.7}, ts_ms=7)
    queue.finish(local, "FINISHED", now_ms=9)

    report = sync(queue, client, log=lambda *a: None)

    assert report.error is None and report.runs_sent == 1
    assert client.named("create_run") == [("create_run", "hy", local)]
    assert ("log_batch", "remote-1", {"lr": "1e-05"}, {}, []) in client.calls
    assert ("log_batch", "remote-1", {}, {"git.commit": "abc"}, []) in client.calls
    assert ("log_batch", "remote-1", {}, {}, [("eval.chrf", 66.7, 7, 0)]) in client.calls
    assert client.named("set_terminated") == [("set_terminated", "remote-1", "FINISHED", 9)]
    assert queue.pending() == []


def test_the_events_are_replayed_in_the_order_they_were_recorded(queue, client):
    local = _run(queue, params={"a": "1"})
    queue.add_tags(local, {"b": "2"})
    queue.add_metrics(local, {"m": 1.0}, ts_ms=1)
    queue.finish(local, "FAILED", now_ms=2)

    sync(queue, client, log=lambda *a: None)

    assert [c[0] for c in client.calls if c[0] in ("log_batch", "set_terminated")] == ["log_batch", "log_batch", "log_batch", "set_terminated"]


def test_params_and_tags_are_sent_in_batches_of_a_hundred(queue, client):
    local = queue.start_run("e", "n")
    queue.add_params(local, {f"p{i}": str(i) for i in range(250)})

    sync(queue, client, log=lambda *a: None)

    assert [len(c[2]) for c in client.named("log_batch")] == [100, 100, 50]


def test_metrics_are_sent_in_batches_of_a_thousand(queue, client, tmp_path):
    local = queue.start_run("e", "n", now_ms=1_000)
    log = tmp_path / "trainer_log.jsonl"
    log.write_text("".join(json.dumps({"current_steps": i, "loss": 1.0, "elapsed_time": "0:00:01"}) + "\n" for i in range(2500)), encoding="utf-8")
    queue.add_step_log(local, str(log))

    sync(queue, client, log=lambda *a: None)

    batches = client.named("log_batch")
    assert [len(b[4]) for b in batches] == [1000, 1000, 500]
    assert batches[0][4][0] == ("loss", 1.0, 2_000, 0)  # (name, value, timestamp = the run's start + elapsed_time, step)


def test_artifacts_go_up_with_their_folder(queue, client, tmp_path):
    local = queue.start_run("e", "n")
    report = tmp_path / "eval.txt"
    report.write_text("chrF 66.7", encoding="utf-8")
    queue.add_artifact(local, str(report), artifact_path="eval")

    sync(queue, client, log=lambda *a: None)

    assert client.named("log_artifact") == [("log_artifact", "remote-1", "chrF 66.7", "eval")]


def test_when_the_nas_is_down_nothing_is_sent_and_everything_stays_queued(queue, client):
    local = _run(queue, params={"a": "1"})
    client.down = True
    lines = []

    report = sync(queue, client, log=lines.append)

    assert report.reachable is False and report.runs_sent == 0 and client.calls == []
    assert [r["local_run_id"] for r in queue.pending()] == [local]
    assert any("unreachable" in line for line in lines) and any("NAS is off" in line for line in lines)


def test_the_backlog_of_earlier_failed_runs_is_sent_oldest_first_before_the_current_one(queue, client):
    older = _run(queue, "monday", now=1, params={"a": "1"})
    newer = _run(queue, "tuesday", now=2, params={"a": "2"})
    current = _run(queue, "today", now=3, params={"a": "3"})

    sync(queue, client, log=lambda *a: None)

    assert [c[1:] for c in client.named("create_run")] == [("monday", older), ("tuesday", newer), ("today", current)]


def test_a_failure_stops_the_queue_in_place_and_the_next_sync_resumes_without_resending(queue, client):
    first = _run(queue, "first", now=1, params={"a": "1"})
    queue.add_tags(first, {"b": "2"})
    second = _run(queue, "second", now=2, params={"c": "3"})
    client.fail_on_call["log_batch"] = 2  # the first run's tags batch

    report = sync(queue, client, log=lambda *a: None)

    assert report.error and "log_batch failed" in report.error and report.runs_sent == 0
    assert client.named("create_run") == [("create_run", "first", first)]  # the second run was not touched: the order is kept
    assert queue.get(second)["remote_id"] is None
    assert [e["sent"] for e in queue.get(first)["events"]] == [True, False]

    report = sync(queue, client, log=lambda *a: None)

    assert report.error is None and report.runs_sent == 2 and queue.pending() == []
    sent = [(c[1], c[2], c[3]) for c in client.named("log_batch")]  # (remote run, params, tags) of every batch that got through
    assert sent == [("remote-1", {"a": "1"}, {}), ("remote-1", {}, {"b": "2"}), ("remote-2", {"c": "3"}, {})]  # nothing twice
    assert len(client.named("create_run")) == 2


def _run_with_artifact(queue, tmp_path, name="hy", now=1):
    """params, then an artifact, then the status that closes the run (the order finish_training records them)."""
    local = _run(queue, name, now=now, params={"a": "1"})
    report = tmp_path / f"{name}.txt"
    report.write_text("log", encoding="utf-8")
    queue.add_artifact(local, str(report), artifact_path="train")
    queue.finish(local, "FINISHED", now_ms=9)
    return local


def test_a_failed_artifact_does_not_stop_the_status_that_closes_the_run(queue, client, tmp_path):
    """MinIO unreachable made the log upload fail; the run stayed RUNNING for good because the status came after it."""
    local = _run_with_artifact(queue, tmp_path)
    client.fail_on_call["log_artifact"] = 1
    lines = []

    report = sync(queue, client, log=lines.append)

    assert client.named("set_terminated") == [("set_terminated", "remote-1", "FINISHED", 9)]
    assert [(e["type"], e["sent"]) for e in queue.get(local)["events"]] == [("params", True), ("artifact", False), ("status", True)]
    assert report.error is None and report.runs_sent == 1
    assert any("artifact" in line and "again" in line for line in lines)


def test_a_failed_artifact_does_not_hold_back_the_runs_after_it(queue, client, tmp_path):
    _run_with_artifact(queue, tmp_path, "first", now=1)
    second = _run(queue, "second", now=2, params={"c": "3"})
    client.fail_on_call["log_artifact"] = 1

    sync(queue, client, log=lambda *a: None)

    assert queue.get(second)["remote_id"] == "remote-2"  # created and filled although the first run's artifact failed


def test_the_failed_artifact_is_retried_at_the_next_sync_and_goes_up_then(queue, client, tmp_path):
    local = _run_with_artifact(queue, tmp_path)
    client.fail_on_call["log_artifact"] = 1
    sync(queue, client, log=lambda *a: None)

    sync(queue, client, log=lambda *a: None)

    assert client.named("log_artifact") == [("log_artifact", "remote-1", "log", "train")] and queue.pending() == []
    assert all(e["sent"] for e in queue.get(local)["events"])


def test_an_artifact_that_keeps_failing_is_given_up_after_three_tries_so_it_cannot_slow_every_start(queue, client, tmp_path):
    local = _run_with_artifact(queue, tmp_path)
    lines = []
    for _ in range(3):
        client.fail_on_call["log_artifact"] = client.counts.get("log_artifact", 0) + 1
        sync(queue, client, log=lines.append)

    artifact = queue.get(local)["events"][1]
    assert artifact["sent"] is True and artifact["gave_up"] is True and artifact["attempts"] == 3
    assert queue.pending() == [] and any("giving up" in line for line in lines)
    calls = client.counts.get("log_artifact", 0)
    sync(queue, client, log=lambda *a: None)
    assert client.counts.get("log_artifact", 0) == calls  # not tried a fourth time


def test_another_failure_still_stops_the_queue_in_place(queue, client, tmp_path):
    first = _run_with_artifact(queue, tmp_path, "first", now=1)
    second = _run(queue, "second", now=2, params={"c": "3"})
    client.fail_on_call["set_terminated"] = 1

    report = sync(queue, client, log=lambda *a: None)

    assert report.error and "set_terminated failed" in report.error and queue.get(second)["remote_id"] is None
    assert [e["sent"] for e in queue.get(first)["events"]] == [True, True, False]


def test_a_run_created_on_the_server_before_a_crash_is_found_again_not_created_twice(queue, client):
    local = _run(queue, params={"a": "1"})
    client.remote_runs["remote-9"] = local  # the server has it: we crashed before writing its id down

    sync(queue, client, log=lambda *a: None)

    assert client.named("create_run") == [] and client.named("log_batch")[0][1] == "remote-9"
    assert queue.get(local)["remote_id"] == "remote-9"


def test_the_server_not_answering_the_ping_is_not_an_error_for_the_caller(queue, client):
    client.down = True

    assert sync(queue, client, log=lambda *a: None).runs_sent == 0  # and nothing was raised


def test_synced_runs_are_pruned_after_a_sync(queue, client, monkeypatch):
    for n in range(4):
        local = _run(queue, f"r{n}", now=n, params={"a": "1"})
        queue.finish(local, "FINISHED")
    monkeypatch.setattr(tracker, "KEEP_SYNCED", 2)

    sync(queue, client, log=lambda *a: None)

    assert len(queue.runs()) == 2


# --- the tracker the stages use -----------------------------------------------------------------------------------


def test_begin_records_locally_first_then_sends_everything_pending(queue, client):
    old = _run(queue, "old", now=1, params={"a": "1"})
    t = Tracker(queue, client, log=lambda *a: None)

    local = t.begin("resonance-lab", "today")
    t.params({"lr": "1e-05"})
    t.flush()

    assert local != old and t.run_id == local
    assert [c[1:] for c in client.named("create_run")] == [("old", old), ("today", local)]
    assert ("log_batch", "remote-2", {"lr": "1e-05"}, {}, []) in client.calls


def test_with_the_nas_down_a_whole_run_is_recorded_locally_and_sent_by_the_next_one(queue, client):
    client.down = True
    t = Tracker(queue, client, log=lambda *a: None)
    local = t.begin("resonance-lab", "offline run")
    t.params({"lr": "1e-05"})
    t.tags({"git.commit": "abc"})
    t.metrics({"eval.chrf": 66.7})
    t.finish("FINISHED")
    assert client.calls == [] and [r["local_run_id"] for r in queue.pending()] == [local]

    client.down = False
    Tracker(queue, client, log=lambda *a: None).begin("resonance-lab", "next run")  # the next start flushes the queue

    assert client.named("create_run")[0][1:] == ("offline run", local)
    assert ("set_terminated", "remote-1", "FINISHED", client.named("set_terminated")[0][3]) in client.calls
    assert queue.get(local)["events"][-1]["sent"] is True


def test_a_later_stage_resumes_the_same_run_by_its_local_id(queue, client):
    first = Tracker(queue, client, log=lambda *a: None)
    local = first.begin("resonance-lab", "run")
    first.params({"a": "1"})

    later = Tracker(queue, client, log=lambda *a: None)
    assert later.resume(local) is True
    later.metrics({"eval.chrf": 1.0})
    later.flush()

    assert len(client.named("create_run")) == 1 and later.run_id == local
    assert Tracker(queue, client, log=lambda *a: None).resume("nope") is False


class _BrokenClient(FakeClient):
    def _enter(self, name, *args):
        raise RuntimeError("boom")


class _BrokenQueue(RunQueue):
    def _fail(self, *args, **kwargs):
        raise OSError("disk")

    start_run = has_run = add_params = add_tags = add_metrics = add_step_log = add_artifact = finish = pending = _fail


@pytest.mark.parametrize("call", [
    lambda t: t.begin("e", "n"), lambda t: t.resume("x"), lambda t: t.params({"a": "1"}), lambda t: t.tags({"a": "1"}),
    lambda t: t.metrics({"a": 1.0}), lambda t: t.step_log("nofile.jsonl"), lambda t: t.artifact("nofile.txt"),
    lambda t: t.finish("FINISHED"), lambda t: t.flush(),
])
def test_no_tracker_call_ever_raises_into_a_training_run(tmp_path, call):
    lines = []
    t = Tracker(_BrokenQueue(str(tmp_path / "q.json"), str(tmp_path / "f")), _BrokenClient(), log=lines.append)
    t.run_id = "x"

    call(t)  # a dead NAS, a full disk or a bug here must not stop the training

    assert any(line.startswith("[tracking]") for line in lines)


def test_the_null_tracker_does_nothing_and_says_it_is_off():
    t = NullTracker()

    assert t.enabled is False and t.begin("e", "n") is None and t.resume("x") is False
    for call in (lambda: t.params({}), lambda: t.tags({}), lambda: t.metrics({}), lambda: t.step_log("x"),
                 lambda: t.artifact("x"), lambda: t.finish("FINISHED"), lambda: t.flush()):
        assert call() is None


def test_from_environment_is_off_without_a_url_and_on_with_one(tmp_path):
    off = tracker.from_environment(environ={}, env_file=str(tmp_path / "none"), queue=None, apply_env=False)
    assert isinstance(off, NullTracker)

    env = tmp_path / ".env.mlflow"
    env.write_text("MLFLOW_TRACKING_URI=http://nas:5050\nMLFLOW_TRACKING_PASSWORD=pw\n", encoding="utf-8")
    on = tracker.from_environment(environ={}, env_file=str(env), queue=RunQueue(str(tmp_path / "q.json"), str(tmp_path / "f")),
                                  client_factory=lambda settings: FakeClient(), apply_env=False)
    assert isinstance(on, Tracker) and on.enabled is True


def test_a_wrong_url_turns_tracking_off_with_a_message_instead_of_failing_the_run(tmp_path):
    env = tmp_path / ".env.mlflow"
    env.write_text("MLFLOW_TRACKING_URI=mlruns\n", encoding="utf-8")
    lines = []

    t = tracker.from_environment(environ={}, env_file=str(env), apply_env=False, log=lines.append)

    assert isinstance(t, NullTracker) and any("http" in line for line in lines)


def test_from_environment_puts_the_client_settings_in_the_process_environment(tmp_path, monkeypatch):
    import os

    env = tmp_path / ".env.mlflow"
    env.write_text("MLFLOW_TRACKING_URI=http://nas:5050\n", encoding="utf-8")
    monkeypatch.setattr(os, "environ", {})  # restored after the test: the real environment is never touched

    tracker.from_environment(environ={}, env_file=str(env), queue=RunQueue(str(tmp_path / "q.json"), str(tmp_path / "f")),
                             client_factory=lambda settings: FakeClient())

    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "10" and os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://nas:5050"


# --- the MLflow adapter (the real client's calls, with a stand-in module) --------------------------------------------


class FakeMlflow:
    """Just enough of `mlflow` for the adapter: MlflowClient and the three entity classes."""

    class entities:
        Metric = lambda key, value, timestamp, step: ("metric", key, value, timestamp, step)  # noqa: E731
        Param = lambda key, value: ("param", key, value)  # noqa: E731
        RunTag = lambda key, value: ("tag", key, value)  # noqa: E731

    class MlflowClient:
        instances = []

        def __init__(self):
            self.calls = []
            self.runs = []
            FakeMlflow.MlflowClient.instances.append(self)

        def search_experiments(self, max_results=None):
            self.calls.append(("search_experiments", max_results))
            return []

        existing = None  # an experiment the server already has: (id, tags)

        def get_experiment_by_name(self, name):
            self.calls.append(("get_experiment_by_name", name))
            if self.existing is None:
                return None
            return type("E", (), {"experiment_id": self.existing[0], "tags": dict(self.existing[1])})

        def create_experiment(self, name, tags=None):
            self.calls.append(("create_experiment", name, dict(tags or {})))
            return "7"

        def set_experiment_tag(self, experiment_id, key, value):
            self.calls.append(("set_experiment_tag", experiment_id, key, value))

        def create_run(self, experiment_id, start_time=None, tags=None, run_name=None):
            self.calls.append(("create_run", experiment_id, start_time, dict(tags)))
            return type("R", (), {"info": type("I", (), {"run_id": "run-1"})})

        def search_runs(self, experiment_ids, filter_string="", max_results=None):
            self.calls.append(("search_runs", list(experiment_ids), filter_string))
            return self.runs

        def log_batch(self, run_id, metrics=(), params=(), tags=()):
            self.calls.append(("log_batch", run_id, list(metrics), list(params), list(tags)))

        def log_artifact(self, run_id, local_path, artifact_path=None):
            self.calls.append(("log_artifact", run_id, local_path, artifact_path))

        def set_terminated(self, run_id, status=None, end_time=None):
            self.calls.append(("set_terminated", run_id, status, end_time))


@pytest.fixture
def adapter():
    FakeMlflow.MlflowClient.instances.clear()
    a = tracker.MlflowAdapter(mlflow=FakeMlflow)
    return a, FakeMlflow.MlflowClient.instances[0]


def test_the_adapter_pings_with_an_authenticated_call(adapter):
    a, client = adapter

    a.ping()

    assert client.calls == [("search_experiments", 1)]


def test_the_adapter_creates_the_experiment_once_and_the_run_with_its_local_id_and_backdated_start(adapter):
    a, client = adapter

    remote = a.create_run("resonance-lab", "hy-1", 1_700_000_000_000, "local-1")

    assert remote == "run-1"
    assert ("create_experiment", "resonance-lab", {"mlflow.experimentKind": "finetuning"}) in client.calls
    assert ("create_run", "7", 1_700_000_000_000, {"mlflow.runName": "hy-1", "local_run_id": "local-1"}) in client.calls
    a.create_run("resonance-lab", "hy-2", 1, "local-2")
    assert [c[0] for c in client.calls].count("create_experiment") == 1


def test_a_new_experiment_is_marked_as_fine_tuning_so_the_ui_shows_training_runs_not_evaluation_runs(adapter):
    a, client = adapter

    a.create_run("resonance-lab", "hy-1", 1, "local-1")

    assert ("create_experiment", "resonance-lab", {"mlflow.experimentKind": "finetuning"}) in client.calls


def test_an_existing_experiment_without_the_kind_gets_it_once(adapter):
    a, client = adapter
    client.existing = ("3", {})

    a.create_run("resonance-lab", "hy-1", 1, "local-1")
    a.create_run("resonance-lab", "hy-2", 1, "local-2")

    assert client.calls.count(("set_experiment_tag", "3", "mlflow.experimentKind", "finetuning")) == 1
    assert not [c for c in client.calls if c[0] == "create_experiment"]


def test_an_experiment_whose_kind_was_chosen_in_the_ui_is_left_alone(adapter):
    a, client = adapter
    client.existing = ("3", {"mlflow.experimentKind": "custom_model_development"})

    a.create_run("resonance-lab", "hy-1", 1, "local-1")

    assert not [c for c in client.calls if c[0] == "set_experiment_tag"]


def test_the_adapter_finds_a_run_by_its_local_id_tag(adapter):
    a, client = adapter
    assert a.find_run("resonance-lab", "local-1") is None  # no experiment yet: nothing to find

    client.runs = [type("R", (), {"info": type("I", (), {"run_id": "found"})})]
    client.calls.clear()
    client.get_experiment_by_name = lambda name: type("E", (), {"experiment_id": "7"})

    assert a.find_run("resonance-lab", "local-1") == "found"
    assert ("search_runs", ["7"], "tags.local_run_id = 'local-1'") in client.calls


def test_the_adapter_turns_plain_data_into_mlflow_entities(adapter):
    a, client = adapter

    a.log_batch("run-1", params={"lr": "1"}, tags={"t": "v"}, metrics=[("loss", 0.5, 1000, 3)])
    a.log_artifact("run-1", "/tmp/r.txt", "eval")
    a.set_terminated("run-1", "FINISHED", 5000)

    assert ("log_batch", "run-1", [("metric", "loss", 0.5, 1000, 3)], [("param", "lr", "1")], [("tag", "t", "v")]) in client.calls
    assert ("log_artifact", "run-1", "/tmp/r.txt", "eval") in client.calls
    assert ("set_terminated", "run-1", "FINISHED", 5000) in client.calls


def test_the_queue_needs_no_database_package():
    import os
    from config import BASE_DIR

    for name in ("requirements-dev.txt", "requirements-llamafactory.txt"):
        with open(os.path.join(BASE_DIR, name), encoding="utf-8") as f:
            assert "tinydb" not in f.read().lower(), name  # the journal is plain JSON lines (run_queue.py)


def test_trainer_log_points_reach_mlflow_with_the_timestamp_and_the_step_in_the_right_places(tmp_path):
    # Found by a run against a real server: step_metrics and the sender once disagreed on the order, so MLflow got the
    # step as the timestamp and the other way round. This test crosses the three modules (queue -> sync -> adapter).
    queue = RunQueue(str(tmp_path / "q.json"), str(tmp_path / "f"))
    log = tmp_path / "trainer_log.jsonl"
    log.write_text(json.dumps({"current_steps": 7, "loss": 1.5, "elapsed_time": "0:00:02"}) + "\n", encoding="utf-8")
    local = queue.start_run("e", "n", now_ms=1_000_000)
    queue.add_step_log(local, str(log))
    FakeMlflow.MlflowClient.instances.clear()

    sync(queue, tracker.MlflowAdapter(mlflow=FakeMlflow), log=lambda *a: None)

    (call,) = [c for c in FakeMlflow.MlflowClient.instances[-1].calls if c[0] == "log_batch"]
    assert call[2] == [("metric", "loss", 1.5, 1_002_000, 7)]  # key, value, timestamp (start + 2 s), step


def test_an_unreachable_server_costs_one_short_log_line_not_the_whole_error_chain(queue, client):
    _run(queue, params={"a": "1"})
    long_error = "API request to http://nas:5050/api failed with exception HTTPConnectionPool: " + "Max retries exceeded " * 40

    def down():
        raise ConnectionError(long_error)

    client.ping = down
    lines = []

    sync(queue, client, log=lines.append)

    assert len(lines) == 1 and len(lines[0]) < 260 and lines[0].startswith("[tracking] MLflow unreachable")
