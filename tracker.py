"""Sends runs to the MLflow server on the NAS, without ever letting a failure reach a training run.

Flow (maintainer's design): a stage records its run in the local queue FIRST (run_queue.py: on disk before anything is
sent); then it tries the server. If the NAS answers, every run still unsent is replayed as a queue, oldest first, and
the current one with it; if not, everything stays queued and the next start sends it. Tracking never raises: a dead
NAS, a full disk or a bug here costs a log line, not a training. `NullTracker` is what you get when tracking is off.
"""

import functools
import importlib
import os
from typing import NamedTuple, Optional

import tracking
from config import MLFLOW_ENV_FILE
from run_queue import RunQueue

KEEP_SYNCED = 20  # runs the server has completely stay in the queue file for a while, then are pruned
ARTIFACT_TRIES = 3  # an artifact that fails this many times is given up (the file stays in the queue's files folder)


def _short(error, limit=100):
    """One line of an exception: MLflow's connection errors carry the whole retry chain."""
    text = " ".join(str(error).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


class SyncReport(NamedTuple):
    reachable: bool
    runs_sent: int
    error: Optional[str]


def _send_event(queue, client, run, remote_id, event):
    local = run["local_run_id"]
    kind = event["type"]
    if kind == "params":
        for chunk in tracking.chunks(list(event["params"].items()), tracking.MAX_PARAMS_TAGS_PER_BATCH):
            client.log_batch(remote_id, params=dict(chunk))
    elif kind == "tags":
        for chunk in tracking.chunks(list(event["tags"].items()), tracking.MAX_PARAMS_TAGS_PER_BATCH):
            client.log_batch(remote_id, tags=dict(chunk))
    elif kind == "metrics":
        points = [(name, float(value), event["ts_ms"], 0) for name, value in event["metrics"].items()]
        for chunk in tracking.chunks(points, tracking.MAX_METRICS_PER_BATCH):
            client.log_batch(remote_id, metrics=chunk)
    elif kind == "step_log":
        with open(queue.file_path(local, event["file"]), encoding="utf-8") as f:
            points = tracking.step_metrics(f.read().splitlines(), run["start_time_ms"])
        for chunk in tracking.chunks(points, tracking.MAX_METRICS_PER_BATCH):
            client.log_batch(remote_id, metrics=chunk)
    elif kind == "artifact":
        client.log_artifact(remote_id, queue.file_path(local, event["file"]), event["artifact_path"])
    elif kind == "status":
        client.set_terminated(remote_id, event["status"], event["ts_ms"])
    else:
        raise ValueError(f"unknown event type {kind!r}")


def _send_run(queue, client, run, log=print):
    """Send one run's unsent events in order. A failed artifact is only logged and retried at the next start (it is
    optional: it must not keep the run from being closed, nor the runs after it from being sent); any other failure raises."""
    local = run["local_run_id"]
    remote = run["remote_id"]
    if remote is None:
        # A crash between "created on the server" and "id written down" must not create the run twice: look it up by tag first.
        remote = client.find_run(run["experiment"], local) or client.create_run(run["experiment"], run["run_name"], run["start_time_ms"], local)
        queue.set_remote_id(local, remote)
    for index, event in enumerate(run["events"]):
        if event["sent"]:
            continue
        try:
            _send_event(queue, client, run, remote, event)
        except Exception as e:
            if event["type"] != "artifact":
                raise
            name = os.path.basename(event["file"])
            if queue.fail_event(local, index, ARTIFACT_TRIES) >= ARTIFACT_TRIES:
                log(f"[tracking] artifact {name} of run {run['run_name']} failed {ARTIFACT_TRIES} times ({_short(e)}); giving up on it")
            else:
                log(f"[tracking] artifact {name} of run {run['run_name']} not uploaded ({_short(e)}); the run goes on, it is tried again at the next start")
            continue
        queue.mark_sent(local, index)  # marked at once: a stop halfway resumes at the next event


def sync(queue, client, log=print):
    """Replay every pending run to the server, oldest first; stop at the first failure so the order is kept.

    Never raises for a server problem: an unreachable server or a failed send is in the report and the log, and
    everything unsent stays queued.
    """
    pending = queue.pending()
    if not pending:
        return SyncReport(True, 0, None)
    try:
        client.ping()
    except Exception as e:
        log(f"[tracking] MLflow unreachable ({_short(e)}); {len(pending)} run(s) stay queued in {os.path.basename(queue.path)} for the next start")
        return SyncReport(False, 0, str(e))
    sent = 0
    for run in pending:
        try:
            _send_run(queue, client, run, log)
        except Exception as e:
            log(f"[tracking] sending run {run['run_name']} stopped: {_short(e)}; it and the runs after it stay queued")
            return SyncReport(True, sent, str(e))
        sent += 1
    queue.prune(KEEP_SYNCED)
    log(f"[tracking] sent {sent} run(s) to MLflow")
    return SyncReport(True, sent, None)


def _never_raises(method):
    """A tracker call that cannot stop a training: any exception becomes one log line."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            self.log(f"[tracking] {method.__name__} failed ({type(e).__name__}: {e}); training goes on without it")
            return None
    return wrapper


class Tracker:
    enabled = True

    def __init__(self, queue, client, log=print):
        self.queue, self.client, self.log = queue, client, log
        self.run_id = None  # the local id of the run this process records into
        if queue.warning:
            log(f"[tracking] {queue.warning}")

    @_never_raises
    def begin(self, experiment, run_name, local_run_id=None):
        """Start (or continue) a run: recorded locally first, then every pending run is sent if the server answers."""
        self.run_id = self.queue.start_run(experiment, run_name, local_run_id=local_run_id)
        self.flush()
        return self.run_id

    @_never_raises
    def resume(self, local_run_id):
        """Continue a run an earlier stage started (its id travels in the run's directory / environment)."""
        if not self.queue.has_run(local_run_id):
            return False
        self.run_id = local_run_id
        return True

    @_never_raises
    def params(self, params):
        if self.run_id:
            self.queue.add_params(self.run_id, params)

    @_never_raises
    def tags(self, tags):
        if self.run_id:
            self.queue.add_tags(self.run_id, tags)

    @_never_raises
    def metrics(self, metrics):
        if self.run_id:
            self.queue.add_metrics(self.run_id, metrics)

    @_never_raises
    def step_log(self, path):
        if self.run_id:
            self.queue.add_step_log(self.run_id, path)

    @_never_raises
    def artifact(self, path, artifact_path=None):
        if self.run_id:
            self.queue.add_artifact(self.run_id, path, artifact_path)

    @_never_raises
    def finish(self, status):
        if self.run_id:
            self.queue.finish(self.run_id, status)

    @_never_raises
    def flush(self):
        """Send everything pending to the server, if it answers."""
        return sync(self.queue, self.client, self.log)


class NullTracker:
    """Tracking is off (no URL, RESONANCE_MLFLOW=0, mlflow not installed): every call does nothing."""

    enabled = False
    run_id = None

    def _nothing(self, *args, **kwargs):
        return None

    begin = params = tags = metrics = step_log = artifact = finish = flush = _nothing

    def resume(self, local_run_id):
        return False


class MlflowAdapter:
    """The few calls sync needs, on the real `mlflow` client (imported here, never by the gate): plain data in,
    MLflow entities out. `mlflow` is injectable so the gate tests it with a stand-in."""

    def __init__(self, mlflow=None):
        self._mlflow = mlflow or importlib.import_module("mlflow")
        self._client = self._mlflow.MlflowClient()
        self._experiments = {}

    def ping(self):
        self._client.search_experiments(max_results=1)  # an authenticated call: it also proves the credentials work

    def _experiment_id(self, name, create=False):
        if name not in self._experiments:
            found = self._client.get_experiment_by_name(name)
            if found is not None:
                self._experiments[name] = found.experiment_id
                if tracking.EXPERIMENT_KIND_TAG not in (getattr(found, "tags", None) or {}):
                    try:  # a nicety (training runs view): never a reason to stop sending
                        self._client.set_experiment_tag(found.experiment_id, tracking.EXPERIMENT_KIND_TAG, tracking.EXPERIMENT_KIND)
                    except Exception:  # noqa: BLE001
                        pass
            elif create:
                self._experiments[name] = self._client.create_experiment(
                    name, tags={tracking.EXPERIMENT_KIND_TAG: tracking.EXPERIMENT_KIND})
        return self._experiments.get(name)

    def find_run(self, experiment, local_id):
        experiment_id = self._experiment_id(experiment)
        if experiment_id is None:
            return None
        runs = self._client.search_runs([experiment_id], f"tags.local_run_id = '{local_id}'", max_results=1)
        return runs[0].info.run_id if runs else None

    def create_run(self, experiment, run_name, start_ms, local_id):
        run = self._client.create_run(self._experiment_id(experiment, create=True), start_time=start_ms,
                                      tags={"mlflow.runName": run_name, "local_run_id": local_id})
        return run.info.run_id

    def log_batch(self, run_id, params=None, tags=None, metrics=None):
        entities = self._mlflow.entities
        self._client.log_batch(
            run_id,
            metrics=[entities.Metric(name, value, ts, step) for name, value, ts, step in (metrics or [])],
            params=[entities.Param(key, value) for key, value in (params or {}).items()],
            tags=[entities.RunTag(key, value) for key, value in (tags or {}).items()],
        )

    def log_artifact(self, run_id, path, artifact_path):
        self._client.log_artifact(run_id, path, artifact_path)

    def set_terminated(self, run_id, status, end_ms):
        self._client.set_terminated(run_id, status, end_ms)


def from_environment(environ=None, env_file=MLFLOW_ENV_FILE, queue=None, client_factory=None, apply_env=True, log=print):
    """The tracker for this process: a NullTracker unless `.env.mlflow` (or the environment) names the NAS server.

    A wrong URL or a missing mlflow package turns tracking off with a message; it never stops the run.
    """
    try:
        settings = tracking.tracking_settings(environ, env_file)
    except ValueError as e:
        log(f"[tracking] off: {e}")
        return NullTracker()
    if settings is None:
        return NullTracker()
    if apply_env:
        os.environ.update(tracking.client_environment(settings))
    try:
        client = (client_factory or (lambda _settings: MlflowAdapter()))(settings)
    except ImportError:
        log("[tracking] off: mlflow-skinny is not installed in this environment (pip install -r requirements-llamafactory.txt)")
        return NullTracker()
    return Tracker(queue or RunQueue(), client, log=log)
