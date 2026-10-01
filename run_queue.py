"""The local record of every run, written before anything is sent to the MLflow server. No network, no mlflow.

A document store (TinyDB: schemaless JSON in `.run.result.backup.json`) holds one document per run. A run is a list of
events in the order the stages recorded them (params, tags, metrics, a trainer log, an artifact, the final status); each
event is marked sent once the server has it. So a run made while the NAS was off, or a sync that stopped halfway, is
simply the events that are still unsent: tracker.sync replays them as a queue, oldest run first, without sending
anything twice. Small files are copied into `.run.result.backup.files/<run>/` so the originals may be deleted.
"""

import copy
import json
import os
import re
import shutil
import time
import uuid

from tinydb import Query, TinyDB
from tinydb.storages import Storage

from config import RUN_QUEUE_FILES, RUN_QUEUE_PATH

_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")


class CorruptQueue(Exception):
    """The queue file is not valid JSON (a write was cut off)."""


class AtomicStorage(Storage):
    """TinyDB's JSON storage writes in place, so a crash mid-write corrupts the whole file and every pending run with
    it. This one writes a temp file and replaces the old one in a single step."""

    def __init__(self, path):
        self.path = path

    def read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            return None
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except ValueError as e:
            raise CorruptQueue(str(e)) from None

    def write(self, data):
        temp = self.path + ".tmp"
        try:
            with open(temp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.remove(temp)

    def close(self):
        pass


def _now_ms():
    return int(time.time() * 1000)


class RunQueue:
    def __init__(self, path=RUN_QUEUE_PATH, files_dir=RUN_QUEUE_FILES):
        self.path, self.files_dir, self.warning = path, files_dir, None
        self._db = self._open()
        self._runs_table = self._db.table("runs")

    def _open(self):
        try:
            db = TinyDB(self.path, storage=AtomicStorage)
            db.table("runs").all()  # reads the file now: a corrupt one fails here, not in the middle of a run
            return db
        except CorruptQueue as e:
            kept = f"{self.path}.corrupt-{int(time.time())}"
            os.replace(self.path, kept)  # never lose it silently, never stop a training because of it
            self.warning = f"queue file {self.path} was corrupt ({e}); kept as {kept}, starting a new one"
            return TinyDB(self.path, storage=AtomicStorage)

    # --- reading ---

    def runs(self):
        """Every run document, oldest start first."""
        return sorted((copy.deepcopy(dict(doc)) for doc in self._runs_table.all()), key=lambda doc: doc["start_time_ms"])

    def has_run(self, local_run_id):
        return self._runs_table.contains(Query().local_run_id == local_run_id)

    def get(self, local_run_id):
        """A copy of the run's document (changing it changes nothing)."""
        doc = self._runs_table.get(Query().local_run_id == local_run_id)
        if doc is None:
            raise KeyError(local_run_id)
        return copy.deepcopy(dict(doc))

    def pending(self):
        """The runs that still have something to send, oldest first: not created on the server yet, or with an unsent event."""
        return [run for run in self.runs() if run["remote_id"] is None or any(not e["sent"] for e in run["events"])]

    def file_path(self, local_run_id, name):
        return os.path.join(self.files_dir, local_run_id, name)

    # --- writing: every call is on disk before it returns ---

    def _update(self, local_run_id, change):
        if not self._runs_table.update(change, Query().local_run_id == local_run_id):
            raise KeyError(local_run_id)

    def start_run(self, experiment, run_name, now_ms=None, local_run_id=None):
        """Record a new run (or return the id of the run that already has `local_run_id`)."""
        if local_run_id is not None:
            if not _SAFE_ID.fullmatch(local_run_id):
                raise ValueError(f"local_run_id {local_run_id!r} may only hold letters, digits, - and _")
            if self.has_run(local_run_id):
                return local_run_id
        local_run_id = local_run_id or uuid.uuid4().hex
        self._runs_table.insert({"local_run_id": local_run_id, "experiment": experiment, "run_name": run_name,
                                 "start_time_ms": now_ms if now_ms is not None else _now_ms(), "remote_id": None, "events": []})
        return local_run_id

    def _add_event(self, local_run_id, event):
        self._update(local_run_id, lambda doc: doc["events"].append({**event, "sent": False}))

    def add_params(self, local_run_id, params):
        self._add_event(local_run_id, {"type": "params", "params": dict(params)})

    def add_tags(self, local_run_id, tags):
        self._add_event(local_run_id, {"type": "tags", "tags": dict(tags)})

    def add_metrics(self, local_run_id, metrics, ts_ms=None):
        self._add_event(local_run_id, {"type": "metrics", "metrics": dict(metrics), "ts_ms": ts_ms if ts_ms is not None else _now_ms()})

    def _copy_in(self, local_run_id, path):
        number = len(self.get(local_run_id)["events"])
        name = f"{number}-{os.path.basename(path)}"
        os.makedirs(os.path.join(self.files_dir, local_run_id), exist_ok=True)
        shutil.copyfile(path, self.file_path(local_run_id, name))
        return name

    def add_step_log(self, local_run_id, path):
        """The trainer's trainer_log.jsonl: its points are sent after training (tracking.step_metrics)."""
        self._add_event(local_run_id, {"type": "step_log", "file": self._copy_in(local_run_id, path)})

    def add_artifact(self, local_run_id, path, artifact_path=None):
        self._add_event(local_run_id, {"type": "artifact", "file": self._copy_in(local_run_id, path), "artifact_path": artifact_path})

    def finish(self, local_run_id, status, now_ms=None):
        self._add_event(local_run_id, {"type": "status", "status": status, "ts_ms": now_ms if now_ms is not None else _now_ms()})

    def set_remote_id(self, local_run_id, remote_id):
        self._update(local_run_id, lambda doc: doc.update(remote_id=remote_id))

    def mark_sent(self, local_run_id, index):
        self._update(local_run_id, lambda doc: doc["events"][index].update(sent=True))

    def prune(self, keep=20):
        """Forget the oldest runs the server has completely (created, every event sent, closed by a status event -- later stages
        may have added events after it), keeping the newest `keep`."""
        done = [run for run in self.runs()
                if run["remote_id"] is not None and run["events"] and all(e["sent"] for e in run["events"])
                and any(e["type"] == "status" for e in run["events"])]
        old = done[:-keep] if keep else done
        for run in old:
            self._runs_table.remove(Query().local_run_id == run["local_run_id"])
            shutil.rmtree(os.path.join(self.files_dir, run["local_run_id"]), ignore_errors=True)
        return len(old)
