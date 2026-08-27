"""Executor backend for the Miyabi-C PBS cluster.

A batch of tasks becomes one or more PBS jobs, one node per task (`pbsdsh`), sized to
the group's limits. Missing datasets are built first. Results are read back from the
DBTune history files. Nothing above this module knows about nodes, queues or qsub.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass

from ..executor import Executor, ResultCache
from ..tasks import EvalResult, TuneResult, TuneTask
from . import taskgen

SUCCESS = 0   # trial_state of a usable DBTune history row


@dataclass
class Run:
    """One task materialized on disk, to be executed on one node."""
    task: object
    dir: str
    spec: dict

    @property
    def walltime_s(self) -> int:
        return self.spec["timeout_s"] + 30 * 60      # datadir copy + reset + cleanup margin


@dataclass
class Job:
    runs: list
    job_id: str | None = None

    @property
    def walltime_s(self) -> int:
        return max(r.walltime_s for r in self.runs)


class MiyabiExecutor(Executor):
    def __init__(self, root: str, out_dir: str, project: str = "xg26g002", queue: str = "regular-c",
                 max_nodes: int = 16, max_jobs: int = 2, poll_s: int = 60, tune_hours: float = 24,
                 eval_minutes_per_config: float = 8, prep_hours: float = 2, online: bool = True,
                 knob_file: str = "./experiment/gen_knobs/mysql_perf_8.0_online.json",
                 dry_run: bool = False, cache: ResultCache | None = None):
        super().__init__(cache)
        self.root, self.out_dir, self.project, self.queue = root, out_dir, project, queue
        self.max_nodes, self.max_jobs, self.poll_s = max_nodes, max_jobs, poll_s
        self.tune_hours, self.eval_minutes_per_config, self.prep_hours = tune_hours, eval_minutes_per_config, prep_hours
        self.online, self.knob_file, self.dry_run = online, knob_file, dry_run
        self.submitted: list[Job] = []

    @classmethod
    def from_config(cls, cfg: dict, out_dir: str, cache: ResultCache) -> "MiyabiExecutor":
        return cls(out_dir=out_dir, cache=cache, **cfg)

    # ---- Executor API -------------------------------------------------------------------
    def _run(self, tasks: list) -> list:
        runs = [self._materialize(t) for t in tasks]
        self._ensure_datasets(runs)
        self._execute(self._pack(runs))
        return [self._collect(r) for r in runs]

    # ---- materialize ---------------------------------------------------------------------
    def _materialize(self, task) -> Run:
        run_dir = os.path.join(self.out_dir, "runs", task.key())
        spec = taskgen.write_run(self.root, task, run_dir, self.knob_file, self.online, self._timeout_s(task))
        return Run(task, run_dir, spec)

    def _timeout_s(self, task) -> int:
        if isinstance(task, TuneTask):
            return int(self.tune_hours * 3600)
        return int((len(task.configs) * self.eval_minutes_per_config + 15) * 60)

    def _ensure_datasets(self, runs: list) -> None:
        missing = {r.spec["snapshot"]: r.spec["dataset"] for r in runs
                   if not os.path.isdir(os.path.join(self.root, "mysql_build", r.spec["snapshot"]))}
        if not missing:
            return
        prep_runs = []
        for name, dataset in missing.items():
            run_dir = os.path.join(self.out_dir, "runs", "dataset_" + name)
            spec = taskgen.write_prep_run(self.root, taskgen.Dataset(**dataset), run_dir, int(self.prep_hours * 3600))
            prep_runs.append(Run(None, run_dir, spec))
        self._execute(self._pack(prep_runs))
        still_missing = [n for n in missing if not os.path.isdir(os.path.join(self.root, "mysql_build", n))]
        if still_missing and not self.dry_run:
            raise RuntimeError("dataset build failed for %s" % still_missing)

    # ---- jobs ----------------------------------------------------------------------------
    def _pack(self, runs: list) -> list[Job]:
        return [Job(runs[i:i + self.max_nodes]) for i in range(0, len(runs), self.max_nodes)]

    def _execute(self, jobs: list[Job]) -> None:
        """Submit up to max_jobs at a time; poll until every job has left the queue."""
        pending, active = list(jobs), []
        while pending or active:
            while pending and len(active) < self.max_jobs:
                job = pending.pop(0)
                self._submit(job)
                active.append(job)
            if self.dry_run:
                return
            time.sleep(self.poll_s)
            active = [j for j in active if self._in_queue(j.job_id)]

    def _submit(self, job: Job) -> None:
        self.submitted.append(job)
        script = self._write_job_script(job, index=len(self.submitted))
        if self.dry_run:
            job.job_id = "dry-run"
            return
        out = subprocess.run(["qsub", script], check=True, capture_output=True, text=True, cwd=self.root).stdout
        job.job_id = out.strip()

    def _write_job_script(self, job: Job, index: int) -> str:
        template = open(os.path.join(self.root, "scripts", "lab", "prepare", "job_template.sh")).read()
        name = "prep-" + job.runs[0].spec["kind"]
        stem = os.path.join(self.out_dir, "logs", "job_%s_%d" % (name, index))
        script = template.format(queue=self.queue, select=len(job.runs), walltime=_hms(job.walltime_s),
                                 project=self.project, name=name, root=self.root, log=stem + ".log",
                                 run_dirs=" ".join(r.dir for r in job.runs))
        os.makedirs(os.path.join(self.out_dir, "logs"), exist_ok=True)
        path = stem + ".sh"
        with open(path, "w") as f:
            f.write(script)
        return path

    @staticmethod
    def _in_queue(job_id: str) -> bool:
        out = subprocess.run(["qstat", "-a"], capture_output=True, text=True).stdout
        return any(line.startswith(job_id.split(".")[0]) for line in out.splitlines())

    # ---- collect -------------------------------------------------------------------------
    def _collect(self, run: Run):
        rows = self._history_rows(run.spec["history"])
        if isinstance(run.task, TuneTask):
            return self._tune_result(rows, run.spec["history"])
        return EvalResult(_tps_per_config(rows, len(run.task.configs)))

    @staticmethod
    def _history_rows(path: str) -> list:
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return json.load(f)["data"]

    @staticmethod
    def _tune_result(rows: list, history: str) -> TuneResult:
        ok = [r for r in rows if r["trial_state"] == SUCCESS and r.get("external_metrics")]
        if not ok:
            return TuneResult({}, 0.0, 0, len(rows), history)
        best = max(ok, key=lambda r: r["external_metrics"]["tps"])
        return TuneResult(dict(best["configuration"]), float(best["external_metrics"]["tps"]), len(ok), len(rows), history)


def _tps_per_config(rows: list, n_configs: int) -> list:
    """Row 0 is the default config the Sampler always plays first; rows 1.. follow the replay list."""
    tps = []
    for row in rows[1:1 + n_configs]:
        ok = row["trial_state"] == SUCCESS and row.get("external_metrics")
        tps.append(float(row["external_metrics"]["tps"]) if ok else None)
    return tps + [None] * (n_configs - len(tps))


def _hms(seconds: int) -> str:
    return "%02d:%02d:00" % (seconds // 3600, (seconds % 3600) // 60)
