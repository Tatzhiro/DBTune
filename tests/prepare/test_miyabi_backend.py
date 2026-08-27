"""Dry-run tests of the Miyabi backend against a real DBTune tree (skipped if absent)."""
import json
import os
import tempfile
import unittest

from autotune.prepare.backends.miyabi import MiyabiExecutor, _tps_per_config
from autotune.prepare.executor import ResultCache
from autotune.prepare.tasks import EvalTask, TuneBudget, TuneTask, interleave
from autotune.prepare.workload import Workload

ROOT = os.environ.get("DBTUNE_ROOT", "/work/xg26g002/x10563/DBTune")
HAVE_TREE = os.path.exists(os.path.join(ROOT, "mysql_build", "cnf", "my.cnf.clean"))
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@unittest.skipUnless(HAVE_TREE, "DBTune tree with mysql_build not available")
class MiyabiDryRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ex = MiyabiExecutor(root=ROOT, out_dir=self.tmp.name, dry_run=True, max_nodes=3,
                                 cache=ResultCache(None))
        # the backend reads its base ini / templates from the tree it runs against; point them here
        self._link("scripts/lab/prepare")
        self._link("scripts/experiment/gen_knobs/mysql_perf_8.0_online.json")

    def _link(self, rel):
        src, dst = os.path.join(HERE, rel), os.path.join(ROOT, rel)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
            self.addCleanup(os.unlink, dst)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tune_task_materializes_llamatune_ini(self):
        w = Workload.make("sysbench", rows=400000, threads=64, zipf=0.7)
        run = self.ex._materialize(TuneTask(w, TuneBudget(min_success=50, max_attempts=120)))
        ini = open(os.path.join(run.dir, "config.ini")).read()
        self.assertIn("optimize_method = LlamaTune", ini)
        self.assertIn("min_success = 50", ini)
        self.assertIn("sysbench_table_size = 400000", ini)
        self.assertIn("thread_num = 64", ini)
        self.assertIn("sysbench_zipfian_exp = 0.7", ini)
        self.assertIn("online_mode = True", ini)
        self.assertEqual(run.spec["snapshot"], "data_150x400k")

    def test_uniform_workload_uses_rand_type(self):
        w = Workload.make("sysbench", rows=800000, threads=128, zipf=0.0)
        run = self.ex._materialize(TuneTask(w))
        ini = open(os.path.join(run.dir, "config.ini")).read()
        self.assertIn("sysbench_rand_type = uniform", ini)
        self.assertNotIn("sysbench_zipfian_exp", ini)

    def test_eval_task_writes_replay_and_job_script(self):
        w = Workload.make("sysbench", rows=800000, threads=128, zipf=0.7)
        cfg = {"innodb_buffer_pool_size": 21474836480, "innodb_flush_log_at_trx_commit": "2"}
        runs = [self.ex._materialize(EvalTask(w, interleave([cfg], 3))),
                self.ex._materialize(TuneTask(w)), self.ex._materialize(TuneTask(w.with_param("threads", 32))),
                self.ex._materialize(TuneTask(w.with_param("threads", 256)))]
        replay = json.load(open(os.path.join(runs[0].dir, "replay.json")))
        self.assertEqual(len(replay["configs"]), 3)
        jobs = self.ex._pack(runs)
        self.assertEqual([len(j.runs) for j in jobs], [3, 1])          # max_nodes = 3
        self.ex._execute(jobs)                                          # dry run: scripts only
        script = open(os.path.join(self.tmp.name, "logs", "job_prep-eval_1.sh")).read()
        self.assertIn("#PBS -l select=3", script)
        self.assertIn("pbsdsh -n", script)
        self.assertIn(runs[0].dir, script)

    def test_rerun_attaches_to_a_job_still_in_the_queue(self):
        """A restarted program must not qsub a duplicate for run dirs a live job already covers."""
        w = Workload.make("sysbench", rows=800000, threads=128, zipf=0.7)
        first = MiyabiExecutor(root=ROOT, out_dir=self.tmp.name, dry_run=False, cache=ResultCache(None))
        first._qsub = lambda script: "424242.opbs"
        first._queued_job_ids = lambda: set()
        job = first._pack([first._materialize(TuneTask(w))])[0]
        first._submit(job)
        self.assertEqual(job.job_id, "424242.opbs")

        second = MiyabiExecutor(root=ROOT, out_dir=self.tmp.name, dry_run=False, cache=ResultCache(None))
        second._qsub = lambda script: self.fail("must not resubmit while the job is queued")
        second._queued_job_ids = lambda: {"424242.opbs"}
        job2 = second._pack([second._materialize(TuneTask(w))])[0]
        second._submit(job2)
        self.assertEqual(job2.job_id, "424242.opbs")           # attached, not duplicated

        third = MiyabiExecutor(root=ROOT, out_dir=self.tmp.name, dry_run=False, cache=ResultCache(None))
        third._qsub = lambda script: "424300.opbs"
        third._queued_job_ids = lambda: set()                   # old job finished -> resubmit is fine
        job3 = third._pack([third._materialize(TuneTask(w))])[0]
        third._submit(job3)
        self.assertEqual(job3.job_id, "424300.opbs")

    def test_tpcc_is_refused_until_runner_exists(self):
        with self.assertRaises(NotImplementedError):
            self.ex._materialize(TuneTask(Workload.make("tpcc", warehouses=100, threads=32, write_ratio=0.88, skew=0.7)))

    def test_history_parsing_on_real_files(self):
        hist = os.path.join(ROOT, "scripts", "DBTune_history", "history_prep_anchor_S3.json")
        if not os.path.exists(hist):
            self.skipTest("no S3 probe history")
        rows = json.load(open(hist))["data"]
        tps = _tps_per_config(rows, 14)
        self.assertEqual(len(tps), 14)
        self.assertGreater(tps[0], 20000)                               # first replayed config on 80k rows
        result = MiyabiExecutor._tune_result(rows, hist)
        self.assertEqual(result.n_attempts, len(rows))
        self.assertIn("innodb_buffer_pool_size", result.best_config)


if __name__ == "__main__":
    unittest.main()
