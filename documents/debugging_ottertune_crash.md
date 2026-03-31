# OtterTune Crash Investigation

## Problem

Running OtterTune (`config_sysbench_ot_rw50.ini`) sometimes gets stuck indefinitely during MySQL restart, causing the experiment to hang. This typically happens at iteration 2 when the optimizer suggests a config with large `innodb_buffer_pool_size` (e.g. 25GB) and/or large `innodb_log_file_size`.

## Symptoms

- `_kill_mysqld()` logs `Force close!` after `TIMEOUT_CLOSE` (300s) expires
- `_start_mysqld()` enters the connection retry loop and never connects
- Logs show repeated `Can't connect to local MySQL server through socket` for 600+ seconds
- The Python process itself does not crash — it hangs in the retry loop at `mysqldb.py:228`

## What We Know

1. **Config changes are fast in isolation**: A standalone test (`scripts/test_restart_slowdown.sh`) proved that changing `innodb_log_file_size` from 5GB to 48MB with a force kill + restart takes only ~33 seconds (30s shutdown + 3s startup).

2. **History loading is slow**: Loading 254 source history JSON files into `HistoryContainer` objects takes ~380 seconds. This was fixed with pickle caching (`tuner.py:load_history`), reducing subsequent loads to ~7 seconds. The cache is stored as `{data_repo}.pkl`.

3. **The hang happens during `_start_mysqld`**: After force kill, `subprocess.Popen` starts a new mysqld, but the code never checks if the mysqld process is still alive during the connection loop. If mysqld crashes on startup, the loop waits blindly for 600 seconds.

4. **matplotlib exit code 1 (fixed)**: `tuner.py` previously crashed with exit code 1 when saving the convergence plot on a headless server. Fixed by setting `matplotlib.use('Agg')` before importing pyplot.

## What We Don't Know

- **Why mysqld sometimes fails to start** after a force kill in the context of a full experiment, when the standalone test shows it should only take ~33 seconds.
- Whether the issue is corrupted redo logs, stale lock files, port conflicts, memory pressure from other processes (Elasticsearch uses 128GB on this server), or something else entirely.

## How to Reproduce

### Quick reproduction
```bash
cd scripts
export SYSBENCH_BIN=/usr/local/bin/sysbench
export MYSQL_SOCK=../mysql_build/mysql.sock
source ../venv/bin/activate

# Run OtterTune — crash typically happens at iteration 2
python optimize.py --config=config_sysbench_ot_rw50.ini
```

### More reliable reproduction (DML before OtterTune)
The issue is more likely to occur when running OtterTune after DML, because DML's last iteration leaves MySQL running with large knobs (17GB buffer pool, 5GB redo logs):
```bash
cd scripts
export SYSBENCH_BIN=/usr/local/bin/sysbench
export MYSQL_SOCK=../mysql_build/mysql.sock

# Run DML first (leaves MySQL with large config)
python optimize.py --config=config_sysbench_dml_rw50.ini

# Then run OtterTune (iteration 1 must restart MySQL from DML's large config)
python optimize.py --config=config_sysbench_ot_rw50.ini
```

### VSCode Debugger
A debug configuration is available in `.vscode/launch.json`:
- **"OtterTune sysbench rw50 (crash repro)"**

Set breakpoints at:
- `autotune/database/mysqldb.py:183` — `Force close!` (TIMEOUT_CLOSE exceeded)
- `autotune/database/mysqldb.py:214` — `subprocess.Popen` (mysqld launch)
- `autotune/database/mysqldb.py:228` — connection retry loop entry
- `autotune/database/mysqldb.py:243` — `count > 600` (connection timeout)
- `autotune/dbenv.py:344` — `if not flag:` (apply_knobs_offline returned False)
- `autotune/dbenv.py:350` — `raise Exception('Apply knobs failed!')`

When it hangs at the connection loop, check:
```bash
# Is mysqld actually running?
ps aux | grep mysqld | grep -v grep | grep -v exporter

# Check MySQL error log for startup failures
tail -50 ../mysql_build/data/*.err

# Is the socket file created?
ls -la ../mysql_build/mysql.sock

# Is something else using port 3306?
ss -tlnp | grep 3306
```

## Code Path

```
optimize.py
  -> DBTuner.tune()
    -> PipleLine.run()
      -> PipleLine.iterate()
        -> PipleLine.evaluate(config)
          -> DBEnv.step(config)              # dbenv.py:396
            -> DBEnv.step_GP(knobs)          # dbenv.py:314
              -> MysqlDB.apply_knobs_offline(knobs)  # mysqldb.py:276
                -> MysqlDB._kill_mysqld()    # mysqldb.py:153  <-- TIMEOUT_CLOSE=300
                -> MysqlDB._gen_config_file(knobs)   # mysqldb.py:105
                -> MysqlDB._start_mysqld()   # mysqldb.py:189  <-- hangs here
```

## Timeline of a Typical Failing Run

```
10:50:39  [step 2] generate knobs: {innodb_buffer_pool_size: 25GB, innodb_log_file_size: 600MB, ...}
10:50:39  _kill_mysqld: mysqladmin shutdown (waits up to TIMEOUT_CLOSE=300s)
10:51:39  Force close! (timeout hit)
10:51:39  force_kill_cmd: kill -9 (via ps aux | grep | xargs)
10:51:39  mysql is shut down
10:51:39  _gen_config_file: restore clean cnf, write new knobs
10:51:39  _start_mysqld: subprocess.Popen(mysqld)
10:51:39  wait for connection...
10:51:39  Can't connect... (repeats every 1 second)
  ...     (mysqld either doing crash recovery or failed to start — unknown)
11:01:39  count > 600 -> start_sucess = False
11:01:39  apply_knobs_offline returns False
11:01:39  step_GP: raise Exception('Apply knobs failed!')
11:01:39  step: catches exception, returns FAILED
11:01:39  evaluate: sets objs = FAILED_PERF, continues
```

## Related Files

| File | Relevance |
|------|-----------|
| `autotune/database/mysqldb.py` | `_kill_mysqld`, `_start_mysqld`, `apply_knobs_offline` |
| `autotune/dbenv.py` | `step()`, `step_GP()` — where exceptions are raised/caught |
| `autotune/pipleline/pipleline.py` | `run()`, `iterate()`, `evaluate()` — main loop |
| `autotune/tuner.py` | `tune()` — entry point, history loading, pickle cache |
| `scripts/config_sysbench_ot_rw50.ini` | OtterTune config for sysbench read-write |
| `scripts/test_restart_slowdown.sh` | Standalone test proving config change takes ~33s |
| `.vscode/launch.json` | Debug configuration for reproducing the crash |
