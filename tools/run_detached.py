#!/usr/bin/env python
"""Run a long job detached from the calling session, so that a time limit on the
session's background tasks does not kill it (a 40-epoch MAGI training or a prior
refit takes longer than 30 minutes on CPU).

    python tools/run_detached.py <log file> <command> [args ...]

The job gets its own process group (start_new_session), its output goes to the log
file, and the launcher returns at once with the PID. Pair it with a resumable job
(fit_model(checkpoint_dir=..., resume=True), tools/train_cr_v1.py --resume,
tools/refit_prior_posthoc.py) so that a killed or crashed job continues from its last
epoch when the same line is run again. Follow it with `tail -f <log>`.
"""
import os
import subprocess
import sys

if len(sys.argv) < 3:
    raise SystemExit(__doc__)
log, cmd = sys.argv[1], sys.argv[2:]
os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
with open(log, "ab") as f:
    p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, cwd=os.getcwd())
print(f"started pid {p.pid}, log {log}")
