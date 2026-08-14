#!/bin/bash
# Run the KDSource-vs-MAGI benchmark once the speed-up measurement has finished.
# It must not overlap: a KDE fit on millions of points would compete for CPU
# with the speed-up timing, which is only valid on an idle machine.
set -u
SPEEDUP=/Volumes/X10Pro/Geant4-projects/Geant4-10.4.3/S1GDMLSRON_CryoSphereEmission/speedup_run
PY=/Users/francesco/mambaforge/envs/tf-metal/bin/python

echo "=== $(date) waiting for the speed-up run ==="
while ! grep -q "SPEEDUP DONE" "$SPEEDUP/speedup.log" 2>/dev/null; do
  if ! pgrep -f "run_speedup.sh" > /dev/null && \
     ! grep -q "waiting for the CR seed run" "$SPEEDUP/speedup.log" 2>/dev/null; then
    echo "*** speed-up runner is gone and never completed - running anyway ***"
    break
  fi
  sleep 300
done
while pgrep -x athena > /dev/null; do sleep 120; done
echo "=== $(date) machine free, starting benchmark ==="
"$PY" /Volumes/X10Pro/MAGI/tools/kdsource_vs_magi.py 2>&1 | tee /Volumes/X10Pro/MAGI/logs/kdsource_vs_magi.txt
echo "=== $(date) BENCHMARK DONE ==="
