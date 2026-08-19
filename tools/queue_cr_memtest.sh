#!/bin/bash
# Memorisation test on the CR checkpoint, queued behind the detector-level retest.
# Both need the machine, so this waits rather than competing: the retest is the
# time-critical one (it scores a prediction made before the retrain).
#
# Uses the SAME harness validated on DM1.2 on 17/08, whose controls read
#   held-out real -> ratio 0.999 (KS p=0.81)   and   literal copies -> ratio 0.000,
# so the metric is calibrated in both directions before it is trusted on CR.
set -u
MAGI=/Volumes/X10Pro/MAGI
PY=/Users/francesco/mambaforge/envs/tf-metal/bin/python
CKPT=$MAGI/trained_models/v0_8_2_CR_ingoingfix
GEN=/Volumes/X10Pro/tmp/magi_cr_memtest.txt

echo "=== $(date) waiting for the CR retest to finish ==="
while pgrep -f run_cr_retest.sh > /dev/null || pgrep -x athena > /dev/null; do sleep 120; done
sleep 60

echo "=== $(date) generating the CR sample for the test ==="
env -u TF_USE_LEGACY_KERAS CUDA_VISIBLE_DEVICES=-1 "$PY" \
    "$MAGI/scripts/generate_geant_source.py" \
    --save-dir "$CKPT" --model-name mix_CR \
    --metadata-file "$CKPT/mix_CR_metadata.json" \
    --transformers-file "$CKPT/mix_CR_quantile_transformers.joblib" \
    --output-file "$GEN" --n-events 400000 --seed 20260818 \
    --format text > "$MAGI/tools/cr_memtest_gen.log" 2>&1

# Same gate as the retest: the sphere must be the corrected one, not a fallback.
if grep -q "WARNING" "$MAGI/tools/cr_memtest_gen.log"; then
  echo "*** ABORT: generation fell back to a default sphere" >&2
  grep "WARNING" "$MAGI/tools/cr_memtest_gen.log" >&2; exit 1
fi
grep -h "CryoSphere for reconstruction" "$MAGI/tools/cr_memtest_gen.log"
echo "generated: $(wc -l < "$GEN") records"

echo "=== $(date) memorisation test, CR ==="
cd "$MAGI" && "$PY" tools/memorisation_test.py cr 2>&1 | tee docs/_data/memorisation_cr.txt
echo "=== $(date) CR MEMORISATION TEST DONE ==="
