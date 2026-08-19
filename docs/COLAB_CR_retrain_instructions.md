# Colab: CR retrain on the corrected training set — instructions

Notebook: `MAGI_v0_8_2_CR_Colab.ipynb` (repo root)
Written 2026-08-17. Target: a v0.8.2 CR checkpoint trained on `alloutputDSCryoSphereCR_ingoingfix.dat`.

The notebook was **generated mechanically from `MAGI_v0_8_2.ipynb`**, not written by hand,
so every preprocessing and model cell is byte-identical to the pipeline that produced the
earlier checkpoints. Four things differ, and nothing else:

1. GPU enabled. The local notebook forces CPU with `CUDA_VISIBLE_DEVICES=-1` because
   tensorflow-metal is ~8–10 % *slower* than the M1 CPU for this tfp op mix. That is an
   Apple-Metal result; it does not carry over to CUDA.
2. Paths point at Drive.
3. `SOURCE_FILES["CR"]` is the 17 Aug corrected file.
4. Generation / validation / Geant4-export cells are removed. This notebook trains and
   saves. Everything downstream runs locally.

---

## 1 · Getting the code and data onto Colab

The notebook has `PACKAGE_SOURCE = "github"` or `"drive"`. **GitHub is the default and the
better option** — not because it is faster, but because it pins provenance.

### Always via Drive, either way

| Path on Drive | Size |
|---|---|
| `MAGI/TrainingData/alloutputDSCryoSphereCR_ingoingfix.dat` | **339 MB** |
| `MAGI_v0_8_2_CR_Colab.ipynb` | 60 KB |

`TrainingData/` is gitignored, so **the big upload happens regardless** — a token does not
avoid it. Start it first.

### `PACKAGE_SOURCE = "github"` — recommended

Supplies `MAGI_package/` and `CandidateLines/` (both tracked), so those two drop off the
Drive list. The real gain is that the run is pinned to a **commit SHA**, recorded into the
checkpoint metadata as `magi_commit`, instead of "whatever copy happened to be on Drive".
Given how much of this project's lost time has been provenance failures, that is worth more
than the 1.5 MB saved.

**Token setup — do this yourself; the notebook never sees it typed.**

1. GitHub → Settings → Developer settings → **Fine-grained** personal access token.
2. Repository access: **only** `francescomonastra/MAGI`.
3. Permissions: **Contents → Read-only**. Nothing else.
4. Set a short expiry.
5. In Colab: the **key icon** in the left sidebar → new secret named `GITHUB_TOKEN` →
   paste the value → enable notebook access.

The notebook reads it with `userdata.get("GITHUB_TOKEN")` at runtime, scrubs it out of all
git and pip output (both echo the URL they were handed — that is how tokens end up in saved
notebooks), and `del`s it from the kernel once the clone is done. Do not paste it into a cell.

The clone uses `--depth 1 --filter=blob:none --sparse` and checks out only `MAGI_package`
and `CandidateLines`, so it does not drag down `trained_models/` (189 files) or `Plots/` (80).

**One caveat, checked on 2026-08-17:** local is **14 commits ahead** of
`origin/energy-mixture`, and one of them (`c058eef`) touches `MAGI_package`. I verified this
does not matter for a v0.8.2 run — `prior_zone_conditioning`, `zone_probs`,
`continuum_flow_warp` and `CVAE_MixEnergy_ContPhi_TaskAdaptive` all appear with identical
symbol counts in the pushed and local trees. The unpushed change is purely v0.8.3-additive:
an `energy_condition_geometry` parameter that defaults to `False` precisely so pre-v0.8.3
checkpoints rebuild unchanged, plus one new validation export. A checkpoint trained against
the pushed package therefore loads correctly in the newer local one.

If you would rather remove the doubt entirely, push `energy-mixture` first and the two trees
are identical.

### `PACKAGE_SOURCE = "drive"` — fallback

Needs `MAGI/MAGI_package/` (1.5 MB) and `MAGI/CandidateLines/…_EADL.json` (30 KB) on Drive.
No token. Records `magi_commit = "drive-copy-unpinned"` so the checkpoint is honest about it.

Note the candidate-lines file is the **EADL** one, not the plain `_fixed.json` beside it —
v0.8.1 replaced the Bearden table because it pinned every fluorescence line 4–11 detector
FWHM from its real peak.

Once the repo is public at release, the token stops being necessary at all.

## 2 · Runtime

Runtime → Change runtime type → **T4 GPU** is enough (the model is small: latent 8,
hidden [128,128,64]). Cell 1 runs `nvidia-smi` and will say so plainly if you got a CPU
runtime instead.

## 3 · Run order

Run top to bottom. The first four cells are, in order: GPU check → **mount Drive** →
install `magi` → import check. That order matters and the first version of this notebook
got it wrong (install before mount, so the package path did not exist yet and the failure
was hidden by a quiet pip flag — the symptom was `ModuleNotFoundError: No module named
'magi'` several cells later).

The install cell copies `MAGI_package` from Drive to local disk before installing, because
`pip install -e` against a Drive FUSE path is unreliable. It prints pip's real output, and
falls back to `sys.path` if pip fails at all — `magi` is pure Python, so that works.

Four checkpoints where you should stop and look:

**After the mount cell** — every path must print `OK` (it asserts otherwise), and the
provenance guard must say `columns = 13`. A 9 means an old file got uploaded.

**After the import check** — `magi.__file__`, a version, and a non-empty GPU list. If this
cell fails, stop; nothing below it can work.

**After `filter_particle_types_continuous_geometry`** — `n_types` must be **4**
(gamma, mu−, e−, e+). The old checkpoint carried **6**, including `anti_nu_e` and `nu_mu`
at ~2.3 % each; the corrected file has no neutrinos. If it says 6, the wrong file loaded.

**After training** — 40 epochs. On the M1 CPU this was ~48 min; expect meaningfully less on
a T4, though not 10×: the RQS flow and the tfp mixture ops do not all vectorise well on GPU.
`MAGI_Colab_GPU_Benchmark.ipynb` in the repo has prior numbers for this exact question.

## 4 · The one failure mode worth guarding against

**The weights and the fitted quantile transformers are one artifact.** Generation reloads
`*_quantile_transformers.joblib` alongside `*.weights.h5` and the metadata JSON, and
`geometry_transform` (`"quantile_u_r_u_v_phi_r_phi_v"`) must match on both sides. If the
transformers are refit anywhere — locally, or in a re-run cell — the reconstructed physics
comes out **wrong with no exception raised**. Every distribution will look plausible and
every number will be false.

The save cell writes them together. Keep them together, and copy the whole `save_dir` as a
unit — never individual files.

## 5 · Bring back

`MyDrive/MAGI/trained_models/v0_8_2_CR_ingoingfix/`, which will contain:

```
mix_CR.weights.h5
mix_CR_config.json                 <- self-describing, authoritative
mix_CR_history.json
mix_CR_metadata.json
mix_CR_task_weights.json
mix_CR_summary.txt
mix_CR_quantile_transformers.joblib   <- must travel with the weights
mix_CR_colab_fingerprint.json         <- for the local check below
```

The last cell also zips it to `/content/v0_8_2_CR_ingoingfix.zip` if you prefer a download
over a Drive sync.

## 6 · Local verification, before anything downstream

Five minutes, and it catches every version of the drift failure. Copy the directory into
`/Volumes/X10Pro/MAGI/trained_models/`, then reload it locally with
`magi.load_task_adaptive_model_for_generation`, generate a small sample, and check against
`mix_CR_colab_fingerprint.json`:

- `n_types` = 4 and the same `idx_to_type` mapping
- `type_probs` matching the fingerprint
- species fractions of the generated sample matching the training set to a few 0.1 %
- `geometry_transform` string identical in the metadata

Only after that does the detector-level retest mean anything.

## 7 · What the retrain is expected to change

Prediction on record, from the corrected training set, stated **before** the retrain:

| quantity | factor vs old |
|---|---|
| aimed flux, b < 20 mm | **×1.268** |
| aimed muon flux, b < 20 mm | **×1.092** |
| derived: 1–7 keV MIP band ratio | 0.750 → **≈0.82** |

The MIP band is 84 % muon events, so it should move with the muon factor, and the soft band
with the total. **The SRON deficit should narrow from ~25 % to ~18 % — not close.** If it
closes completely, something else changed and the result needs explaining before it is used.
