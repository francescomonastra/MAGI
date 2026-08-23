# Versions Guideline

Compact development history of MAGI's model lineage: what each version changed,
what it achieved, what its drawback was, and where it landed. Model-variant details
for the current package are in `CLAUDE.md`; the evidence behind any v0.8+ entry
below lives in `docs/` (linked inline).

## v0.6 — `CVAE_CatEnergy_CatUV`

- **Change:** first package-native model. Categorical energy head (binned logE) +
  discretized $u_v$. Introduces task-adaptive loss weighting
  (`training/adaptive_callbacks.py`).
- **Achievement:** first reproducible, checkpointed training pipeline outside a
  single notebook.
- **Drawback:** both energy and $u_v$ are categorical — the energy head is a
  boxcar at bin width, far coarser than any real detector resolution. This
  limitation persists through every later "Cat" head, including v0.7.2.
- **v0.6.1:** validation pass specifically on cosmic-ray (CR) data.

## v0.7 — `CVAE_CatEnergy_ContGeom_TaskAdaptive`

- **Change:** replaces discretized $u_v$ (and $u_r$) with a quantile-transform
  continuous target — `filter_particle_types_continuous_geometry`,
  `geometry_transform="quantile_u_r_u_v"`.
- **Achievement:** geometry is no longer bin-limited; energy head still
  categorical.
- **v0.7.1:** quantile transform tested and trained on K-40 (Small) and CR real
  data, first reference plots for this line.

## v0.7.2 — `CVAE_CatEnergy_ContPhi_TaskAdaptive` (stable fallback, superseded as default by v0.8.2)

- **Change:** extends the continuous-quantile treatment to the angles
  ($\phi_r$, $\phi_v$ continuous, not just $u_r$/$u_v$) —
  `geometry_transform="quantile_u_r_u_v_phi_r_phi_v"`. Full geometry is now
  continuous; only energy remains categorical.
- **Achievement:** validated on CR, Geant4 interface script adapted to the new
  model. Reliable enough to have been **the beta default / fallback** before
  v0.8.2 — see [`docs/v0.8_v072_comparison.md`](docs/v0.8_v072_comparison.md).
- **Drawback (the reason v0.8 exists):** the categorical energy head is still a
  boxcar at bin width — ~20 keV wide at 511 keV, ~5000× the 4 eV X-IFU detector
  resolution. Cannot deliver detector-resolution spectral lines, which the
  science case needs.

## v0.8 — `CVAE_MixEnergy_ContPhi_TaskAdaptive`, gen 1 (gated mixture energy head)

- **Change:** the energy head becomes a gated mixture of a continuum density
  (conditional RQS normalizing flow, `core/flows.py`) and fixed-position Gaussian
  line components pinned at detector resolution (4 eV FWHM for X-IFU), plus a
  learnable conditional prior `p(z|cond)` (`core/priors.py`, MC-KL) so the encoder
  posterior and the sampling prior actually agree on real data.
- **Achievement (the headline, and it holds up):** every real cross-correlation
  over (logE, $u_r$, $u_v$, $\phi_r$, $\phi_v$) reproduced within ±0.04 (CR) /
  ±0.02 (Small) — the aggregated-posterior/prior mismatch that undermines a
  plain VAE prior is closed on real data. CR high-energy continuum + muon
  structure essentially perfect. Cycle 1 (CDF pre-warp of the flow's
  standardization) fixed the CR Compton edge and Small's low-E tail. Cycle 2
  (focal-weighted gate CE) got Small's three lines to ~1.0–1.3 recovery.
- **Drawback:** CR fluorescence lines (Al Kα1, "Ni Kβ") stayed badly
  under-recovered (0.13, 0.34) even after Cycle 2 — below the ≳0.8 bar needed
  to replace v0.7.2 as the beta default. See
  [`docs/v0.8_v072_comparison.md`](docs/v0.8_v072_comparison.md).
- **Result:** did not clear the go/no-go bar; v0.7.2 stayed the beta default.
  Later found (in v0.8.1) that most of this "line gap" was actually a broken
  measurement, not a broken model.

## v0.8.1 — line truth & measurement hardening (current, in progress)

- **Change:** audited the v0.8 line-recovery *measurement* itself rather than
  the model. Found three compounding defects: the recovery ratio wasn't
  normalized for generating 1M events against 3.44M/1.87M real ones; the
  matching window was ~190× the pinned 4 eV line width (pure continuum, not
  line); and line positions were taken from Geant4's `fluor_Bearden` table
  while the simulation actually emits **EADL** energies, pinning most lines
  4–11 detector-FWHM away from where the real events are. A greedy
  nearest-candidate matcher also mislabeled CR's real Cu Kα peak as "Ni Kβ".
  See [`docs/v0.8.1_line_truth.md`](docs/v0.8.1_line_truth.md).
- **Achievement:** corrected metric shows lines were mostly **over**-generated,
  not under — the opposite of v0.8's headline problem. Rebuilt the candidate
  table from EADL energies, fixed the gate-target bandwidth to track detector
  resolution instead of the detection-bin width (Phase 2), and modelled Cu Kα2
  (3,940 real CR events, previously invisible to the coarse detection grid) via
  a dedicated fine-resolution confirmation pass. Built a `--seeds` flag for
  `tools/acceptance_v0_8.py` that reports mean±std across seeds instead of a
  single run.
- **Drawback / open finding:** per-line `gate_class_weights` calibration was
  tried and is a **documented negative result** — the effect was smaller than
  run-to-run noise (§10). A continuum "Phase 3" polish attempt (extra CDF-warp
  knots + more RQS spline bins on CR) looked promising on one seed but, run
  properly across 3 seeds each config, showed **no effect distinguishable from
  noise** (§11–§11.3) — a finding about the per-band Wasserstein diagnostic's
  own variance as much as about the fix. Multi-seed evaluation is now treated
  as a prerequisite for trusting any single-run tuning result on this
  architecture, not just a nice-to-have.
- **Result:** cleared the go/no-go bar in v0.8.2 below; superseded as the
  in-progress line.

## v0.8.2 — prior zone-conditioning (current, `CVAE_MixEnergy_ContPhi_TaskAdaptive`)

- **Change:** widens the coupling prior's conditioning from `[type]` to
  `[type, zone]` — zone being the real `[continuum, line_1..line_L]` gate
  target at train time, sampled from each type's empirical zone frequency
  (`zone_probs`) at generation time. Everything else inherited from v0.8.1
  (EADL line positions, resolution-bandwidth gate targets, `gate_focal_gamma=1`)
  is unchanged.
- **Achievement:** the one confirmed improvement — on CR, 3 seeds, took Al Kα1
  recovery from 0.702 ± 0.290 (FAIL) to 0.959 ± 0.025 (PASS), with the coupling
  residual unaffected (0.0345 ± 0.0048 → 0.0285 ± 0.0101). This is the version
  shipped in this repo's `trained_models/` and used by `MAGI_v0_8_2.ipynb`.
- **Drawback / rejected idea:** `build_gate_targets(..., bandwidth_mode="exact")`
  — labelling gate targets by exact energy match instead of a Gaussian kernel of
  the detector resolution — measured on CR, did not fix the line it targeted,
  broke the confirmed Al Kα1 result, and pushed the coupling residual outside
  its bar. `"resolution"` remains the default.
- **Result:** current package default. Per-line intensities still do not pass
  reliably across all lines (0.7×–4.9×, unstable across seeds) — see
  `MAGI_package/docs/USAGE.md` § Accuracy you can rely on, or
  `docs/manual/magi_manual.pdf` for the full validity envelope.
