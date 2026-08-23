# Versions Guideline

Compact development history of MAGI's model lineage: what each version changed,
what it achieved, what its drawback was, and where it landed. Model-variant details
for the current package are in `CLAUDE.md`; the evidence behind any v0.8+ entry
below lives in `docs/` (linked inline).

## Pre-package line (GEEANNT, notebook-only, `OldNotebooks/`)

Before the codebase became an installable package. Working name GEEANNT; single
detector-crossing events (energy + geometry), no task-adaptive training yet.

| Version | Change | Result |
|---|---|---|
| 1.3.sv | Direction $u_v=\cos\theta_v$ modeled via $s_v=\mathrm{atanh}(u_v)$, plain Gaussian head | shape too crude, superseded |
| 1.3.tv | Added a smoothing transform $t_v=\mathrm{sign}(s_v)\log(1+\lvert s_v\rvert)$ on top of $s_v$ | smoother $s_v$, still single-mode |
| 1.4 | $t_v$ modeled with a mixture of Gaussians ($K{=}3$–$5$) instead of one | first multi-modal fit for $\theta_v$ |
| 1.4RS / 1.5RS | 3-phase Optuna random search (30 trials → 5 best → Wasserstein-scored refinement); categorical discretized $u_v$ head, split decoder branches, dedicated $\phi_v$ angular loss | best pre-package result — categorical $u_v$ + split decoder + angular-loss weighting |

Refactored into the first installable package structure immediately after 1.5RS
(`5ab0920`); later renamed GEEANNT → **magi** (repo/paper name **MAGI**, `17b4bec`).

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

## v0.7.2 — `CVAE_CatEnergy_ContPhi_TaskAdaptive` (current beta fallback)

- **Change:** extends the continuous-quantile treatment to the angles
  ($\phi_r$, $\phi_v$ continuous, not just $u_r$/$u_v$) —
  `geometry_transform="quantile_u_r_u_v_phi_r_phi_v"`. Full geometry is now
  continuous; only energy remains categorical.
- **Achievement:** validated on CR, Geant4 interface script adapted to the new
  model. Reliable enough to be **the current beta default / fallback** —
  see [`docs/v0.8_v072_comparison.md`](docs/v0.8_v072_comparison.md).
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
- **Result:** still open. v0.7.2 remains the beta default; go/no-go for v0.8.1
  is pending Phase 4 (release hardening) and a clean multi-seed read on both
  CR and Small. See [`docs/v0.8.2_plan.md`](docs/v0.8.2_plan.md) for the
  current backlog and sequencing.
