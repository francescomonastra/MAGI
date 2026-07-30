"""Transform MAGI_v0_8_1.ipynb -> MAGI_v0_8_2.ipynb, same structure.

v0.8.2 changes over v0.8.1 (architecture, gate-target bandwidth, and cell
order are otherwise identical):
  - prior_zone_conditioning=True: widen the coupling prior's conditioning
    from [type] to [type, zone]. CONFIRMED on CR, 3 seeds: Al Ka1
    0.702+/-0.290 -> 0.959+/-0.025, coupling unaffected. New cell computes
    zone_probs exactly as tools/run_v0_8_real.py does.
  - bandwidth_mode="exact" was tried as a v0.8.2 follow-up and REJECTED:
    measured worse on every axis, including breaking the coupling bar.
    "resolution" remains the default; noted where it's set.
  - default SOURCE -> "Small", since the immediate next step is training
    Small and comparing it against a Geant4 simulation.
  - save paths v0_8_1_* -> v0_8_2_*.

Run from the repo root: python make_v0_8_2_nb.py
"""
import copy
import json
import pathlib

nb = json.load(open("MAGI_v0_8_1.ipynb"))
cells = nb["cells"]


def src(i):
    return "".join(cells[i]["source"])


def set_src(i, text):
    cells[i]["source"] = text.split("\n")
    cells[i]["source"] = [l + "\n" for l in cells[i]["source"][:-1]] + [cells[i]["source"][-1]]
    cells[i]["outputs"] = []
    if cells[i]["cell_type"] == "code":
        cells[i]["execution_count"] = None


def replace(i, old, new, count=1):
    s = src(i)
    assert old in s, f"cell {i}: pattern not found:\n{old[:200]}"
    s = s.replace(old, new, count)
    set_src(i, s)


# ---------------------------------------------------------------------
# Cell 0 - intro
# ---------------------------------------------------------------------
set_src(0, r"""# MAGI version 0.8.2 — prior zone-conditioning, and a rejected line-label idea

v0.8.2 keeps v0.8.1's fixes (corrected recovery metric, EADL line positions,
resolution-bandwidth gate targets) and the architecture unchanged —
`CVAE_MixEnergy_ContPhi_TaskAdaptive`. What changes is the coupling prior's
conditioning, plus one negative result worth knowing before you touch the
gate-target bandwidth again.

**Confirmed, 3 seeds (42/7/13) on CR** (`docs/v0.8.1_line_truth.md` §14):

| | v0.8.1 | v0.8.2 |
|---|---|---|
| Prior conditioning | `[type]` | `[type, zone]` — `prior_zone_conditioning=True` |
| CR Al Kα1 recovery | 0.702 ± 0.290 (FAIL) | **0.959 ± 0.025 (PASS)** |
| CR Cu Kβ (control) | 1.166 ± 0.224 | 1.034 ± 0.050 |
| Coupling max\|Δcorr\| | 0.0345 ± 0.0048 | 0.0285 ± 0.0101 (unaffected) |

**Why it works.** The gate is trained on `z ~ q(z|x)` but generates from
`z ~ p(z|cond)`. When `cond` is only the particle-type one-hot, the prior has
no way to express which mixture component (continuum vs. a specific line) an
event belongs to, so at generation time rare lines get routed by whatever the
type marginal happens to imply. Widening `cond` to `[type, zone]` — zone being
the real `[continuum, line_1..line_L]` gate target at train time, sampled from
each type's empirical zone frequency (`zone_probs`) at generation time — gives
the prior what it needs to place `z` correctly for a line-routed event.

**Rejected, measured on CR** (`docs/v0.8.1_line_truth.md` §14.2):
`build_gate_targets(..., bandwidth_mode="exact")` — labelling gate targets by
exact energy match instead of a Gaussian kernel of the detector resolution,
since raw Geant4 data has no detector response and fluorescence lines are
exactly monoenergetic. The physics premise is correct and confirmed in the
data. It did not help: Cu Kα1 routing was unchanged (2.011 vs. the confirmed
2.052), the previously-passing Al Kα1 regressed to 0.578, and the coupling
residual moved from 0.0285 to 0.054 — outside its bar. `"resolution"` remains
the default; do not switch without re-reading that section.

**What is different from `MAGI_v0_8_1.ipynb`**

- a new **zone probabilities** cell, right after the dataset build, computing
  each type's empirical `[continuum, line_1..line_L]` frequency from the real
  gate-target columns (needed because there is no real event to sample the
  zone from at generation time);
- the model built and configured with `prior_zone_conditioning=True` and
  `zone_probs`;
- checkpoints saved under `trained_models/v0_8_2_<source>/`, `config_version: 2`.
  A `config_version: 1` checkpoint (e.g. any v0.8.1 run) still loads unchanged:
  `prior_zone_conditioning` defaults to `False`, which reconstructs exactly
  the pre-existing architecture.

**Notebook layout** (same order as `MAGI_v0_8_1.ipynb` / `MAGI_v0_7_2.ipynb`)

1. Dataset Import and Checks — load, physical features, diagnostics, energy line detection.
2. Line-position audit — where the lines really are.
3. Standardization and Preprocessing — feature dataframe, gate targets, **zone probabilities**, split/conditioning, CDF warp knots.
4. Training with MAGI Package.
5. Generation and Validation — acceptance report, spectra, residuals, pairgrid, covariance/correlation.
6. Saving Model.
7. Generating Input Files for Geant4.
8. Appendix — flow vs Gaussian continuum (synthetic).

**One source per run.** Set `SOURCE = "CR"`, `"Small"`, or `"Torio"` and
re-run; the unattended both-sources equivalent is
`tools/run_v0_8_real.py --prior-zone-conditioning`, scored by
`tools/acceptance_v0_8.py --seeds 42 7 13` and `tools/plot_v0_8_real_corr.py`.

> **Validity envelope (v0.8.2 beta).** Coupling and the energy marginal pass
> with error bars on both reference sources; per-line intensities do not
> (0.7×–4.9×, unstable across seeds). Full table:
> `MAGI_package/docs/USAGE.md` § Accuracy you can rely on, or
> `docs/manual/magi_manual.pdf` §7. **No v0.8.2 checkpoint existed for Small
> before this notebook** — Phase C's confirmation was CR-only; running this
> notebook on Small is the first v0.8.2 measurement of it.""")

# ---------------------------------------------------------------------
# Cell 4 - source selection: default to Small
# ---------------------------------------------------------------------
replace(4, 'SOURCE = "Torio"          # "CR", "Small", or "Torio"',
        'SOURCE = "Small"          # "CR", "Small", or "Torio"')

# ---------------------------------------------------------------------
# Cell 6 markdown - v0.8.1 corrections list -> keep as historical baseline,
# just note nothing new here (leave as-is; still accurate for v0.8.2).
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Cell 15 - gate targets: note the exact-mode result
# ---------------------------------------------------------------------
replace(15,
    '''gate_targets = magi.build_gate_targets(
    E_full,
    feature_pack["energy_bins"],
    matched,
    bandwidth_mode="resolution",          # v0.8.1
    bandwidth_fwhm_mev=FWHM_MEV,
)''',
    '''# v0.8.2 follow-up: bandwidth_mode="exact" (exact energy match, since raw
# Geant4 data has no detector response and lines are truly monoenergetic) was
# tried and REJECTED - measured on CR it left Cu Kalpha1 unchanged and broke
# both the Al Kalpha1 fix below and the coupling bar (docs/v0.8.1_line_truth.md
# section 14.2). "resolution" remains the default.
gate_targets = magi.build_gate_targets(
    E_full,
    feature_pack["energy_bins"],
    matched,
    bandwidth_mode="resolution",          # v0.8.1, confirmed again in v0.8.2
    bandwidth_fwhm_mev=FWHM_MEV,
)''')

# ---------------------------------------------------------------------
# New cell after cell 16 (dataset build) - zone probabilities.
# Insert AFTER dataset_pack/report_continuous_geometry_features but the
# original cell 16 also does split/scale/conditioning/tf-datasets in one go.
# zone_probs needs dataset_pack (built) before split, matching
# tools/run_v0_8_real.py's ordering (computed right after
# filter_particle_types_continuous_geometry, before split_feature_data).
# Simplest correct transform: split cell 16 into
#   16a. dataset_pack + report (unchanged)
#   16b. NEW: zone_probs
#   16c. split/scale/conditioning/datasets (unchanged, renumbered)
# ---------------------------------------------------------------------
cell16_src = src(16)
marker = "split_pack = magi.split_feature_data("
assert marker in cell16_src
head, tail = cell16_src.split(marker, 1)
tail = marker + tail

zone_cell_src = r"""# ==========================================================
# v0.8.2 Phase C candidate 1: per-type zone probabilities
#
# The learned prior p(z|cond) is trained on z ~ q(z|x), where the real event's
# gate-target columns are visible, but generation draws z ~ p(z|cond) with NO
# real event to read the zone from. prior_zone_conditioning widens cond from
# [type] to [type, zone] at TRAIN time (the real gate target); at GENERATION
# time the zone is instead sampled from each type's empirical zone frequency
# computed here. Confirmed on CR, 3 seeds: CR Al Kalpha1 recovery
# 0.702+/-0.290 (FAIL) -> 0.959+/-0.025 (PASS), coupling unaffected
# (docs/v0.8.1_line_truth.md section 14). Computed unconditionally - it is
# cheap, and this is the first v0.8.2 measurement of it on a non-CR source.
# ==========================================================

n_zones = gate_targets.shape[1]
zone_cols = dataset_pack["X_cont_raw"][:, -n_zones:]
y_type_for_zones = dataset_pack["y_type"]

zone_probs = np.zeros((dataset_pack["n_types"], n_zones), dtype=np.float64)
for t in range(dataset_pack["n_types"]):
    mask = (y_type_for_zones == t)
    row = zone_cols[mask].mean(axis=0) if mask.any() else np.zeros(n_zones)
    row_sum = row.sum()
    zone_probs[t] = (row / row_sum) if row_sum > 0 else np.eye(n_zones)[0]

zone_labels = ["continuum"] + [m["label"] for m in matched]
print(f"{SOURCE}: zone_probs (per type, {zone_labels}):")
for t, tname in dataset_pack["idx_to_type"].items():
    print(f"    {tname:12s} {np.round(zone_probs[t], 4)}")
"""

set_src(16, head)
zone_cell = copy.deepcopy(cells[16])
set_src(16, head)
zone_cell["source"] = zone_cell_src.split("\n")
zone_cell["source"] = [l + "\n" for l in zone_cell["source"][:-1]] + [zone_cell["source"][-1]]
zone_cell["outputs"] = []
zone_cell["execution_count"] = None
zone_cell["metadata"] = {}

tail_cell = copy.deepcopy(cells[16])
tail_cell["source"] = tail.split("\n")
tail_cell["source"] = [l + "\n" for l in tail_cell["source"][:-1]] + [tail_cell["source"][-1]]
tail_cell["outputs"] = []
tail_cell["execution_count"] = None
tail_cell["metadata"] = {}

cells[16:17] = [cells[16], zone_cell, tail_cell]

# All indices from 17 on have shifted by +2. Re-derive src()/replace() against
# the mutated `cells` list directly from here.
def find(pattern, start=0):
    for i in range(start, len(cells)):
        if pattern in "".join(cells[i]["source"]):
            return i
    raise AssertionError(f"pattern not found: {pattern[:80]}")


# ---------------------------------------------------------------------
# Model description markdown
# ---------------------------------------------------------------------
i = find("### Model v0.8.1")
replace(i,
    "### Model v0.8.1 `CVAE_MixEnergy_ContPhi_TaskAdaptive`",
    "### Model v0.8.2 `CVAE_MixEnergy_ContPhi_TaskAdaptive`")
replace(i,
    '''Architecture identical to v0.8 — flow continuum (24 bins, 3 transforms) with the CDF
pre-warp, fixed lines pinned to 4 eV FWHM, `energy_flow_condition="z_cond"` with the
learnable coupling prior `p(z|cond)`. Two configuration changes:''',
    '''Architecture identical to v0.8 — flow continuum (24 bins, 3 transforms) with the CDF
pre-warp, fixed lines pinned to 4 eV FWHM, `energy_flow_condition="z_cond"` with the
learnable coupling prior `p(z|cond)`. Three configuration changes over v0.8:''')
replace(i,
    '''`gate_class_weights` (per-line manual weights) remains available and
checkpoint-safe as the next lever, and is left `None` here. If it is needed, the
update must be **two-sided** (`clip(w·(target/recovery)^α, 1/cap, cap)`) — the
failure mode is over-generation, which a floor of 1.0 cannot correct.''',
    '''`gate_class_weights` (per-line manual weights) remains available and
checkpoint-safe as the next lever, and is left `None` here. If it is needed, the
update must be **two-sided** (`clip(w·(target/recovery)^α, 1/cap, cap)`) — the
failure mode is over-generation, which a floor of 1.0 cannot correct.

- **`prior_zone_conditioning=True`, `zone_probs`** (v0.8.2). Widens the coupling
  prior's conditioning from `[type]` to `[type, zone]`, using the `zone_probs`
  computed above. Confirmed on CR (3 seeds): CR Al Kα1 recovery
  0.702 ± 0.290 (FAIL) → **0.959 ± 0.025 (PASS)**, coupling unaffected. This is
  the run's first measurement of it on a source other than CR.''')

# ---------------------------------------------------------------------
# Config cell: add prior_zone_conditioning + zone_probs to model_config
# ---------------------------------------------------------------------
i = find('"gate_focal_gamma": 1.0,          # v0.8.1: was 2.0')
replace(i,
    '''    "gate_focal_gamma": 1.0,          # v0.8.1: was 2.0
    "gate_class_weights": None,
    "line_logsigma_trainable": False,
    "x_ifu_resolution_ev": X_IFU_RESOLUTION_EV,
}''',
    '''    "gate_focal_gamma": 1.0,          # v0.8.1: was 2.0
    "gate_class_weights": None,
    "line_logsigma_trainable": False,
    "x_ifu_resolution_ev": X_IFU_RESOLUTION_EV,

    "prior_zone_conditioning": True,   # v0.8.2
    "zone_probs": zone_probs.tolist(),
}''')
replace(i, "# Configuration for later saving and reference - v0.8.1",
           "# Configuration for later saving and reference - v0.8.2")
replace(i,
    '"gate_target_bandwidth_mode": "resolution",',
    '"gate_target_bandwidth_mode": "resolution",  # "exact" tried and rejected, v0.8.2')

# ---------------------------------------------------------------------
# Model build cell: pass prior_zone_conditioning / zone_probs
# ---------------------------------------------------------------------
i = find("gate_class_weights=model_config[\"gate_class_weights\"],")
replace(i,
    '''    gate_class_weights=model_config["gate_class_weights"],
    line_logsigma_init=line_logsigma_init,
    line_logsigma_trainable=False,
)''',
    '''    gate_class_weights=model_config["gate_class_weights"],
    line_logsigma_init=line_logsigma_init,
    line_logsigma_trainable=False,

    # --- v0.8.2: widen the prior's conditioning with the routing zone ---
    prior_zone_conditioning=model_config["prior_zone_conditioning"],
    zone_probs=zone_probs,
)''')

# ---------------------------------------------------------------------
# Load-for-generation cell: path + config
# ---------------------------------------------------------------------
i = find('save_dir = f"trained_models/v0_8_1_{SOURCE}"\nmodel_name = f"mix_{SOURCE}"\n\nwith open')
replace(i, 'save_dir = f"trained_models/v0_8_1_{SOURCE}"',
           'save_dir = f"trained_models/v0_8_2_{SOURCE}"')

# ---------------------------------------------------------------------
# Final save cell
# ---------------------------------------------------------------------
i = find('save_dir = f"trained_models/v0_8_1_{SOURCE}"\nmodel_name = f"mix_{SOURCE}"\n\nsave_info')
replace(i, 'save_dir = f"trained_models/v0_8_1_{SOURCE}"',
           'save_dir = f"trained_models/v0_8_2_{SOURCE}"')
replace(i,
    '''    notes=(
        "v0.8 mixture energy head: conditional RQS-flow continuum (24 bins, 3 "
        "transforms) with a CDF pre-warp, plus fixed-position Gaussian lines with "
        "widths pinned to the X-IFU 4 eV FWHM resolution. Energy head and gate "
        "conditioned on the latent z (energy_flow_condition='z_cond') with a "
        "learnable conditional coupling prior p(z|cond), so energy<->geometry "
        "coupling survives generation. Gate supervised with a focal-weighted "
        "auxiliary CE (w_gate_aux=2.0, gamma=2.0) against the ~99% continuum "
        "majority. Geometry unchanged from v0.7.2 (quantile u_r/u_v/phi_r/phi_v)."
    ),''',
    '''    notes=(
        "v0.8.2 mixture energy head: conditional RQS-flow continuum (24 bins, 3 "
        "transforms) with a CDF pre-warp, plus fixed-position Gaussian lines with "
        "widths pinned to the X-IFU 4 eV FWHM resolution. Energy head and gate "
        "conditioned on the latent z (energy_flow_condition='z_cond') with a "
        "learnable conditional coupling prior p(z|cond) additionally conditioned "
        "on the routing zone (prior_zone_conditioning=True), so energy<->geometry "
        "coupling survives generation AND rare-line routing is correctly placed "
        "in latent space. Gate supervised with a focal-weighted auxiliary CE "
        "(w_gate_aux=2.0, gamma=1.0) against the continuum majority, with "
        "resolution-bandwidth gate targets (bandwidth_mode='exact' was tried and "
        "rejected, see docs/v0.8.1_line_truth.md section 14.2). Geometry "
        "unchanged from v0.7.2 (quantile u_r/u_v/phi_r/phi_v)."
    ),''')

# ---------------------------------------------------------------------
# Generated-particles export path
# ---------------------------------------------------------------------
i = find('filepath=f"./checkpoints/generated_particles/v0_8_1_{SOURCE}_generated.txt",')
replace(i,
    'filepath=f"./checkpoints/generated_particles/v0_8_1_{SOURCE}_generated.txt",',
    'filepath=f"./checkpoints/generated_particles/v0_8_2_{SOURCE}_generated.txt",')

# ---------------------------------------------------------------------
# Notes markdown at the end
# ---------------------------------------------------------------------
i = find("## Notes on this run")
set_src(i, r"""## Notes on this run

**Configuration.** Flow continuum (24 bins, 3 transforms) + CDF pre-warp; fixed
lines pinned to 4 eV FWHM at **EADL** energies; `energy_flow_condition="z_cond"`;
`prior="coupling"` with **`prior_zone_conditioning=True`** (v0.8.2); gate targets
with a **resolution** bandwidth (`"exact"` tried and rejected, see below);
`w_gate_aux=2.0`, `gate_focal_gamma=1.0`; 40 epochs on CPU (~48 min CR, ~33 min
Small).

**What v0.8.2 changed, and what it did not touch.** Prior zone-conditioning is
the one confirmed improvement: on CR, 3 seeds, it took Al Kα1 from
0.702 ± 0.290 (FAIL) to 0.959 ± 0.025 (PASS) with coupling unaffected
(0.0345 ± 0.0048 → 0.0285 ± 0.0101). Everything else inherited from v0.8.1 —
EADL line positions, resolution-bandwidth gate targets, `gate_focal_gamma=1` —
is unchanged.

**Rejected: `bandwidth_mode="exact"`.** Raw Geant4 crossing data has no
detector response applied, so fluorescence lines really are exactly
monoenergetic (verified: 7,542 Cu Kα1 events at one identical float64 energy
in CR). That physics is correct, and it suggested labelling gate targets by
exact match instead of a Gaussian kernel of the detector resolution. Measured
on CR it did not fix the line it targeted (Cu Kα1 routing 2.011 vs. the
confirmed 2.052 — no change), it broke the confirmed Al Kα1 result (→ 0.578),
and it pushed the coupling residual outside its bar (→ 0.054). The soft
resolution kernel appears to regularize the gate in a way a hard 0/1 target
does not. Full account: `docs/v0.8.1_line_truth.md` §14.2. Do not re-try
without new information.

**This run.** If `SOURCE = "Small"`, this is the **first v0.8.2 measurement**
on this source — Phase C's 3-seed confirmation above was CR-only. Compare this
checkpoint's acceptance report against the v0.8.1 Small baseline
(`trained_models/v0_8_1_Small`) before treating any Small-specific number as
confirmed.

**Still open.**
- Per-line `gate_class_weights`, if the acceptance report still shows a
  heterogeneous per-line gap — with a two-sided clip. Already tried once on CR
  with no effect distinguishable from noise (`docs/v0.8.1_line_truth.md` §10).
- CR's low-energy Compton edge is still slightly rounded; a CDF-warp knot boost
  and `continuum_flow_bins` 24→32 were tried together and **regressed** every
  CR band (§11) — if retried, vary the two levers separately.
- An intermittent optimizer crash during `fit` — `Incompatible shapes: [N] vs.
  [0]` inside Adam's `apply_gradients`, not reproduced deterministically, root
  cause not identified. See `docs/v0.8.1_line_truth.md` §15. If `fit` fails
  with a shape mismatch below, just re-run the cell.
- Multi-seed error bars for whatever source this notebook is run on beyond
  seed 42 — use `tools/run_v0_8_real.py --seed <7|13>` and
  `tools/acceptance_v0_8.py --seeds 42 7 13`.
- The defaults flip (making `continuum_mode="flow"`, `prior="coupling"`,
  `prior_zone_conditioning=True` etc. the constructor defaults) is gated
  behind the go/no-go.

**Escape peaks.** Requested and implemented as candidates
(`--escape-peaks` in `tools/build_candidate_lines_from_geant4.py`, derived from the
detector volumes' materials in the GDML: Au absorber, Si plate and CryoAC, Ge
thermistor, Cu clamps). All 358 candidates were audited against the real spectra and
**none is present** — expected, since this training data is the spectrum of particles
*crossing* a virtual sphere, not energy *deposited* in the detector, and an escape
peak is a deposited-energy artifact. The escaping fluorescence photon appears in this
data as its own line, which is already modelled. Documented as a null result in
`docs/v0.8.1_line_truth.md` §5.""")

out = pathlib.Path("MAGI_v0_8_2.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out} with {len(cells)} cells")
