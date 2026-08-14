# INCOMPLETE CHECKPOINT — do not load

Partially retrieved from Google Drive on 2026-08-13. **This directory is not a
loadable checkpoint.** Per `CLAUDE.md`, a run's files are one unit; the binaries
below are still missing, so `load_task_adaptive_model_for_generation` and
`scripts/generate_geant_source.py` will fail against this directory.

Source: `MyDrive/MAGI_data/trained_models/v0_8_2_Radio_colab_gpu`
(Drive folder id `18_b43Xc20Qw2gawqQEcgNn3ZtMrX_8S-`, account
`f.monastra97@gmail.com`).

## Present

| file | note |
|---|---|
| `mix_Radio_summary.txt` | run provenance |
| `mix_Radio_task_weights.json` | all 1.0 |
| `validation/wasserstein_scores.csv` | per-variable W1 |
| `validation/correlation_diff.csv` | 5×5 Δρ matrix |
| `validation/line_integral_recovery.csv` | 2 lines, recovery + significance |

## Missing — fetch from Drive

| file | size |
|---|---|
| `mix_Radio.weights.h5` | 3.48 MB |
| `mix_Radio_quantile_transformers.joblib` | 641 KB |
| `mix_Radio_metadata.json` | 31 KB |
| `mix_Radio_config.json` | 14 KB |
| `mix_Radio_history.json` | 26 KB |
| `validation/*.png` (7 files, incl. 1.4 MB pairgrid) | ~1.8 MB |
| `validation/covariance_diff.csv` | 593 B |

Fastest route: open the folder in Drive, **Download** (Drive returns a zip), and
unzip over this directory — that supersedes everything here, including this file.

## Config, verified against the Drive copy

Read from `mix_Radio_config.json` on Drive and **identical on every shared key**
to the local `v0_8_1_{CR,Small,Torio}` reference configs, so the four-source
line-recovery comparison in
[`../../docs/v0.8.2_release_validation_research.md`](../../docs/v0.8.2_release_validation_research.md)
§2.5 is on a consistent configuration:

```
model_class  CVAE_MixEnergy_ContPhi_TaskAdaptive     config_version 2
n_types 3    2 lines    latent_dim 8    hidden [128,128,64]    beta 0.2
continuum_mode flow      energy_flow_condition z_cond
continuum_flow_bins 24   _transforms 3   _interval 5.0   _warp cdf (256 knots)
n_continuum_components 1
gate_focal_gamma 1.0     gate_class_weights null
prior coupling (6 layers, [64,64], clamp 3.0)   prior_zone_conditioning true
line_logsigma_trainable false                   energy_sampling_temperature 1.0
min_log_sigma -6.0  max_log_sigma 1.5  sigma_target -2.0  lambda_sigma 0.001
stem_width 64  deep_decoder_hidden [128,128,64]
energy_branch_hidden [48,48]  energy_cont_head_hidden [64,32]
line_positions_y [-0.29158, -2.09660]
```

Training: 1,392,031 crossings, 40 epochs, 9.9 min on Colab GPU
(10.5 min including generation + validation).

## One flag

Radio's **raw φ_v marginal is W1 = 0.0549**, over the 0.05 bar (φ_v_q = 0.0407,
cphi_v = 0.0407). `tab:accept` states the bar for log₁₀E, which Radio passes at
0.0104 — so this is not a formal failure, but it travels with the number.
