# Graph Report - MAGI-dev  (2026-08-23)

## Corpus Check
- Large corpus: 445 files · ~1,191,226 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 1036 nodes · 1877 edges · 71 communities (55 shown, 16 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.85)
- Token cost: 3,200 input · 1,500 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 67
- Community 68
- Community 69

## God Nodes (most connected - your core abstractions)
1. `CVAE_MixEnergy_ContPhi_TaskAdaptive` - 42 edges
2. `CVAE_CatEnergy_CatUV_TaskAdaptive` - 34 edges
3. `v0.8.2 release validation research` - 28 edges
4. `CVAE_CatEnergy_ContPhi_TaskAdaptive` - 26 edges
5. `CVAE_CatEnergy_CatUV` - 25 edges
6. `CVAE_CatEnergy_ContGeom_TaskAdaptive` - 25 edges
7. `v0.8 learnable prior theory` - 23 edges
8. `ConditionalCouplingPrior` - 22 edges
9. `save_full_circuit()` - 22 edges
10. `v0.8.1 line truth` - 22 edges

## Surprising Connections (you probably didn't know these)
- `VAE with a VampPrior (Tomczak & Welling 2018)` --semantically_similar_to--> `ConditionalCouplingPrior`  [INFERRED] [semantically similar]
  docs/v0.8_learnable_prior_theory.md → MAGI_package/magi/core/priors.py
- `mix_CR Full Circuit (seed13, Expected Conductance)` --references--> `save_full_circuit()`  [INFERRED]
  trained_models/v0_8_2_priorzone_CR_seed13/mix_CR_full_circuit.html → MAGI_package/magi/utils/full_circuit.py
- `mix_Small Full Circuit (Gradient x Activation)` --references--> `save_full_circuit()`  [INFERRED]
  trained_models/v0_8_2_priorzone_Small/mix_Small_full_circuit.html → MAGI_package/magi/utils/full_circuit.py
- `mix_CR Routing Circuit (v0_8_2_priorzone_CR)` --references--> `save_routing_circuit()`  [INFERRED]
  trained_models/v0_8_2_priorzone_CR/mix_CR_routing_circuit.html → MAGI_package/magi/utils/circuit_viz.py
- `mix_Small Routing Circuit` --references--> `save_routing_circuit()`  [INFERRED]
  trained_models/v0_8_2_priorzone_Small/mix_Small_routing_circuit.html → MAGI_package/magi/utils/circuit_viz.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **CR flux closure comparison for SRON XFDM Detector 1: full Geant4 vs two MAGI training variants** — paper_figures_fig_cr_vs_full_full_geant4_run_cryoac, paper_figures_fig_cr_vs_full_magi_censored_training_set, paper_figures_fig_cr_vs_full_magi_ingoing_fixed_15_jobs, paper_figures_fig_cr_vs_full_sron_xfdm_detector1_tes_array [EXTRACTED 1.00]
- **40K albedo training-population correction chain** — docs_magi_state_reference_result_3_40k_sron_buildingmodel, docs_magi_state_reference_training_population_is_albedo, docs_magi_state_reference_sphere_center_bug_2500mm, docs_magi_state_reference_gap_22x_superseded_explanation, docs_magi_state_reference_eventaction_cc [EXTRACTED 1.00]
- **DM1.2 CryoSphere/PlateCu1 overlap bug and fix** — docs_magi_state_reference_dm1_2_laboratory_cryostat, docs_magi_state_reference_cryosphere, docs_magi_state_reference_platecu1, docs_magi_state_reference_geometry_overlap_bug_39pct [EXTRACTED 1.00]
- **SRON CR deficit investigation (ingoing-cut convention mismatch)** — docs_magi_state_reference_result_2_sron_xifu_cosmic_rays, docs_magi_state_reference_sron_deficit_24pct, docs_magi_state_reference_ingoing_particle_cut_inconsistency, docs_magi_state_reference_sron_x_ifu_model, docs_magi_state_reference_dm1_2_laboratory_cryostat [EXTRACTED 1.00]
- **v0.8.2 Four-Source Line-Recovery Validation Set (CR/Small/Torio/Radio)** — trained_models_v0_8_2_cr_ingoingfix_mix_cr_summary_mix_cr, trained_models_v0_8_2_radio_colab_gpu_incomplete_readme, trained_models_v0_8_2_torio_colab_gpu_incomplete_readme, docs_v0_8_2_release_validation_research [EXTRACTED 1.00]
- **MAGI CVAE model-class version lineage** — magi_package_magi_core_model_cvae_catenergy_catuv, magi_package_magi_core_model_cvae_catenergy_catuv_taskadaptive, magi_package_magi_core_model_cvae_catenergy_contgeom_taskadaptive, magi_package_magi_core_model_cvae_catenergy_contphi_taskadaptive, magi_package_magi_core_model_cvae_mixenergy_contphi_taskadaptive [EXTRACTED 1.00]
- **Field validation-layer citations underpinning the seven-layer framework** — ref_calochallenge_2024, ref_kansal_2023, ref_lopez_paz_2016, ref_naeem_2020, ref_meehan_2020, ref_atlfast3_2022, ref_hashemi_krause_2024 [EXTRACTED 1.00]
- **v0.8 line planning/evidence document chain** — docs_v0_8_beta_plan, docs_v0_8_fixing_plan, docs_v0_8_learnable_prior_plan, docs_v0_8_learnable_prior_theory, docs_v0_8_v072_comparison, docs_v0_8_1_improvement_plan, docs_v0_8_1_line_truth, docs_v0_8_2_roadmapforadoption [EXTRACTED 1.00]
- **MAGI vs KDSource Comparison Figure Set** — paper_figures_fig_kds_marginals_marginal_comparison, paper_figures_fig_kds_lines_line_comparison, paper_figures_fig_kds_correlations_correlation_comparison [INFERRED 0.85]
- **Geometry/Energy-Head A/B Study Family (mix_CR variants)** — trained_models_coupling_baseline_mix_cr_summary_mix_cr_baseline, trained_models_coupling_coupled_mix_cr_summary_mix_cr_coupled [INFERRED 0.85]
- **Task-Adaptive Energy CVAE Run Family (5 checkpoints)** — trained_models_task_adaptive_energy_cr_run_001_task_adaptive_cvae_energy_summary_run, trained_models_task_adaptive_energy_run_002_task_adaptive_cvae_energy_summary_run, trained_models_task_adaptive_energy_run_003_task_adaptive_cvae_energy_summary_run, trained_models_task_adaptive_energy_run_004_task_adaptive_cvae_energy_summary_run, trained_models_task_adaptive_energy_run_005_task_adaptive_cvae_energy_summary_run [INFERRED 0.85]
- **Full-Geant4 baseline and MAGI v0.8.2 samples jointly establish the flux-agreement claim** — paper_figures_fig1_spectra_full_geant4_baseline, paper_figures_fig1_spectra_magi_v0_8_2_generated_events, paper_figures_fig1_spectra_magi_geant4_flux_agreement [INFERRED 0.85]
- **Break-even Cost Analysis Across DM1.2 and SRON Geometries** — paper_figures_fig_breakeven_cost_amortization_method, paper_figures_fig_breakeven_full_simulation_baseline, paper_figures_fig_breakeven_dm1_2_result, paper_figures_fig_breakeven_sron_result [INFERRED 0.85]
- **Two-detector Geant4-vs-MAGI validation shown together in Figure 1** — paper_figures_fig1_spectra_figure, paper_figures_fig1_spectra_dm1_2_mu_panel, paper_figures_fig1_spectra_sron_xfdm_tes_panel [INFERRED 0.95]

## Communities (71 total, 16 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (39): energy_condition_geometry flag, Head-Coupling A/B Test, function, CVAE_MixEnergy_ContPhi_TaskAdaptive, Multiply one task weight by `factor`, floored at `min_value`. Returns (old,…, Run one gradient step on one batch. Called by Keras, not directly. The loss is…, Run one validation step on one batch. Called by Keras, not directly. Same loss…, Inference-mode decode: latent + conditioning -> per-head parameters. (+31 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (32): v0.8.1 model schematic, ConditionalRQSFlow, _piecewise_linear(), Conditional normalizing-flow density for the continuum term of the v0.8 mixture…, Per-sample chained RQS bijector (forward maps base u -> y_std)., Map y -> standardized working coordinate w, returning (w, log|dw/dy|), both…, Map standardized working coordinate w -> y, shape (batch,)., Fully normalized log density log p(y | feat), shape (batch,). (+24 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (45): _append_detector_binary_chunk(), _ensure_parent_dir(), _filter_non_transport_particles(), generate_detector_input_file(), generate_detector_table_to_file(), generated_physics_to_detector_dataframe(), _normalize_idx_to_type(), _normalize_output_format() (+37 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (34): build_conditioning_and_weights(), build_tf_datasets(), filter_particle_types_and_discretize_uv(), frac_per_type(), make_dummy_targets(), Dataset construction utilities for MAGI. This module supports three geometry…, Report diagnostics for the legacy discrete-u_v dataset. Printed sanity check…, Report diagnostics for the v0.7 continuous-geometry dataset. Printed sanity… (+26 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (38): Conditional RQS-Flow Continuum (24 bins, 3 transforms, CDF pre-warp), energy_flow_condition='z_cond', energy_vs_impact_parameter metric artefact (species-mix confound), Fixed-Position Gaussian Lines (pinned to X-IFU 4 eV FWHM), f_min ~ 1e-4 line-fraction modelling threshold, Focal-Weighted Auxiliary CE Gate Supervision (w_gate_aux=2.0, gamma=1.0), Four-Source Line-Recovery Comparison (CR/Small/Torio/Radio), KDSource (kernel-density Monte Carlo source resampler) (+30 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (35): prior_zone_conditioning ([type, zone] prior conditioning), zone_probs Per-Type Routing, MAGI usage guide, Interactive HTML "circuit" visualizations of the v0.8 mixture head's…, Load a v0.8 mixture checkpoint's saved config/metadata and write its…, Build a self-contained interactive HTML "circuit" diagram of how the v0.8…, render_routing_circuit_html(), save_routing_circuit() (+27 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (33): Geometry utilities for MAGI., Legacy v0.6 helper. Compute expected continuous u_v from predicted categorical…, Reconstruct x and y coordinates on a sphere from physical u_r. This is useful…, Reconstruct u_r = cos(theta_r) from s_r = arctanh(u_r)., Reconstruct x, y, z coordinates on a sphere from physical u_r. This is useful…, Legacy v0.6 helper. Reconstruct x and y coordinates on a sphere from: - s_r =…, Legacy v0.6 helper. Reconstruct x, y, z coordinates on a sphere from: - s_r =…, Reconstruct vx and vy from: - uv = cos(theta_v) - phi_pair = (cos(phi_v),… (+25 more)

### Community 7 - "Community 7"
Cohesion: 0.08
Nodes (37): bin_counts(), build_energy_bins(), build_feature_dataframe(), build_physical_features(), compute_primary_fraction(), confirm_unresolved_candidate_lines(), detect_energy_lines(), detect_line_bins() (+29 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (35): CryoSphere, CVAE_MixEnergy_ContPhi_TaskAdaptive, DM127 (anti-coincidence detector, ACDSD), DM1.2 iso rerun (vs. focused macro), DM1.2 laboratory cryostat model, Efficiency factor (epsilon), EventAction.cc, Figure 1 — spectral comparison (+27 more)

### Community 9 - "Community 9"
Cohesion: 0.11
Nodes (30): build_real_generated_featureframes(), compare_hist_with_residuals(), Comparison plots between generated and real distributions., Print min/max of every reconstructed quantity, real vs generated. A coarse but…, Print the mean |(cos, sin)| for phi_r and phi_v, real vs generated. These…, Pack the real and generated arrays into DataFrames for the plot helpers.…, Overlay real vs generated histograms with a residual panel underneath. Residual…, Save and/or show a matplotlib figure. Parameters ---------- fig :… (+22 more)

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (26): initialize_environment(), print_tf_info(), Configuration helpers for MAGI., Configure the runtime environment. Parameters ---------- seed : int Random seed…, Print TensorFlow / Keras version and visible devices. Use it right after…, plot_training(), High-level user-facing API for MAGI. A thin convenience layer over the modules…, Compile and train a model in a single call. (+18 more)

### Community 11 - "Community 11"
Cohesion: 0.10
Nodes (28): Expected Conductance (Integrated Gradients for internal units), filter_particle_types_continuous_geometry(), Continuous-geometry dataset builder. Supports: v0.7: ParticleName, E_idx,…, Split raw feature matrices into train/val/test. Works for both: - dataset_mode…, split_feature_data(), load_candidate_energy_lines(), Load a candidate-energy-lines payload previously written by…, build_gate_targets() (+20 more)

### Community 12 - "Community 12"
Cohesion: 0.11
Nodes (27): build_model_metadata(), _ensure_model_saveable(), extract_callback_metadata(), Checkpointing utilities for MAGI. This module provides robust save utilities…, Convert common non-JSON-serializable objects into JSON-safe objects., Collect all information needed to reproduce or reuse a trained model., Save Keras History object or plain history dictionary., Save task weights if the model has a task_weights attribute. (+19 more)

### Community 13 - "Community 13"
Cohesion: 0.11
Nodes (11): Compact training monitor for task-adaptive models. Prints at the end of each…, Reduce selected task weights when their monitored validation metric: 1) is…, Print the selected metrics, the learning rate and the task weights., Adaptive training callbacks for MAGI., Compute validation energy distribution metrics at epoch end. It adds to logs: -…, Generate a validation-sized sample and score its energy spectrum. Adds the…, Reset the per-task plateau/cooldown bookkeeping., Check each task's monitor and decay its weight if it has plateaued. (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.16
Nodes (23): _count_params_from_weights(), _fmt_module_line(), _is_mixture_energy(), _line_table(), _module_param_dict(), _module_status_line(), _print_energy_head(), _print_generative_story() (+15 more)

### Community 15 - "Community 15"
Cohesion: 0.21
Nodes (21): _build_tiny_model(), _n_hidden_layers(), compute_full_circuit_trace / render_full_circuit_html. Uses a tiny…, Validates the exact trapezoidal-Riemann-sum recipe…, renderType() colors each unit by setting the `fill` presentation attribute. A…, The rank-based node color cannot answer 'is this layer too wide' - the absolute…, The color scale must actually use its range. Usage is heavy-tailed, so a naive…, X_cont_test in the [ur_q,uv_q,phi_r_q,phi_v_q,energy_y,… (+13 more)

### Community 16 - "Community 16"
Cohesion: 0.16
Nodes (19): Colab CR retrain instructions, Plan to 24 August meeting, build_candidate_lines(), build_escape_peak_lines(), build_parser(), main(), _parse_fluor_file(), parse_fluor_lines() (+11 more)

### Community 17 - "Community 17"
Cohesion: 0.17
Nodes (18): Utility functions for MAGI., plot_correlation_matrix(), plot_covariance_matrix(), plot_dist(), plot_dist_by_class(), plot_pairgrid_physics(), plot_pairwise_sample(), Plotting utilities for MAGI. (+10 more)

### Community 18 - "Community 18"
Cohesion: 0.14
Nodes (11): Task-Adaptive Energy Loss Down-weighting, CVAE_CatEnergy_CatUV_TaskAdaptive, Conditional Variational Autoencoder with task-adaptive decoder design. Main…, Trackers Keras resets each epoch and reports in the logs. Every entry here…, Current weight of one task in the reconstruction loss., Set one task weight (see decay_task_weight on mid-fit changes)., task_adaptive_cvae_energy (CR_run_001), task_adaptive_cvae_energy (run_002) (+3 more)

### Community 19 - "Community 19"
Cohesion: 0.12
Nodes (17): v0.8 learnable prior theory, NCP-VAE (Aneja et al. 2021), Resampled Priors for VAEs (Bauer & Mnih 2019), CaloFlow (Krause & Shih 2021), Variational Lossy Autoencoder (Chen et al. 2017), RealNVP (Dinh et al. 2017), Neural Spline Flows (Durkan et al. 2019), ELBO surgery (Hoffman & Johnson 2016) (+9 more)

### Community 20 - "Community 20"
Cohesion: 0.18
Nodes (16): build_parser(), inbin_peak_retention(), load_energy_bins(), load_real_energies(), main(), make_report(), model_predicted_and_generated(), _plot() (+8 more)

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (7): GEEANNT (pre-package model line), CVAE_CatEnergy_CatUV, Trackers Keras resets each epoch and reports in the logs. Every entry here…, Reparametrization trick: z = mu + sigma * eps, eps ~ N(0, I)., Conditional Variational Autoencoder with: - categorical energy head - Gaussian…, build_model(), Convenience wrapper around the v0.6 CVAE constructor (CVAE_CatEnergy_CatUV:…

### Community 22 - "Community 22"
Cohesion: 0.13
Nodes (15): load_detector_table(), load_normalization_summary(), _peek_ncols(), Input/output utilities for MAGI datasets., Print basic integrity checks on the loaded dataframe. Prints row count, a head…, Save a compute_primary_fraction()/build_physical_features() normalization dict…, Load a normalization dict previously written by save_normalization_summary().…, Save a candidate-energy-lines payload (as built by… (+7 more)

### Community 23 - "Community 23"
Cohesion: 0.13
Nodes (7): Core neural-network components for MAGI., CVAE_CatEnergy_ContGeom_TaskAdaptive, Continuous-geometry Task-Adaptive CVAE. Continuous targets: y_cont = [ u_r_q,…, Trackers Keras resets each epoch and reports in the logs. Every entry here…, Current weight of one task in the reconstruction loss., Set one task weight (see decay_task_weight on mid-fit changes)., Reparametrization trick: z = mu + sigma * eps, eps ~ N(0, I).

### Community 24 - "Community 24"
Cohesion: 0.26
Nodes (14): _build_model(), v0.8.2 Phase C candidate 1: conditioning the coupling prior on more than…, A config with no prior_zone_conditioning/zone_probs key (every checkpoint saved…, train_step's prior_cond must be built from the REAL gate_target columns (not…, _required_keys_snapshot(), test_generate_samples_zone_from_type_conditional_table_and_runs(), test_legacy_checkpoint_without_zone_keys_loads_unchanged(), test_train_step_runs_and_uses_real_zone_at_training_time() (+6 more)

### Community 25 - "Community 25"
Cohesion: 0.21
Nodes (5): CouplingLayer, make_comparison_plots(), RealNVP, theta_from_minus_z(), train_flow()

### Community 26 - "Community 26"
Cohesion: 0.15
Nodes (6): CVAE_CatEnergy_ContPhi_TaskAdaptive, Continuous-geometry Task-Adaptive CVAE. Continuous targets: y_cont = [ u_r_q,…, Trackers Keras resets each epoch and reports in the logs. Every entry here…, Current weight of one task in the reconstruction loss., Set one task weight (see decay_task_weight on mid-fit changes)., Reparametrization trick: z = mu + sigma * eps, eps ~ N(0, I).

### Community 27 - "Community 27"
Cohesion: 0.32
Nodes (10): EADL vs Bearden fluorescence line positions, Per-line gate_class_weights calibration (reverted), Multi-seed validation requirement, v0.8.1 improvement plan, v0.8.1 line truth, v0.8.2 plan and open questions, v0.8.2 roadmap for adoption, v0.8 vs v0.7.2 comparison (+2 more)

### Community 28 - "Community 28"
Cohesion: 0.23
Nodes (8): Aggregated-posterior / prior mismatch, Learnable conditional coupling prior fix, v0.8 fixing plan for energy mixture head, v0.8 IAF integration plan, v0.8 learnable prior plan, Diagnosing and Enhancing VAE Models / two-stage VAE (Dai & Wipf 2019), Harder synthetic stress test for CVAE_MixEnergy_ContPhi_TaskAdaptive (v0.8 Part…, Synthetic stress test reproducing SMALL's specific pathology (see…

### Community 29 - "Community 29"
Cohesion: 0.24
Nodes (11): DM1.2 Iso Corner Plot (default output), SRON XFDM Cosmic-Ray Corner Plot, DM1.2 Muons (Iso) Corner Plot, DM1.2 Muons (Iso) Corner Plot, Paper Version, SRON XFDM 40K (Small) Corner Plot, corner(), features(), load_gen() (+3 more)

### Community 30 - "Community 30"
Cohesion: 0.20
Nodes (6): draw_resid(), poisson_pull(), poisson_ratio_ci(), Bottom-panel residuals in whichever convention RESID selects., Exact CI on (n_m/expo_m)/(n_r/expo_r) for two Poisson counts. Conditional on T…, Signed-root likelihood-ratio pull for two Poisson counts, in sigma. Same…

### Community 31 - "Community 31"
Cohesion: 0.27
Nodes (9): corr_resid(), features(), kde_resample(), load_magi(), load_real(), main(), marginal_rms(), RMS of the generated/reference density ratio, over well-populated bins. Bins… (+1 more)

### Community 32 - "Community 32"
Cohesion: 0.22
Nodes (4): Sample `n_samples` events from the prior, conditioned on `cond`. Returns the…, Sample `n_samples` events from the prior, conditioned on `cond`. Returns the…, Sample `n_samples` events from the prior, conditioned on `cond`. Returns the…, Sample `n_samples` events from the prior, conditioned on `cond`. Returns the…

### Community 33 - "Community 33"
Cohesion: 0.27
Nodes (10): _lines(), bandwidth_mode='exact' for magi.build_gate_targets (v0.8.2 Phase C follow-up on…, An event a few eV from the line - which 'resolution' mode (sigma ~1.7 eV for 4…, Cu Kalpha1/Kalpha2 analogue: two lines 21 eV apart, plus continuum events…, Direct A/B on the same synthetic data: 'resolution' mode hands a continuum…, test_close_doublet_no_cross_talk(), test_exact_event_gets_full_line_weight_default_floor(), test_invalid_bandwidth_mode_raises() (+2 more)

### Community 34 - "Community 34"
Cohesion: 0.20
Nodes (10): _exported_names(), parametrize, Every name in magi.__all__ must carry a hover-usable docstring. VSCode (and any…, Map name -> (path, ast node) for every top-level def/class in the package., Names bound by a module-level assignment anywhere in the package. Covers the…, __all__ must not advertise a name the package does not define. Guards the test…, test_exported_names_are_resolvable(), test_public_symbol_is_documented_for_hover() (+2 more)

### Community 35 - "Community 35"
Cohesion: 0.29
Nodes (10): featurise(), kde_resample(), load_table(), main(), memoriser(), nn_dist(), KDSource's smoothed bootstrap: draw a training point, perturb by the kernel.…, NEGATIVE CONTROL. A deliberate memoriser: resample reference points and barely… (+2 more)

### Community 36 - "Community 36"
Cohesion: 0.24
Nodes (9): log(), plot_matrix_triptych(), plot_pairgrid(), Post-run multivariate validation for the v0.8 real CR+Small run. Reloads each…, Real physical variables of the filtered events., Deterministic rebuild of the driver's Phase-A/B pipeline for one source.…, real_dataframe(), rebuild_pipeline() (+1 more)

### Community 37 - "Community 37"
Cohesion: 0.33
Nodes (10): DM1.2 efficiency: eps=194+-6 (all), eps=200+-7 (MIP), ideal 1/chi=200, Panel (a): DM1.2 mu deposited-energy spectrum, Figure 1: MAGI v0.8.2 vs Full Geant4 Validation Spectra, Full Geant4 simulation (reference baseline curve), Claim: MAGI v0.8.2 samples reproduce full-Geant4 deposited-energy flux, MAGI v0.8.2 generated particle events (plotted points), MIP-selection band highlighted at 100-300 keV in DM1.2 spectrum, Pull (sigma) residual sub-panels showing MAGI-Geant4 agreement within roughly 2-3 sigma (+2 more)

### Community 38 - "Community 38"
Cohesion: 0.33
Nodes (6): render_routing_circuit_html / save_routing_circuit. Structural checks only - no…, test_embedded_data_is_valid_json_matching_input(), test_rejects_shape_mismatch(), test_renders_self_contained_html(), test_wraps_many_zones_without_error(), _toy()

### Community 39 - "Community 39"
Cohesion: 0.57
Nodes (7): CR - SRON XFDM Detector 1 Flux Comparison Figure, Full Geant4 simulation, RUN CRYOAC (cosmic-ray reference), MAGI flux, censored training set, MAGI/full Geant4 flux ratio result (CR spectrum closure), MAGI flux, ingoing-fixed (15 jobs), MIP energy region (2.5-4 keV), SRON XFDM Detector 1 (TES array)

### Community 40 - "Community 40"
Cohesion: 0.48
Nodes (6): band_wasserstein(), coupling_residuals(), log(), Phase A1 (docs/v0.8.2_RoadmapForAdoption.md S5, S6): score v0.7.2…, score_checkpoint(), verdict()

### Community 41 - "Community 41"
Cohesion: 0.60
Nodes (5): _build_and_save_tiny_checkpoint(), End-to-end check of the Geant4 integration path: a Geant4 macro's…, _run_generate_geant_source(), test_geant4_export_binary_format(), test_geant4_export_text_format()

### Community 42 - "Community 42"
Cohesion: 0.47
Nodes (5): audit_checkpoint(), log(), Phase A2 (docs/v0.8.2_RoadmapForAdoption.md S5.2a, S6): posterior-vs-prior gate…, Mirror tools/acceptance_v0_8.py's rebuild_pipeline, but only as far as the…, rebuild_dataset_pack()

### Community 43 - "Community 43"
Cohesion: 0.47
Nodes (4): Transform MAGI_v0_8_1.ipynb -> MAGI_v0_8_2.ipynb, same structure. v0.8.2…, replace(), set_src(), src()

### Community 44 - "Community 44"
Cohesion: 0.53
Nodes (5): feats(), fit_ratio(), load(), main(), AUC and CALIBRATED classifier for a-vs-b (a=real=1). Calibration matters: w =…

### Community 45 - "Community 45"
Cohesion: 0.90
Nodes (5): MAGI Cost-Amortization-Across-Design-Variants Method, DM1.2-MAGI Break-even Result (N=1.43, 2.09x cost at N=6), Break-even, Reuse Across Design Variants (Figure), Full Geant4 Simulation Cost Baseline (linear cost per design variant), SRON-MAGI Break-even Result (N=1.93, 3.33x cost at N=6)

### Community 48 - "Community 48"
Cohesion: 0.50
Nodes (4): chunked_generate(), log(), Real CR+Small v0.8 run (flow continuum + coupling prior + z_cond, per-line…, Generate + reconstruct in chunks to bound peak memory; returns concatenated…

### Community 49 - "Community 49"
Cohesion: 0.60
Nodes (4): impact(), load(), main(), RAW output is 13 columns (pos 7-9, dir 10-12); CLEANED is 9 (3-5, 6-8).

### Community 50 - "Community 50"
Cohesion: 0.50
Nodes (4): Checkpoint config-match guard (config_version), v0.8 beta coding plan, load_task_adaptive_model_for_generation(), Load a saved task-adaptive MAGI model for generation. Supports: -…

## Ambiguous Edges - Review These
- `MAGI flux, censored training set` → `MIP energy region (2.5-4 keV)`  [AMBIGUOUS]
  paper_figures/fig_cr_vs_full.pdf · relation: conceptually_related_to

## Knowledge Gaps
- **57 isolated node(s):** `build.sh script`, `queue_cr_memtest.sh script`, `queue_kdsource_bench.sh script`, `Seven-layer generative-model validation framework`, `AtlFast3 (ATLAS Collaboration 2022)` (+52 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `MAGI flux, censored training set` and `MIP energy region (2.5-4 keV)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `CVAE_MixEnergy_ContPhi_TaskAdaptive` connect `Community 0` to `Community 1`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 12`, `Community 16`, `Community 48`, `Community 50`, `Community 23`, `Community 26`, `Community 27`?**
  _High betweenness centrality (0.124) - this node is a cross-community bridge._
- **Why does `CVAE_CatEnergy_CatUV_TaskAdaptive` connect `Community 18` to `Community 0`, `Community 32`, `Community 1`, `Community 3`, `Community 6`, `Community 59`, `Community 10`, `Community 12`, `Community 50`, `Community 21`, `Community 23`, `Community 58`, `Community 27`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `CVAE_CatEnergy_CatUV` connect `Community 21` to `Community 0`, `Community 32`, `Community 1`, `Community 3`, `Community 6`, `Community 10`, `Community 18`, `Community 23`, `Community 27`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **What connects `build.sh script`, `queue_cr_memtest.sh script`, `queue_kdsource_bench.sh script` to the rest of the system?**
  _57 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05081081081081081 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.052525252525252523 - nodes in this community are weakly interconnected._