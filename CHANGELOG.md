# Changelog

## 1.0.0 (2026-11-11)

First public release of MAGI (CVAE).

Added
- Prior refit as the last training stage: `magi.refit_prior`, `save_refit_model`, `refit_prior_run`.
- Uncollided primaries: a pinned E = E_prim component in the energy head of primary-conditioned models.
- Fixed-position line components in the mixture energy head (511 keV and detector lines).
- Virtual tracking ball and first-entry rule (`select_first_entry`, `--first-entry`, `crossing_rule` in the metadata), validated on DM1.2 at R = 95 and 105 mm.
- Inward-constrained direction head for the outgoing-track problem.
- SRON Th (warp2) and Ra (warp) models with the energy-cap and line-criterion warp flags.
- IAPS Triple trainer with `--extra-lines` for the Cu K fluorescence energies.
- Geant4 binary export (GNTBIN) and text export.
- `save_routing_circuit` and `save_full_circuit`: interactive HTML of unit usage.

Documentation
- README and `CITATION.cff` link the ML4ASTRO3 proceedings paper, arXiv:2611.14710.

Known limits (fixes planned for 1.0.1)
- Th232 and mu `_qtfull` 5x closures, SRON swap, K-40 delta and seeds 8 and 9 are open.
- Sources in passive material (G4) and the Athena mass model (G5) are not covered.
- Geometry is a sphere only.
