#!/usr/bin/env python

import argparse
import json
import numpy as np
import joblib
import magi
import os


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


SPECIES_PDG = {"mu": 13, "mu-": 13, "gamma": 22, "e-": 11, "electron": 11}


def generate_groups(args, model_config, preprocessing, idx_to_type,
                    quantile_transformers, geometry_metadata, energy_bins,
                    energy_head_mode, energy_transform, qt_energy,
                    center, radius):
    """Group mode: whole showers of each primary, tagged with the primary's
    key, PDG and energy (magi.generate_primary_groups). With --species, one
    species; without, every species the --primary-spectra macro fires, in
    the macro's own intensity ratios (multi-species GPS, e.g. the SRON CR)."""
    if "multiplicity_table" not in preprocessing:
        raise ValueError(
            "--n-primaries needs a primary-conditioned checkpoint (trained with "
            "tools/train_dm12_primtruth.py); this metadata has no multiplicity_table"
        )
    pdg = None
    if args.species is not None:
        species = args.species.lower()
        pdg = SPECIES_PDG[species] if species in SPECIES_PDG else int(species)

    gps_macro = args.primary_spectra
    if gps_macro is None and pdg is not None:
        gps_macro = preprocessing.get("gps_macros", {}).get(str(pdg))
    if gps_macro is None:
        raise ValueError(
            "group mode needs --primary-spectra (the GPS macro to draw primaries "
            "from), or --species of a single-species training run"
        )

    primary_type_to_idx = {
        int(p): int(i) for i, p in preprocessing["idx_to_primary_type"].items()
    }
    model = magi.load_task_adaptive_model_for_generation(
        save_dir=args.save_dir,
        model_name=args.model_name,
        model_config=model_config,
        energy_bins=energy_bins,
        n_types=int(preprocessing["n_types"]),
        type_weights=preprocessing.get("type_weights", None),
        radius=radius,
        verbose=1,
    )
    if args.latent_bank is not None:
        model.latent_bank = magi.LatentBank.load(args.latent_bank, seed=args.seed)
        print(f"[MAGI] z from the latent bank {args.latent_bank} "
              f"({len(model.latent_bank.z_mean):,} training events, "
              f"K = {model.latent_bank.k_neighbours}), not the learned prior", flush=True)
    output_format = args.format
    magi.generate_primary_groups(
        model=model,
        filepath=args.output_file,
        species_pdg=pdg,
        n_primaries=args.n_primaries,
        gps_macro=gps_macro,
        multiplicity_table=preprocessing["multiplicity_table"],
        idx_to_type=idx_to_type,
        primary_type_to_idx=primary_type_to_idx,
        qt_primary_energy=quantile_transformers["qt_primary_energy"],
        center=center,
        radius=radius,
        geometry_metadata=geometry_metadata,
        energy_head_mode=energy_head_mode,
        energy_transform=energy_transform,
        qt_energy=qt_energy,
        output_format=output_format,
        seed=args.seed,
        key_offset=args.key_offset,
        verbose=1,
    )
    if args.latent_bank is not None:
        print(f"[MAGI] latent bank: {model.latent_bank.n_fallback} rows with no bank entry "
              f"took z from the prior", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Geant4 input particle file using a trained MAGI model."
    )

    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--output-file", required=True)

    parser.add_argument("--transformers-file", default=None)

    count = parser.add_mutually_exclusive_group(required=True)
    count.add_argument("--n-events", type=int,
                       help="independent crossings, one per Geant4 event")
    count.add_argument("--n-primaries", type=int,
                       help="HENABICS group mode (primary-conditioned models): "
                            "throw this many primaries and write each one's "
                            "crossings as one tagged group")
    parser.add_argument("--species", default=None,
                        help="group mode: primary PDG code, or mu/gamma/e-; "
                             "default: every species in --primary-spectra")
    parser.add_argument("--primary-spectra", default=None,
                        help="group mode: GPS macro the primary energies are drawn "
                             "from (default: the one the training run used), or the "
                             "run_summary.json of a full Athena run, whose thrown "
                             "histogram is then the E_prim spectrum")
    parser.add_argument("--primary-hist", dest="primary_spectra", default=None,
                        help="alias of --primary-spectra for a run_summary.json")
    parser.add_argument("--key-offset", type=int, default=0,
                        help="group mode: first GroupKey, to keep keys unique "
                             "across files")
    parser.add_argument("--latent-bank", default=None,
                        help="group mode: .npz written by tools/build_latent_bank.py; z is "
                             "drawn from the encoded training events instead of the prior")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--format", choices=["text", "binary"], default="binary")
    parser.add_argument("--max-energy-mev", type=float, default=None,
                        help="independent-crossing mode: reject and redraw crossings above this "
                             "energy (MeV), normally the largest energy in the training data; "
                             "the rejected fraction is printed")
    parser.add_argument("--line-sampling", choices=["gaussian", "delta"], default=None,
                        help="override the checkpoint's line_sampling: 'delta' emits "
                             "line-routed events at exactly the line energy")

    args = parser.parse_args()

    metadata = load_json(args.metadata_file)
    preprocessing = metadata["preprocessing_metadata"]
    model_config = metadata["model_config"]
    crossing_rule = preprocessing.get("crossing_rule")
    if crossing_rule == "first_entry":
        print("[MAGI] crossing_rule = first_entry: this model was trained on the first entries of the "
              "tracking sphere (Track G). Inject the sample into a run that simulates the whole interior "
              "of the sphere, at the same radius (see USAGE.md, 'first-entry crossings').", flush=True)
    if args.line_sampling is not None:
        model_config["line_sampling"] = args.line_sampling

    idx_to_type = preprocessing["idx_to_type"]
    if isinstance(idx_to_type, dict):
        idx_to_type = {int(k): v for k, v in idx_to_type.items()}

    # Only categorical-energy models (v0.6/v0.7/v0.7.2) need the bins.
    energy_bins = preprocessing.get("energy_bins", None)
    if energy_bins is not None:
        energy_bins = np.asarray(energy_bins, dtype=np.float64)

    type_probs = np.asarray(preprocessing["type_probs"], dtype=np.float64)

    geometry_transform = preprocessing.get(
        "geometry_transform",
        preprocessing.get("geometry_metadata", {}).get("geometry_transform", "legacy_sr_discrete_uv"),
    )

    geometry_metadata = dict(preprocessing.get("geometry_metadata", {}))

    model_class = model_config.get(
        "model_class",
        model_config.get("class_name", "CVAE_CatEnergy_CatUV_TaskAdaptive"),
    )

    energy_transform = preprocessing.get(
        "energy_transform",
        metadata.get("training_metadata", {}).get("energy_transform", None),
    )

    # v0.8 mixture head: energy is a continuous target, not a bin index.
    energy_head_mode = (
        "mixture" if model_class.startswith("CVAE_MixEnergy") else None
    )

    transformers_file = args.transformers_file

    if transformers_file is None:
        transformers_file = os.path.join(
            args.save_dir,
            f"{args.model_name}_quantile_transformers.joblib",
        )

    quantile_geometry = geometry_transform in [
        "quantile_u_r_u_v",
        "quantile_u_r_u_v_phi_r_phi_v",
        "quantile_u_r_mu_phi_r_psi",
    ]
    quantile_energy = (energy_transform == "quantile")

    quantile_transformers = {}

    if quantile_geometry or quantile_energy:
        if not os.path.exists(transformers_file):
            raise FileNotFoundError(
                "This model uses quantile transforms "
                f"(geometry={geometry_transform}, energy={energy_transform}), "
                "but the QuantileTransformer file was not found: "
                f"{transformers_file}"
            )

        quantile_transformers = joblib.load(transformers_file)
        print(f"[MAGI] Loaded quantile transformers: {transformers_file}")

    # ------------------------------------------------------
    # v0.7 / v0.7.2 quantile geometry
    # ------------------------------------------------------
    if quantile_geometry:
        geometry_metadata.update(quantile_transformers)
        geometry_metadata["geometry_transform"] = geometry_transform

    # ------------------------------------------------------
    # v0.8 quantile energy
    # ------------------------------------------------------
    qt_energy = quantile_transformers.get("qt_energy", None)

    # ------------------------------------------------------
    # Legacy v0.6 compatibility
    # ------------------------------------------------------
    u_v_bins = None
    s_r_mean = None
    s_r_std = None

    if "u_v_bins" in preprocessing:
        u_v_bins = np.asarray(preprocessing["u_v_bins"], dtype=np.float64)

    if "s_r_mean" in preprocessing:
        s_r_mean = float(preprocessing["s_r_mean"])

    if "s_r_std" in preprocessing:
        s_r_std = float(preprocessing["s_r_std"])

    # The sphere is what maps generated (u_r, phi_r) back to physical (x, y, z).
    # Getting it wrong does not fail - it silently emits particles on the wrong
    # surface, which Geant4 then transports from the wrong place. That cost a
    # full DM1.2 run: its sphere is centre (0,0,5) R=105, but the metadata used
    # the key names `sphere_center`/`sphere_radius`, so these lookups missed and
    # fell back to the SRON default, putting every particle 512 mm away on a
    # 100 mm sphere. Warn loudly whenever the fallback is used rather than
    # raising, because 30 of the existing SRON checkpoints omit these keys too
    # and are only correct by coincidence.
    DEFAULT_RADIUS = 100.0
    DEFAULT_CENTER = (0.0, 0.0, -507.66)

    radius_src = "model_config['sphere_R']"
    radius = model_config.get("sphere_R")
    if radius is None:
        radius = preprocessing.get("radius")
        radius_src = "preprocessing['radius']"
    if radius is None:
        radius = preprocessing.get("sphere_radius")
        radius_src = "preprocessing['sphere_radius']"
    if radius is None:
        radius, radius_src = DEFAULT_RADIUS, "DEFAULT"
    radius = float(radius)

    center_src = "preprocessing['center']"
    center = preprocessing.get("center")
    if center is None:
        center = preprocessing.get("sphere_center")
        center_src = "preprocessing['sphere_center']"
    if center is None:
        center, center_src = DEFAULT_CENTER, "DEFAULT"
    center = tuple(center)

    print(f"[MAGI] CryoSphere for reconstruction: centre {center} radius {radius} "
          f"(centre from {center_src}, radius from {radius_src})")
    if center_src == "DEFAULT" or radius_src == "DEFAULT":
        print("[MAGI] *** WARNING: falling back to the SRON CryoSphere "
              "(0, 0, -507.66) R=100. If this model was trained on any other "
              "geometry - e.g. DM1.2, which is (0, 0, 5) R=105 - every particle "
              "will be emitted on the WRONG surface and the run is invalid. Set "
              "preprocessing_metadata['center'] and ['radius'] at training time.",
              flush=True)

    if args.max_energy_mev is not None and args.n_primaries is not None:
        parser.error("--max-energy-mev is implemented for --n-events mode only")
    if args.latent_bank is not None and args.n_primaries is None:
        parser.error("--latent-bank is implemented for group mode (--n-primaries) only")
    if args.n_primaries is not None:
        generate_groups(args, model_config, preprocessing, idx_to_type,
                        quantile_transformers, geometry_metadata, energy_bins,
                        energy_head_mode, energy_transform, qt_energy,
                        center, radius)
        return

    magi.generate_detector_input_file(
        save_dir=args.save_dir,
        model_name=args.model_name,
        model_config=model_config,
        energy_bins=energy_bins,
        geometry_metadata=geometry_metadata,
        u_v_bins=u_v_bins,
        n_types=int(preprocessing["n_types"]),
        type_weights=preprocessing.get("type_weights", None),
        type_probs=type_probs,
        idx_to_type=idx_to_type,
        s_r_mean=s_r_mean,
        s_r_std=s_r_std,
        energy_head_mode=energy_head_mode,
        energy_transform=energy_transform,
        qt_energy=qt_energy,
        output_file=args.output_file,
        n_events=args.n_events,
        radius=radius,
        center=center,
        seed=args.seed,
        chunk_size=args.chunk_size,
        output_format=args.format,
        max_energy_mev=args.max_energy_mev,
        verbose=1,
    )


if __name__ == "__main__":
    main()