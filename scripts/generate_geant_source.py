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


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Geant4 input particle file using a trained MAGI model."
    )

    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--output-file", required=True)

    parser.add_argument("--transformers-file", default=None)

    parser.add_argument("--n-events", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--format", choices=["text", "binary"], default="binary")

    args = parser.parse_args()

    metadata = load_json(args.metadata_file)
    preprocessing = metadata["preprocessing_metadata"]
    model_config = metadata["model_config"]

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
        verbose=1,
    )


if __name__ == "__main__":
    main()