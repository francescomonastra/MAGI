#!/usr/bin/env python

import argparse
import json
import numpy as np
import joblib
import GEEANNT as ge
import os


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Geant4 input particle file using a trained GEEANNT model."
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

    energy_bins = np.asarray(preprocessing["energy_bins"], dtype=np.float64)
    type_probs = np.asarray(preprocessing["type_probs"], dtype=np.float64)

    geometry_transform = preprocessing.get(
        "geometry_transform",
        preprocessing.get("geometry_metadata", {}).get("geometry_transform", "legacy_sr_discrete_uv"),
    )

    geometry_metadata = dict(preprocessing.get("geometry_metadata", {}))

    # ------------------------------------------------------
    # v0.7 / v0.7.2 quantile transformers
    # ------------------------------------------------------
    if geometry_transform in [
        "quantile_u_r_u_v",
        "quantile_u_r_u_v_phi_r_phi_v",
    ]:
        transformers_file = args.transformers_file

        if transformers_file is None:
            transformers_file = os.path.join(
                args.save_dir,
                f"{args.model_name}_quantile_transformers.joblib",
            )

        if not os.path.exists(transformers_file):
            raise FileNotFoundError(
                "This model uses quantile geometry, but the QuantileTransformer "
                f"file was not found: {transformers_file}"
            )

        quantile_transformers = joblib.load(transformers_file)
        geometry_metadata.update(quantile_transformers)
        geometry_metadata["geometry_transform"] = geometry_transform

        print(f"[GEEANNT] Loaded quantile transformers: {transformers_file}")

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

    radius = float(
        model_config.get(
            "sphere_R",
            preprocessing.get("radius", 100.0),
        )
    )

    center = tuple(
        preprocessing.get(
            "center",
            (0.0, 0.0, -507.66),
        )
    )

    ge.generate_detector_input_file(
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