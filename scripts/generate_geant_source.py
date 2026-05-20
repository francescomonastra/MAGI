#!/usr/bin/env python

import argparse
import json
import numpy as np
import GEEANNT as ge


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

    parser.add_argument("--n-events", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100000)

    args = parser.parse_args()

    metadata = load_json(args.metadata_file)
    preprocessing = metadata["preprocessing_metadata"]
    idx_to_type = preprocessing["idx_to_type"]

    if isinstance(idx_to_type, dict):
        idx_to_type = {int(k): v for k, v in idx_to_type.items()}

    model_config = metadata["model_config"]

    ge.generate_detector_input_file(
        save_dir=args.save_dir,
        model_name=args.model_name,
        model_config=model_config,
        energy_bins=np.asarray(preprocessing["energy_bins"]),
        u_v_bins=np.asarray(preprocessing["u_v_bins"]),
        n_types=int(preprocessing["n_types"]),
        type_weights=preprocessing.get("type_weights", None),
        type_probs=np.asarray(preprocessing["type_probs"]),
        idx_to_type=idx_to_type,
        s_r_mean=float(preprocessing["s_r_mean"]),
        s_r_std=float(preprocessing["s_r_std"]),
        output_file=args.output_file,
        n_events=args.n_events,
        radius=float(model_config["sphere_R"]),
        center=(0.0, 0.0, -507.66),
        seed=args.seed,
        chunk_size=args.chunk_size,
        verbose=1,
    )


if __name__ == "__main__":
    main()