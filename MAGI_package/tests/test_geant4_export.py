"""End-to-end check of the Geant4 integration path: a Geant4 macro's
/generator/mlScript invokes scripts/generate_geant_source.py as a subprocess
against a saved checkpoint's directory + metadata file. This builds a tiny
checkpoint the same way tools/run_v0_8_real.py does (model_config from
to_generation_config(), preprocessing_metadata with n_types/type_probs/
idx_to_type/geometry_transform/energy_transform, a sibling
*_quantile_transformers.joblib) and actually runs the script as a
subprocess, in both output formats.

This is a regression test for a real bug: tools/run_v0_8_real.py used to
call save_final_trained_model without preprocessing_metadata or the joblib
file, which left every v0.8/v0.8.1 checkpoint it produced unusable for
Geant4 export (KeyError on preprocessing_metadata["idx_to_type"] against the
empty dict save_final_trained_model defaults to, before that dict even gets
to the missing-joblib check). See tools/run_v0_8_real.py's checkpoint-save
comment for the fix.
"""
import json
import subprocess
import sys

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import QuantileTransformer

import magi

N_TYPES = 2
LINE_POSITIONS_Y = np.array([np.log10(0.511)], dtype=np.float32)
GENERATE_SCRIPT = str(
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "scripts" / "generate_geant_source.py"
)


def _build_and_save_tiny_checkpoint(save_dir):
    model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        n_types=N_TYPES, line_positions_y=LINE_POSITIONS_Y, latent_dim=4,
        hidden=(16, 16), beta=0.2, continuum_mode="flow",
        continuum_flow_bins=8, continuum_flow_transforms=2,
        continuum_flow_warp="affine",
        energy_flow_condition="z_cond", prior="coupling",
        prior_n_layers=2, prior_hidden=(8, 8),
        line_logsigma_init=np.array([-9.0], dtype=np.float32),
        line_logsigma_trainable=False,
    )
    n_lines = LINE_POSITIONS_Y.shape[0]
    y_cont_dim = 4 + 1 + (n_lines + 1)
    dummy_cond = tf.zeros((2, N_TYPES), dtype=tf.float32)
    _ = model.encoder(
        tf.concat([tf.zeros((2, y_cont_dim), dtype=tf.float32), dummy_cond], axis=1),
        training=False,
    )
    _ = model.decode(tf.zeros((2, model.latent_dim), dtype=tf.float32), dummy_cond)

    rng = np.random.default_rng(0)

    def _fit_qt(lo, hi):
        qt = QuantileTransformer(n_quantiles=50, output_distribution="normal", random_state=0)
        qt.fit(rng.uniform(lo, hi, size=(200, 1)))
        return qt

    quantile_transformers = {
        "qt_u_r": _fit_qt(-1, 1), "qt_u_v": _fit_qt(-1, 1),
        "qt_phi_r": _fit_qt(-np.pi, np.pi), "qt_phi_v": _fit_qt(-np.pi, np.pi),
    }
    preprocessing_metadata = {
        "source": "tiny",
        "geometry_transform": "quantile_u_r_u_v_phi_r_phi_v",
        "energy_transform": "log10",
        "type_probs": [0.5, 0.5],
        "idx_to_type": {0: "e-", 1: "gamma"},
        "n_types": N_TYPES,
        "radius": 100.0,
        "center": [0.0, 0.0, -507.66],
    }

    magi.save_final_trained_model(
        model=model, save_dir=save_dir, model_name="mix_tiny",
        model_config=model.to_generation_config(),
        preprocessing_metadata=preprocessing_metadata,
    )
    joblib.dump(quantile_transformers, f"{save_dir}/mix_tiny_quantile_transformers.joblib")


def _run_generate_geant_source(save_dir, output_file, fmt):
    result = subprocess.run(
        [
            sys.executable, GENERATE_SCRIPT,
            "--save-dir", str(save_dir),
            "--model-name", "mix_tiny",
            "--metadata-file", f"{save_dir}/mix_tiny_metadata.json",
            "--output-file", str(output_file),
            "--n-events", "50",
            "--format", fmt,
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"generate_geant_source.py failed (format={fmt}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result


def test_geant4_export_text_format(tmp_path):
    save_dir = tmp_path / "ckpt"
    save_dir.mkdir()
    _build_and_save_tiny_checkpoint(str(save_dir))

    out_file = tmp_path / "out.txt"
    _run_generate_geant_source(save_dir, out_file, "text")

    assert out_file.exists()
    lines = out_file.read_text().splitlines()
    assert len(lines) == 50
    # EventId, ParticleName, Energy, X, Y, Z, Vx, Vy, Vz (CLAUDE.md convention;
    # generate_geant_source.py's export path uses include_event_id=False).
    cols = lines[0].split()
    assert len(cols) == 8
    energy = float(cols[1])
    assert energy > 0


def test_geant4_export_binary_format(tmp_path):
    save_dir = tmp_path / "ckpt"
    save_dir.mkdir()
    _build_and_save_tiny_checkpoint(str(save_dir))

    out_file = tmp_path / "out.gntbin"
    _run_generate_geant_source(save_dir, out_file, "binary")

    assert out_file.exists()
    assert out_file.stat().st_size > 0
