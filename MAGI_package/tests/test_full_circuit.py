"""compute_full_circuit_trace / render_full_circuit_html.

Uses a tiny CVAE_MixEnergy_ContPhi_TaskAdaptive built directly (no real
checkpoint or training data) with encoder/trunk/energy-branch depths that
deliberately differ from both real checkpoints in this project (3/3/2), to
prove the stage layout generalizes rather than being hardcoded to that one
shape. Structural checks only - no browser, no rendering.
"""
import json

import numpy as np
import pytest
import tensorflow as tf

import magi

N_TYPES = 2
LINE_POSITIONS_Y = np.array([np.log10(0.511)], dtype=np.float32)
N_LINES = LINE_POSITIONS_Y.shape[0]
N_ZONES = N_LINES + 1  # [continuum, line]
IDX_TO_TYPE = {0: "gamma", 1: "e-"}

# Deliberately unlike the real CR (3/3/2) and Small (3/3/2) checkpoints -
# 2 encoder layers, 4 trunk layers, 1 energy-branch layer.
HIDDEN = (12, 10)
DEEP_DECODER_HIDDEN = (10, 10, 10, 10)
ENERGY_BRANCH_HIDDEN = (6,)


def _build_tiny_model():
    model = magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        n_types=N_TYPES, line_positions_y=LINE_POSITIONS_Y, latent_dim=4,
        hidden=HIDDEN, beta=0.2, continuum_mode="flow",
        continuum_flow_bins=8, continuum_flow_transforms=2,
        continuum_flow_warp="affine",
        energy_flow_condition="z_cond", prior="coupling",
        prior_n_layers=2, prior_hidden=(8, 8),
        line_logsigma_init=np.array([-9.0], dtype=np.float32),
        line_logsigma_trainable=False,
        deep_decoder_hidden=DEEP_DECODER_HIDDEN,
        energy_branch_hidden=ENERGY_BRANCH_HIDDEN,
    )
    y_cont_dim = 4 + 1 + N_ZONES
    dummy_cond = tf.zeros((2, N_TYPES), dtype=tf.float32)
    _ = model.encoder(
        tf.concat([tf.zeros((2, y_cont_dim), dtype=tf.float32), dummy_cond], axis=1),
        training=False,
    )
    _ = model.decode(tf.zeros((2, model.latent_dim), dtype=tf.float32), dummy_cond)
    return model


def _synthetic_test_split(n_per_type=6, seed=0):
    """X_cont_test in the [ur_q,uv_q,phi_r_q,phi_v_q,energy_y,
    gate_target_0..N_ZONES-1, energy_mev_physical] layout
    compute_full_circuit_trace expects, with at least one genuine line
    event per type so the "prefer a line hit" pick path is exercised."""
    rng = np.random.default_rng(seed)
    rows, types = [], []
    for t in range(N_TYPES):
        for i in range(n_per_type):
            geom = rng.normal(size=4).astype(np.float32)
            energy_y = rng.normal()
            zone = 0 if i % 3 != 0 else 1  # every third event is a genuine line hit
            gate = np.eye(N_ZONES, dtype=np.float32)[zone]
            e_phys = 0.511 if zone == 1 else float(10 ** rng.uniform(-2, 1))
            rows.append(np.concatenate([geom, [energy_y], gate, [e_phys]]))
            types.append(t)
    return np.asarray(rows, dtype=np.float32), np.asarray(types, dtype=np.int32)


def _synthetic_spectrum():
    edges = np.linspace(-3.0, 3.0, 21)
    rng = np.random.default_rng(1)
    combined = rng.integers(1, 500, size=20).tolist()
    return {
        "log_edges": edges.tolist(),
        "combined": combined,
        "by_type": {name: [max(0, c // 2) for c in combined] for name in IDX_TO_TYPE.values()},
    }


def _n_hidden_layers(container):
    # Dense+LeakyReLU pairs -> only the post-activation half counts as a stage.
    return len([l for l in container.layers if l.__class__.__name__ != "InputLayer"]) // 2


def test_stage_layout_matches_actual_model_depths():
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}
    n_pos = _n_hidden_layers(model.position_branch)
    n_dir = _n_hidden_layers(model.direction_branch)

    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES)

    assert set(trace) == set(IDX_TO_TYPE.values())
    stages = trace["gamma"]["stages"]
    expected = (
        ["enc1", "enc2", "z", "stem", "trunk1", "trunk2", "trunk3", "trunk4", "eb1"]
        + [f"pos{i + 1}" for i in range(n_pos)]
        + [f"dir{i + 1}" for i in range(n_dir)]
    )
    # order must be the actual data-flow order, not alphabetical
    assert list(stages) == expected


def test_rank_is_a_valid_permutation_and_gate_probs_sum_to_one():
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}

    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES)

    for rec in trace.values():
        for stage in rec["stages"].values():
            assert sorted(stage["rank"]) == list(range(stage["width"]))
            assert len(stage["activation"]) == stage["width"]
        assert sum(rec["gate_probs"]) == pytest.approx(1.0, abs=1e-4)
        assert 0 <= rec["true_zone"] < N_ZONES
        assert 0 <= rec["pred_zone"] < N_ZONES


def test_prefers_a_genuine_line_event_when_one_exists():
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}

    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES)

    # every type in the synthetic split has a line event (every 3rd row) -
    # the picked event must be one of them, not an arbitrary continuum row.
    for rec in trace.values():
        assert rec["true_zone"] == 1
        assert rec["energy_mev_true"] == pytest.approx(0.511, abs=1e-6)


def test_render_full_circuit_html_is_self_contained_and_matches_input():
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}
    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES)

    zone_labels = ["continuum", "e+e- annihilation"]
    lines = [{"label": "e+e- annihilation", "energy_mev": 0.511}]
    spectrum = _synthetic_spectrum()
    type_order = list(IDX_TO_TYPE.values())

    doc = magi.render_full_circuit_html(
        trace, zone_labels, lines, spectrum, type_order, source_label="unit-test")

    assert "<style>" in doc and "<svg" in doc and "<script>" in doc
    # No external dependency - the only allowed "http://" is the mandatory
    # SVG namespace URI (as an xmlns attribute and in createElementNS), not
    # a fetch/script-src/network reference.
    non_namespace_http = doc.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in non_namespace_http and "https://" not in doc
    for name in type_order:
        assert f'renderType(\'{name}\')' in doc
    # every stage from the trace has real unit circles in the SVG
    for stage_name, stage in trace["gamma"]["stages"].items():
        assert f'id="u-{stage_name}-0"' in doc
        assert f'id="u-{stage_name}-{stage["width"] - 1}"' in doc
    assert 'id="u-gate-0"' in doc and f'id="u-gate-{N_ZONES - 1}"' in doc

    start = doc.index("const DATA = ") + len("const DATA = ")
    end = doc.index(";\n", start)
    data = json.loads(doc[start:end])
    assert data["zone_labels"] == zone_labels
    assert set(data["types"]) == set(type_order)
    assert data["spectrum"]["combined"] == spectrum["combined"]


def test_render_rejects_empty_or_mismatched_type_order():
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}
    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES)
    zone_labels = ["continuum", "e+e- annihilation"]
    lines = [{"label": "e+e- annihilation", "energy_mev": 0.511}]
    spectrum = _synthetic_spectrum()

    with pytest.raises(ValueError):
        magi.render_full_circuit_html(trace, zone_labels, lines, spectrum, [])
    with pytest.raises(ValueError):
        magi.render_full_circuit_html(trace, zone_labels, lines, spectrum, ["not-a-type"])


def test_percentile_slider_range_present_and_wired():
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}
    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES)
    zone_labels = ["continuum", "e+e- annihilation"]
    lines = [{"label": "e+e- annihilation", "energy_mev": 0.511}]
    spectrum = _synthetic_spectrum()

    doc = magi.render_full_circuit_html(trace, zone_labels, lines, spectrum, list(IDX_TO_TYPE.values()))
    assert 'id="pct-slider"' in doc
    assert "function setPct(" in doc
    assert "function toggleTheme(" in doc
