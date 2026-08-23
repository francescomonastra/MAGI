"""compute_full_circuit_trace / render_full_circuit_html.

Uses a tiny CVAE_MixEnergy_ContPhi_TaskAdaptive built directly (no real
checkpoint or training data) with encoder/trunk/energy-branch depths that
deliberately differ from both real checkpoints in this project (3/3/2), to
prove the stage layout generalizes rather than being hardcoded to that one
shape. Structural checks only - no browser, no rendering.
"""
import json
import re

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
        gate_focal_gamma=0.0,  # pinned: was the default before v0.8.2 flipped it
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
    event per type so the zone-stratified sampler has something to find."""
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
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES,
        n_events_per_type=4, n_baselines=3, n_steps=4)

    assert set(trace) == set(IDX_TO_TYPE.values()) | {"__all__"}
    stages = trace["gamma"]["stages"]
    expected = (
        ["enc1", "enc2", "z", "stem", "trunk1", "trunk2", "trunk3", "trunk4", "eb1"]
        + [f"pos{i + 1}" for i in range(n_pos)]
        + [f"dir{i + 1}" for i in range(n_dir)]
        + ["gate"]
    )
    # order must be the actual data-flow order, not alphabetical
    assert list(stages) == expected
    # the pooled entry has the same stage set
    assert list(trace["__all__"]["stages"]) == expected


def test_usage_values_present_finite_and_nonnegative():
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}

    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES,
        n_events_per_type=4, n_baselines=3, n_steps=4)

    for name, rec in trace.items():
        assert rec["n_sampled"] > 0
        assert rec["n_baselines"] > 0
        for stage in rec["stages"].values():
            assert len(stage["usage"]) == stage["width"]
            usage = np.asarray(stage["usage"])
            assert np.all(np.isfinite(usage))
            # abs-then-mean, so usage can never be negative
            assert np.all(usage >= 0)

    assert trace["__all__"]["n_types_pooled"] == N_TYPES
    # pooling reuses every type's already-sampled events, no extra sampling
    assert trace["__all__"]["n_sampled"] == sum(
        trace[t]["n_sampled"] for t in IDX_TO_TYPE.values()
    )


def test_trapezoidal_conductance_formula_satisfies_completeness():
    """Validates the exact trapezoidal-Riemann-sum recipe
    `_expected_conductance_for_type` uses (interpolate a path, take
    gradient x activation-delta at consecutive steps, average consecutive
    gradients, sum) against a toy smooth function where the Integrated
    Gradients completeness axiom is easy to check independently: for
    `out = sum(h**2)`, summed conductance along a straight-line path in `h`
    must equal `out(target) - out(baseline)`.

    This is checked against a toy function rather than the real model's own
    reconstruction loss because that loss runs through `ConditionalRQSFlow`
    log_prob, whose gradient w.r.t. its *value* argument (not the
    conditioning it's differentiated against everywhere else in this file)
    is not reliably defined for arbitrary off-manifold synthetic inputs -
    `_expected_conductance_for_type` never needs that gradient in practice
    (it only ever differentiates w.r.t. internal activations, which the
    other tests in this file exercise directly against the real model and
    confirm are always finite), so this test isolates and validates the
    Riemann-sum bookkeeping itself instead."""
    rng = np.random.default_rng(0)
    D = 5
    target_h = rng.normal(size=(1, D)).astype(np.float32)
    baseline_h = rng.normal(size=(1, D)).astype(np.float32)

    K = 200
    alphas = np.linspace(0.0, 1.0, K + 1, dtype=np.float32)
    interp = (
        baseline_h[None, :, :] + alphas[:, None, None] * (target_h[None, :, :] - baseline_h[None, :, :])
    ).astype(np.float32).reshape(K + 1, D)

    h = tf.constant(interp)
    with tf.GradientTape() as tape:
        tape.watch(h)
        out = tf.reduce_sum(h * h, axis=1)
        total = tf.reduce_sum(out)
    grad = tape.gradient(total, h).numpy()

    grad_trap = 0.5 * (grad[:-1] + grad[1:])
    h_diff = interp[1:] - interp[:-1]
    conductance = float(np.sum(grad_trap * h_diff))

    delta = float(out[-1].numpy() - out[0].numpy())
    assert conductance == pytest.approx(delta, abs=1e-3, rel=1e-3)


def test_usage_is_not_degenerately_all_zero():
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}

    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES,
        n_events_per_type=4, n_baselines=3, n_steps=4)

    for rec in trace.values():
        total = sum(sum(stage["usage"]) for stage in rec["stages"].values())
        assert total > 0.0


def test_handles_type_with_very_few_held_out_events():
    model = _build_tiny_model()
    # only 3 held-out events per type - fewer than the default n_baselines=16
    X_cont_test, y_type_test = _synthetic_test_split(n_per_type=3)
    model_config = {"hidden": HIDDEN}

    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES,
        n_events_per_type=128, n_baselines=16, n_steps=4)

    for name in IDX_TO_TYPE.values():
        rec = trace[name]
        assert 0 < rec["n_sampled"] <= 3
        assert 0 < rec["n_baselines"] <= 3
        for stage in rec["stages"].values():
            assert np.all(np.isfinite(stage["usage"]))


def test_render_full_circuit_html_is_self_contained_with_all_types_button():
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}
    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES,
        n_events_per_type=4, n_baselines=3, n_steps=4)

    zone_labels = ["continuum", "e+e- annihilation"]
    lines = [{"label": "e+e- annihilation", "energy_mev": 0.511}]
    spectrum = _synthetic_spectrum()
    type_order = list(IDX_TO_TYPE.values())

    doc = magi.render_full_circuit_html(
        trace, zone_labels, lines, spectrum, type_order, source_label="unit-test")

    assert "<style>" in doc and "<svg" in doc and "<script>" in doc
    # No external dependency - the only allowed "http://" is the mandatory
    # SVG namespace URI (as an xmlns attribute), not a fetch/network reference.
    non_namespace_http = doc.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in non_namespace_http and "https://" not in doc

    for name in type_order:
        assert f'renderType(\'{name}\')' in doc
    assert 'renderType(\'__all__\')' in doc
    assert ">All types<" in doc

    # every stage from the trace has real unit circles in the SVG
    for stage_name, stage in trace["gamma"]["stages"].items():
        assert f'id="u-{stage_name}-0"' in doc
        assert f'id="u-{stage_name}-{stage["width"] - 1}"' in doc

    start = doc.index("const DATA = ") + len("const DATA = ")
    end = doc.index(";\n", start)
    data = json.loads(doc[start:end])
    assert data["zone_labels"] == zone_labels
    assert set(data["types"]) == set(type_order) | {"__all__"}
    assert data["spectrum"]["combined"] == spectrum["combined"]

    # the percentile-slider mechanism is gone entirely
    assert "pct-slider" not in doc
    assert "setPct" not in doc
    assert "topKSet" not in doc

    # ...but per-neuron wires between consecutive stages are drawn, colored
    # per type from the same usage scale
    assert 'id="connectors"' in doc
    assert "function drawWire(" in doc
    assert "function topUnits(" in doc
    assert "LAYOUT.edges" in doc
    start_l = doc.index("const LAYOUT = ") + len("const LAYOUT = ")
    end_l = doc.index(";\n", start_l)
    layout = json.loads(doc[start_l:end_l])
    stage_names = set(trace["gamma"]["stages"])
    assert layout["edges"], "no stage-to-stage edges emitted, so no wires can be drawn"
    for a, b in layout["edges"]:
        assert a in stage_names and b in stage_names


def test_unit_fill_is_not_overridden_by_css():
    """renderType() colors each unit by setting the `fill` presentation
    attribute. A `fill` declaration in the `.unit` CSS rule would silently
    win over it (CSS beats presentation attributes in SVG) and every unit
    would render one flat color regardless of its usage - which is exactly
    the bug that made only the spectrum appear to respond to the type
    buttons."""
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}
    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES,
        n_events_per_type=4, n_baselines=3, n_steps=4)

    doc = magi.render_full_circuit_html(
        trace, ["continuum", "e+e- annihilation"],
        [{"label": "e+e- annihilation", "energy_mev": 0.511}],
        _synthetic_spectrum(), list(IDX_TO_TYPE.values()), source_label="unit-test")

    rule = re.search(r"#circuit-svg \.unit \{([^}]*)\}", doc)
    assert rule is not None, ".unit CSS rule not found"
    assert "fill" not in rule.group(1), (
        f"`.unit` CSS rule declares fill ({rule.group(1).strip()!r}); it would "
        "override the per-unit color renderType() sets as an attribute"
    )
    # and the circles still carry an initial fill so nothing flashes unstyled
    assert 'class="unit" fill=' in doc


def test_absolute_magnitude_panel_reports_real_unnormalized_stats():
    """The rank-based node color cannot answer 'is this layer too wide' - the
    absolute panel carries the raw conductance magnitudes that can."""
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}
    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES,
        n_events_per_type=4, n_baselines=3, n_steps=4)

    doc = magi.render_full_circuit_html(
        trace, ["continuum", "e+e- annihilation"],
        [{"label": "e+e- annihilation", "energy_mev": 0.511}],
        _synthetic_spectrum(), list(IDX_TO_TYPE.values()), source_label="unit-test")

    assert 'id="abs-table"' in doc and "function renderAbsTable(" in doc

    start = doc.index("const DATA = ") + len("const DATA = ")
    data = json.loads(doc[start:doc.index(";\n", start)])
    assert set(data["abs_stats"]) == set(data["types"])

    for tname, rows in data["abs_stats"].items():
        stages = trace[tname]["stages"]
        assert [r["stage"] for r in rows] == list(stages)
        assert sum(r["share"] for r in rows) == pytest.approx(1.0, abs=1e-6)
        # load is share-of-work over share-of-width, so weighting each layer's
        # load by its width has to come back to exactly 1.0
        total_units = sum(r["width"] for r in rows)
        assert sum(
            r["load"] * r["width"] / total_units for r in rows
        ) == pytest.approx(1.0, abs=1e-6)
        for r in rows:
            assert r["width"] == stages[r["stage"]]["width"]
            # totals are raw magnitudes, not renormalized to any fixed range
            assert r["total"] == pytest.approx(
                sum(abs(v) for v in stages[r["stage"]]["usage"]), rel=1e-6)
            assert r["mean"] == pytest.approx(r["total"] / r["width"], rel=1e-6)
            # the busiest fifth can never carry less than a proportional share
            assert 0.2 - 1e-9 <= r["top20"] <= 1.0
            assert 0.0 <= r["faint"] <= 1.0


def test_usage_colors_spread_across_the_full_scale():
    """The color scale must actually use its range. Usage is heavy-tailed, so
    a naive linear (or log) min-max leaves ~75% of units in magma's near-black
    lower third and the diagram reads as uniformly dark - percentile-rank
    normalization is what keeps it legible."""
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}
    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES,
        n_events_per_type=4, n_baselines=3, n_steps=4)

    doc = magi.render_full_circuit_html(
        trace, ["continuum", "e+e- annihilation"],
        [{"label": "e+e- annihilation", "energy_mev": 0.511}],
        _synthetic_spectrum(), list(IDX_TO_TYPE.values()), source_label="unit-test")

    start = doc.index("const DATA = ") + len("const DATA = ")
    data = json.loads(doc[start:doc.index(";\n", start)])
    pooled = [
        v for rec in data["types"].values()
        for st in rec["stages"].values() for v in st["usage100"]
    ]
    pooled.sort()
    median = pooled[len(pooled) // 2]
    # a percentile-ranked pooled distribution sits near the middle of the
    # scale, not bunched at the dark end
    assert 30 <= median <= 70, f"median color position {median} is off-center"
    assert max(pooled) >= 90 and min(pooled) <= 10


def test_render_rejects_empty_or_mismatched_type_order():
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}
    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES,
        n_events_per_type=4, n_baselines=3, n_steps=4)
    zone_labels = ["continuum", "e+e- annihilation"]
    lines = [{"label": "e+e- annihilation", "energy_mev": 0.511}]
    spectrum = _synthetic_spectrum()

    with pytest.raises(ValueError):
        magi.render_full_circuit_html(trace, zone_labels, lines, spectrum, [])
    with pytest.raises(ValueError):
        magi.render_full_circuit_html(trace, zone_labels, lines, spectrum, ["not-a-type"])


def test_render_rejects_types_data_missing_all_types_entry():
    model = _build_tiny_model()
    X_cont_test, y_type_test = _synthetic_test_split()
    model_config = {"hidden": HIDDEN}
    trace = magi.compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, IDX_TO_TYPE, n_zones=N_ZONES,
        n_events_per_type=4, n_baselines=3, n_steps=4)
    trace_without_all = {k: v for k, v in trace.items() if k != "__all__"}
    zone_labels = ["continuum", "e+e- annihilation"]
    lines = [{"label": "e+e- annihilation", "energy_mev": 0.511}]
    spectrum = _synthetic_spectrum()

    with pytest.raises(ValueError):
        magi.render_full_circuit_html(
            trace_without_all, zone_labels, lines, spectrum, list(IDX_TO_TYPE.values()))
