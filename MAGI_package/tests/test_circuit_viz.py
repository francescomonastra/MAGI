"""render_routing_circuit_html / save_routing_circuit.

Structural checks only - no browser, no rendering. A regression here means
the emitted HTML/JS/JSON is malformed, not that it "looks wrong".
"""
import json
import os

import numpy as np
import pytest

import magi


def _toy(n_types=3, n_lines=2):
    names = ["gamma", "e-", "mu-", "e+", "nu_mu"]
    idx_to_type = {i: names[i] for i in range(n_types)}
    type_probs = list(np.linspace(0.1, 1.0, n_types) / np.linspace(0.1, 1.0, n_types).sum())
    rng = np.random.default_rng(0)
    zone_probs = rng.dirichlet(np.ones(n_lines + 1), size=n_types)
    zone_labels = ["continuum"] + [f"line {i}" for i in range(n_lines)]
    return idx_to_type, type_probs, zone_probs, zone_labels


def test_renders_self_contained_html():
    idx_to_type, type_probs, zone_probs, zone_labels = _toy()
    doc = magi.render_routing_circuit_html(idx_to_type, type_probs, zone_probs, zone_labels)

    assert doc.startswith("<!doctype html>")
    assert "<style>" in doc and "<svg" in doc and "<script>" in doc
    # No external dependency: nothing should try to fetch another origin.
    assert "http://" not in doc and "https://" not in doc
    for name in idx_to_type.values():
        assert name in doc


def test_embedded_data_is_valid_json_matching_input():
    idx_to_type, type_probs, zone_probs, zone_labels = _toy()
    doc = magi.render_routing_circuit_html(idx_to_type, type_probs, zone_probs, zone_labels)

    start = doc.index("const DATA = ") + len("const DATA = ")
    end = doc.index(";\n", start)
    data = json.loads(doc[start:end])

    assert data["types"] == list(idx_to_type.values())
    assert len(data["zones"]) == len(idx_to_type)
    for row, expected in zip(data["zones"], zone_probs):
        assert row == pytest.approx([float(x) * 100 for x in expected], abs=1e-6)


def test_wraps_many_zones_without_error():
    # 7 zones forces the leaf grid to wrap across rows (per_row=3) - the
    # geometry math is the part most likely to break for an unusual n_lines.
    idx_to_type, type_probs, zone_probs, zone_labels = _toy(n_types=2, n_lines=6)
    doc = magi.render_routing_circuit_html(idx_to_type, type_probs, zone_probs, zone_labels)
    assert doc.count('class="mc-purple"') >= 6 + 2  # 6 line leaves + 2 type cells... at least


def test_rejects_shape_mismatch():
    idx_to_type, type_probs, zone_probs, zone_labels = _toy()
    with pytest.raises(ValueError):
        magi.render_routing_circuit_html(idx_to_type, type_probs[:-1], zone_probs, zone_labels)
    with pytest.raises(ValueError):
        magi.render_routing_circuit_html(idx_to_type, type_probs, zone_probs, zone_labels[:-1])


def test_save_routing_circuit_requires_zone_probs(tmp_path):
    save_dir = str(tmp_path)
    config = {"zone_probs": None, "line_positions_y": [-1.0]}
    metadata = {
        "preprocessing_metadata": {
            "idx_to_type": {"0": "gamma", "1": "e-"},
            "type_probs": [0.9, 0.1],
        }
    }
    with open(os.path.join(save_dir, "m_config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(save_dir, "m_metadata.json"), "w") as f:
        json.dump(metadata, f)

    with pytest.raises(KeyError):
        magi.save_routing_circuit(save_dir, "m")


def test_save_routing_circuit_falls_back_to_pinned_energy(tmp_path):
    save_dir = str(tmp_path)
    config = {
        "zone_probs": [[0.9, 0.1], [0.99, 0.01]],
        "line_positions_y": [-0.29158],  # ~0.511 MeV
    }
    metadata = {
        "preprocessing_metadata": {
            "idx_to_type": {"0": "gamma", "1": "e-"},
            "type_probs": [0.9, 0.1],
            # no matched_lines key -> must fall back to the pinned energy
        }
    }
    with open(os.path.join(save_dir, "m_config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(save_dir, "m_metadata.json"), "w") as f:
        json.dump(metadata, f)

    path = magi.save_routing_circuit(save_dir, "m")
    assert os.path.exists(path)
    with open(path) as f:
        doc = f.read()
    assert "511" in doc  # recovered from line_positions_y, not a generic label
