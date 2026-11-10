"""HENABICS primary truth, data and generation side: the 16-column table
loads with its join key, run summaries sum, the multiplicity table counts
k = 0 from the thrown histogram, the grouped split never splits a shower,
GNTBIN v2 round-trips at 52 bytes/record, the GPS sampler reproduces
Geant4's user-histogram draw, and grouped generation writes contiguous,
correctly tagged groups plus a normalization sidecar, also when one
multi-species GPS macro fires several primary species.
"""
import json

import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing
import tensorflow as tf

import magi
from magi.generation.export import (
    BINARY_RECORD_V2_DTYPE,
    MAGI_BINARY_VERSION_PRIMTRUTH,
    _append_primary_truth_binary_chunk,
    _rewrite_binary_header,
    _write_binary_header,
)

TYPES = ["mu-", "e-", "gamma"]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}


def _crossing_rows():
    # EventId name E PrimBool id parent creator x y z vx vy vz ptid ppdg pE
    rows = []
    # event 10: mu primary, 1000 MeV, k=3
    for name in ("mu-", "e-", "gamma"):
        rows.append((10, name, 5.0, 0, 2, 1, "eIoni", 1, 2, 3, 0, 0, -1, 1, 13, 1000.0))
    # event 11: mu primary, 2 MeV, k=1
    rows.append((11, "mu-", 1.5, 1, 1, 0, "Primary", 1, 2, 3, 0, 0, -1, 1, 13, 2.0))
    # event 3: gamma primary, 1.13 MeV, k=1
    rows.append((3, "gamma", 1.13, 1, 1, 0, "Primary", 1, 2, 3, 0, 0, -1, 1, 22, 1.13))
    return rows


def _write_crossings(path):
    with open(path, "w") as f:
        for r in _crossing_rows():
            f.write("\t".join(str(v) for v in r) + "\n")


def _run_summary(n_mu_low, n_mu_high, n_gamma):
    # 200 bins of 0.05 dex on [-3, 7) plus under/overflow, as the RunAction writes.
    def hist(pairs):
        h = [0] * 202
        for log10e, n in pairs:
            h[1 + int(np.floor((log10e + 3) / 0.05))] += n
        return h
    return {
        "n_events": n_mu_low + n_mu_high + n_gamma,
        "log10E_bins": {"min": -3, "max": 7, "n": 200, "layout": "..."},
        "thrown": {"13": hist([(np.log10(2.0), n_mu_low), (3.0, n_mu_high)]),
                   "22": hist([(np.log10(1.13), n_gamma)])},
        "recorded": {"13": hist([(np.log10(2.0), 1), (3.0, 1)]),
                     "22": hist([(np.log10(1.13), 1)])},
        "job_tag": "t", "source_macro": "m", "seeds": [1, 2],
    }


def test_primary_truth_table_loads_with_join_key(tmp_path):
    p = tmp_path / "cross.dat"
    _write_crossings(p)
    df = magi.load_detector_table(str(p))
    assert list(df.columns) == magi.data.DEFAULT_COLUMNS_DET_PRIMTRUTH
    assert df["EventId"].dtype == np.int64 and df["PrimaryPDG"].dtype == np.int64
    # explicit request still wins
    assert "EventId" not in magi.load_detector_table(str(p), drop_event_id=True).columns


def test_run_summaries_sum_across_jobs(tmp_path):
    for j in range(2):
        with open(tmp_path / f"job{j}.json", "w") as f:
            json.dump(_run_summary(10, 5, 7), f)
    s = magi.load_run_summaries(str(tmp_path))
    assert s["n_events"] == 44
    assert s["thrown"][13].sum() == 30 and s["thrown"][22].sum() == 14
    assert len(s["log10E_edges"]) == 201


def test_multiplicity_table_counts_k0_from_thrown(tmp_path):
    p = tmp_path / "cross.dat"
    _write_crossings(p)
    df = magi.load_detector_table(str(p))
    summ = tmp_path / "s.json"
    with open(summ, "w") as f:
        json.dump(_run_summary(100, 50, 40), f)
    table = magi.build_multiplicity_table(df, magi.load_run_summaries(str(summ)), TYPE_TO_IDX)

    mu = table["species"]["13"]
    k = np.array(mu["k_counts"])
    assert k.sum() == 150 == mu["n_thrown"]
    assert k[:, 3].sum() == 1 and k[:, 1].sum() == 1 and k[:, 0].sum() == 148
    # k=3 shower at 1000 MeV holds one of each type, tabulated under k class 3
    tc = np.array(mu["type_counts"])
    b = int(np.searchsorted(table["log10E_edges"], 3.0, side="right") - 1)
    np.testing.assert_array_equal(tc[b, 2], [1, 1, 1])

    rng = np.random.default_rng(0)
    kk, types = magi.sample_multiplicity(table, 13, np.full(200_000, 3.0), rng)
    assert set(np.unique(kk)) <= {0, 3}
    assert abs(np.mean(kk == 3) - 1 / 50) < 0.002
    assert len(types) == kk.sum()
    # every k=3 group draws its types from the k-class-3 cell: all 3 types present
    np.testing.assert_allclose(np.bincount(types, minlength=3) / len(types), 1 / 3, atol=0.03)


def test_grouped_split_keeps_showers_together():
    n_groups = 400
    rng = np.random.default_rng(1)
    k = rng.integers(1, 5, size=n_groups)
    event = np.repeat(np.arange(n_groups), k)
    n = len(event)
    feat = pd.DataFrame({
        "EventId": event, "PrimaryPDG": np.where(event % 2, 13, 22),
        "PrimaryEnergy": 10 ** rng.uniform(0, 3, size=n),
    })
    pack = {"feat": feat, "X_cont_raw": rng.normal(size=(n, 4)).astype(np.float32),
            "E_idx": np.zeros(n, np.int32), "y_type": rng.integers(0, 3, size=n).astype(np.int32),
            "dataset_mode": "continuous_geometry"}
    pack = magi.build_primary_conditioning(pack)
    assert pack["primary_cond"].shape == (n, 3)
    assert pack["primary_type_to_idx"] == {13: 0, 22: 1}

    split = magi.split_feature_data(pack)
    sets = [set(split[f"group_id_{s}"]) for s in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(len(split[f"X_cont_{s}"]) for s in ("train", "val", "test")) == n

    split["X_cont_train_s"] = split["X_cont_train"]
    cond = magi.build_conditioning_and_weights(split, n_types=3)
    assert cond["cond_dim"] == 6
    np.testing.assert_array_equal(cond["cond_train"].numpy()[:, 3:], split["primary_cond_train"])


def _truth_frame():
    return pd.DataFrame({
        "ParticleName": ["mu-", "e-", "gamma"], "Energy": [1.0, 2.0, 3.0],
        "X": [1.0, 2, 3], "Y": [4.0, 5, 6], "Z": [7.0, 8, 9],
        "Vx": [0.0, 0, 0], "Vy": [0.0, 0, 0], "Vz": [-1.0, -1, -1],
        "GroupKey": [5_000_000_000, 5_000_000_000, 7], "PrimaryTrackId": [1, 1, 1],
        "PrimaryPDG": [13, 13, 22], "PrimaryEnergy": [1000.0, 1000.0, 1.13],
    })


def test_binary_v2_round_trip(tmp_path):
    assert BINARY_RECORD_V2_DTYPE.itemsize == 52
    p = str(tmp_path / "g.gntbin")
    _write_binary_header(p, 0, version=MAGI_BINARY_VERSION_PRIMTRUTH)
    n = _append_primary_truth_binary_chunk(_truth_frame(), p)
    _rewrite_binary_header(p, n, version=MAGI_BINARY_VERSION_PRIMTRUTH)
    import os
    assert os.path.getsize(p) == 20 + 52 * 3

    version, rec = magi.read_detector_binary(p)
    assert version == 2
    np.testing.assert_array_equal(rec["group_key"], [5_000_000_000, 5_000_000_000, 7])
    np.testing.assert_array_equal(rec["pdg"], [13, 11, 22])
    np.testing.assert_array_equal(rec["prim_pdg"], [13, 13, 22])
    np.testing.assert_allclose(rec["prim_energy"], [1000.0, 1000.0, 1.13], rtol=1e-6)
    np.testing.assert_allclose(rec["z"], [7, 8, 9])


def _write_gps(tmp_path):
    spec_a = tmp_path / "ESpectrumA.mac"
    spec_a.write_text("/gps/hist/point 1.0 0.2\n/gps/hist/point 2.0 0.3\n/gps/hist/point 4.0 0.5\n")
    spec_b = tmp_path / "ESpectrumB.mac"
    spec_b.write_text("/gps/hist/point 10.0 0.0\n/gps/hist/point 20.0 1.0\n")
    macro = tmp_path / "gps.mac"
    macro.write_text(
        "# two sources\n/gps/source/intensity 3\n/gps/particle mu-\n"
        "/gps/ene/type User\n/gps/hist/type energy\n"
        f"/control/execute {spec_a}\n\n"
        "/gps/source/add 1\n/gps/particle mu-\n"
        "/gps/ene/type User\n/gps/hist/type energy\n"
        f"/control/execute {spec_b}\n"
    )
    return str(macro)


def test_gps_sampler_matches_geant4_inverse_cdf(tmp_path):
    sources = magi.parse_gps_macro(_write_gps(tmp_path))
    assert [s["intensity"] for s in sources] == [3.0, 1.0]
    rng = np.random.default_rng(2)
    e, src = magi.sample_gps_energies(sources, 400_000, rng)
    assert abs(np.mean(src == 0) - 0.75) < 0.005
    a = e[src == 0]
    # point mass at the first edge: w0 / sum = 0.2
    assert abs(np.mean(a == 1.0) - 0.2) < 0.005
    # cdf 0.2 -> 0.5 linear on [1, 2], 0.5 -> 1 on [2, 4]
    assert abs(np.mean((a > 1.0) & (a <= 2.0)) - 0.3) < 0.005
    assert a.max() <= 4.0
    b = e[src == 1]
    # zero-weight first point: no point mass, uniform on [10, 20]
    assert np.mean(b == 10.0) < 1e-3
    assert abs(b.mean() - 15.0) < 0.05


def _qt(x):
    return sklearn.preprocessing.QuantileTransformer(
        n_quantiles=100, output_distribution="normal").fit(x.reshape(-1, 1))


def _tiny_model(n_primary_types):
    line_y = np.array([np.log10(0.5)], dtype=np.float32)
    return magi.CVAE_MixEnergy_ContPhi_TaskAdaptive(
        n_types=3, line_positions_y=line_y, latent_dim=4, hidden=(16, 16),
        continuum_mode="flow", continuum_flow_bins=8, continuum_flow_transforms=2,
        prior="coupling", prior_n_layers=2, prior_hidden=(8, 8),
        cond_dim=3 + n_primary_types + 1,
    )


def _geometry_metadata(rng):
    return {
        "geometry_transform": "quantile_u_r_u_v_phi_r_phi_v",
        "qt_u_r": _qt(rng.uniform(0, 1, 1000)), "qt_u_v": _qt(rng.uniform(0, 1, 1000)),
        "qt_phi_r": _qt(rng.uniform(-np.pi, np.pi, 1000)),
        "qt_phi_v": _qt(rng.uniform(-np.pi, np.pi, 1000)),
    }


def _species_table(k_class, type_idx):
    """Every primary crosses: half with k=1 (type_idx[0]), half with k=2
    (both of type_idx[1]); k_class picks which k classes are allowed."""
    k_counts = np.zeros((20, 3), dtype=int)
    type_counts = np.zeros((20, 3, 3), dtype=int)
    for k, t in zip(k_class, type_idx):
        k_counts[:, k] = 1
        type_counts[:, k - 1, t] = 1
    return {"n_thrown": 1, "thrown": [1] * 20, "k_counts": k_counts.tolist(),
            "type_counts": type_counts.tolist()}


def test_generate_primary_groups_writes_tagged_contiguous_groups(tmp_path):
    model = _tiny_model(1)
    rng = np.random.default_rng(3)
    qt = _qt
    geometry_metadata = _geometry_metadata(rng)

    # every primary crosses: k in {1, 2}, types all mu- for k=1 and e- for k=2
    edges = np.linspace(-3, 7, 21)
    table = {"log10E_edges": edges.tolist(), "type_names": TYPES, "max_k_class": 3,
             "species": {"13": _species_table((1, 2), (0, 1))}}

    out = str(tmp_path / "gen_mu.gntbin")
    tf.random.set_seed(0)
    side = magi.generate_primary_groups(
        model=model, filepath=out, species_pdg=13, n_primaries=500,
        gps_macro=_write_gps(tmp_path), multiplicity_table=table,
        idx_to_type=dict(enumerate(TYPES)), primary_type_to_idx={13: 0},
        qt_primary_energy=qt(rng.uniform(0, 1.5, 1000)),
        center=(0.0, 0.0, 5.0), radius=95.0, geometry_metadata=geometry_metadata,
        energy_transform="log10", chunk_primaries=200, seed=4, key_offset=1000,
        verbose=0,
    )
    assert side["N_primaries_sampled"] == 500
    assert sum(side["thrown_hist"]) == 500
    assert len(side["thrown_hist"]) == len(side["log10E_edges"]) - 1 == 20
    assert side["n_rows_written"] == side["n_rows_generated"] == sum(
        int(k) * v for k, v in side["k_hist"].items())

    _, rec = magi.read_detector_binary(out)
    keys = rec["group_key"]
    assert keys.min() >= 1000 and keys.max() < 1500
    # contiguous: a key never reappears after its run ends
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    assert len(starts) == len(np.unique(keys)) == side["n_groups_written"]
    sizes = np.diff(np.r_[starts, len(keys)])
    pdg_first = rec["pdg"][starts]
    assert np.all(pdg_first[sizes == 1] == 13) and np.all(pdg_first[sizes == 2] == 11)
    assert np.all(rec["prim_pdg"] == 13) and np.all(rec["prim_track_id"] == 1)
    # one energy per group, from the GPS spectrum
    for s, n in zip(starts, sizes):
        assert np.all(rec["prim_energy"][s:s + n] == rec["prim_energy"][s])
    assert rec["prim_energy"].min() >= 1.0 and rec["prim_energy"].max() <= 20.0
    # crossings land on the sphere
    r = np.sqrt(rec["x"] ** 2 + rec["y"] ** 2 + (rec["z"] - 5.0) ** 2)
    np.testing.assert_allclose(r, 95.0, rtol=1e-4)
    # an untrained model points about half its crossings outward: all resampled
    assert side["n_outgoing_first_draw"] > 0 and side["n_outgoing_kept"] == 0
    radial = rec["x"] * rec["vx"] + rec["y"] * rec["vy"] + (rec["z"] - 5.0) * rec["vz"]
    assert np.all(radial <= 0)
    with open(out + ".json") as f:
        assert json.load(f)["binary_version"] == 2


def _write_mixed_gps(tmp_path):
    # SRON-CR-like: one GPS, several species. mu- on [1, 4] MeV (intensity 3),
    # gamma on [10, 20] MeV (intensity 1).
    macro = _write_gps(tmp_path)
    text = open(macro).read()
    head, tail = text.split("/gps/source/add 1\n/gps/particle mu-\n")
    with open(macro, "w") as f:
        f.write(head + "/gps/source/add 1\n/gps/particle gamma\n" + tail)
    return macro


def test_generate_primary_groups_mixed_species_macro(tmp_path):
    import magi.generation.primaries as prim
    seen = []
    real = prim.generate_latent_outputs
    def spy(**kw):
        seen.append((np.asarray(kw["gen_type_idx"]).copy(), np.asarray(kw["extra_cond"]).copy()))
        return real(**kw)
    prim.generate_latent_outputs = spy
    try:
        _check_mixed_species(tmp_path, seen)
    finally:
        prim.generate_latent_outputs = real


def _check_mixed_species(tmp_path, seen):

    model = _tiny_model(2)
    rng = np.random.default_rng(5)
    edges = np.linspace(-3, 7, 21)
    # mu-: always k=1 of type mu-; gamma: always k=2 of type gamma
    table = {"log10E_edges": edges.tolist(), "type_names": TYPES, "max_k_class": 3,
             "species": {"13": _species_table((1,), (0,)),
                         "22": _species_table((2,), (2,))}}
    common = dict(
        model=model, gps_macro=_write_mixed_gps(tmp_path), multiplicity_table=table,
        idx_to_type=dict(enumerate(TYPES)), primary_type_to_idx={13: 0, 22: 1},
        qt_primary_energy=_qt(rng.uniform(0, 1.5, 1000)),
        center=(0.0, 0.0, 5.0), radius=95.0, geometry_metadata=_geometry_metadata(rng),
        energy_transform="log10", chunk_primaries=300, verbose=0, inward_only=False,
    )
    out = str(tmp_path / "gen_mixed.gntbin")
    tf.random.set_seed(0)
    side = magi.generate_primary_groups(filepath=out, n_primaries=2000, seed=6, **common)

    assert side["species_pdg"] is None and side["species_pdgs"] == [13, 22]
    per = side["per_species"]
    n_mu, n_g = per["13"]["N_primaries_sampled"], per["22"]["N_primaries_sampled"]
    assert n_mu + n_g == side["N_primaries_sampled"] == 2000
    assert abs(n_mu / 2000 - 0.75) < 0.04  # the macro's own intensity ratio
    assert sum(per["22"]["thrown_hist"]) == n_g and per["22"]["k_hist"] == {"2": n_g}
    assert np.array_equal(np.add(per["13"]["thrown_hist"], per["22"]["thrown_hist"]),
                          side["thrown_hist"])

    _, rec = magi.read_detector_binary(out)
    keys = rec["group_key"]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    assert len(starts) == len(np.unique(keys)) == 2000
    sizes = np.diff(np.r_[starts, len(keys)])
    # each group's species, multiplicity, crossing types and energy agree
    mu = rec["prim_pdg"][starts] == 13
    assert np.all(rec["prim_pdg"][starts][~mu] == 22)
    assert np.all(sizes[mu] == 1) and np.all(sizes[~mu] == 2)
    assert np.all(rec["pdg"][rec["prim_pdg"] == 13] == 13)
    assert np.all(rec["pdg"][rec["prim_pdg"] == 22] == 22)
    for s, n in zip(starts, sizes):
        assert np.all(rec["prim_pdg"][s:s + n] == rec["prim_pdg"][s])
    e = rec["prim_energy"]
    assert e[rec["prim_pdg"] == 13].max() <= 4.0 and e[rec["prim_pdg"] == 22].min() >= 10.0

    # the model saw each row's own primary one-hot
    types = np.concatenate([t for t, _ in seen])
    extra = np.concatenate([x for _, x in seen])
    np.testing.assert_array_equal(extra[:, 0], (types == 0).astype(np.float32))
    np.testing.assert_array_equal(extra[:, 1], (types == 2).astype(np.float32))

    # species_pdg picks one species out of the mixed macro
    out_g = str(tmp_path / "gen_gamma.gntbin")
    side_g = magi.generate_primary_groups(filepath=out_g, n_primaries=300, seed=7,
                                          species_pdg=22, **common)
    assert side_g["species_pdg"] == 22 and side_g["per_species"]["22"]["N_primaries_sampled"] == 300
    _, rec_g = magi.read_detector_binary(out_g)
    assert np.all(rec_g["prim_pdg"] == 22) and rec_g["prim_energy"].min() >= 10.0


def test_energy_of_crossing_primaries_follows_the_simulation_inside_a_bin(tmp_path):
    """P(k > 0) rises steeply inside one coarse bin: E_prim of the crossing primaries must follow the simulated
    (E_prim, k), not the spectrum (the slow-muon defect of the G3 arm, docs/CVAE_vs_AutoRegFlow.md A9)."""
    rng = np.random.default_rng(1)
    lo, hi = 1.5, 2.0                                   # one coarse bin of 0.5 dex (31.6 .. 100 MeV)
    n = 200_000
    loge = rng.uniform(lo, hi, n)                       # flat in log10 E inside the bin
    p_cross = 0.02 * ((loge - lo) / (hi - lo)) ** 4     # crossing only near the top of the bin
    crosses = rng.random(n) < p_cross
    rows = [(i, "mu-", 1.0, 1, 1, 0, "Primary", 1, 2, 3, 0, 0, -1, 1, 13, float(10 ** loge[i]))
            for i in np.flatnonzero(crosses)]
    df = pd.DataFrame(rows, columns=magi.data.DEFAULT_COLUMNS_DET_PRIMTRUTH)
    h = [0] * 202
    for x in loge:
        h[1 + int(np.floor((x + 3) / 0.05))] += 1
    summ = {"n_events": n, "log10E_edges": np.linspace(-3, 7, 201), "thrown": {13: np.array(h)},
            "recorded": {13: np.array(h)}, "jobs": []}
    table = magi.build_multiplicity_table(df, summ, TYPE_TO_IDX)
    assert table["species"]["13"]["e_quantiles"][9][0]          # bin 9 = 31.6..100 MeV, k class 1

    draw = rng.uniform(lo, hi, n)
    k, _ = magi.sample_multiplicity(table, 13, draw, rng)
    new = magi.resample_primary_energy(table, 13, draw, k, rng)
    assert np.array_equal(new[k == 0], draw[k == 0])              # non-crossing primaries untouched
    assert np.all((new >= lo) & (new < hi))                      # the coarse bin is kept
    real_med, gen_med = np.median(loge[crosses]), np.median(new[k > 0])
    assert abs(gen_med - real_med) < 0.01                        # follows the simulated joint
    assert abs(np.median(draw[k > 0]) - real_med) > 0.1          # the spectrum draw does not


def test_resample_without_quantiles_is_a_no_op():
    table = {"log10E_edges": [0, 1, 2], "max_k_class": 3,
             "species": {"13": {"k_counts": [[1, 1], [1, 1]], "type_counts": [[[1, 0, 0]] * 3] * 2}}}
    x = np.array([0.5, 1.5]); k = np.array([1, 0])
    assert np.array_equal(magi.resample_primary_energy(table, 13, x, k, np.random.default_rng(0)), x)


def test_group_types_follow_the_resampled_energy():
    """Inside one coarse bin the crossing is an electron at low E_prim and a muon at high E_prim: the types drawn
    for the resampled E_prim must follow that, which the coarse type table cannot."""
    rng = np.random.default_rng(2)
    lo, hi = 1.5, 2.0
    loge = rng.uniform(lo, hi, 20_000)
    name = np.where(loge < 1.75, "e-", "mu-")
    rows = [(i, name[i], 1.0, 1, 1, 0, "Primary", 1, 2, 3, 0, 0, -1, 1, 13, float(10 ** loge[i])) for i in range(len(loge))]
    df = pd.DataFrame(rows, columns=magi.data.DEFAULT_COLUMNS_DET_PRIMTRUTH)
    h = [0] * 202
    for x in loge:
        h[1 + int(np.floor((x + 3) / 0.05))] += 2      # half of the thrown primaries cross
    summ = {"n_events": 2 * len(loge), "log10E_edges": np.linspace(-3, 7, 201), "thrown": {13: np.array(h)},
            "recorded": {13: np.array(h)}, "jobs": []}
    table = magi.build_multiplicity_table(df, summ, TYPE_TO_IDX)
    x = np.r_[np.full(500, 1.6), np.full(500, 1.9)]
    t = magi.sample_group_types(table, 13, x, np.ones(1000, int), rng)
    assert np.mean(t[:500] == TYPE_TO_IDX["e-"]) > 0.95 and np.mean(t[500:] == TYPE_TO_IDX["mu-"]) > 0.95
