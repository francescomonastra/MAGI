"""
Shower multiplicity for HENABICS primary truth.

MAGI generates crossings one at a time, conditioned on the primary that
produced them. To rebuild whole showers it also needs, per primary, how many
crossings to draw (k, the multiplicity at the MAGI sphere) and of which
types. Both are measured here from the full simulation:

    P(k | primary type, log10 E_prim bin),        k >= 0
    P(crossing type | primary type, E bin, k class)

k = 0 matters as much as k > 0: most primaries never reach the sphere, and
dropping them would inflate the flux. They are not in the crossing table, so
they are counted as thrown - recorded, with thrown from the RunAction's
run_summary.json histograms (magi.data.io.load_run_summaries).

The crossings of one primary are then generated independently given the
primary - the MVP approximation; a shared per-shower latent is the planned
second iteration (docs/MAGI_HENABICS_integration_requirements.md).
"""

import numpy as np

# How a group key is built, persisted next to the table so the closure test
# joins the same way.
PRIMARY_JOIN_KEY_RULE = (
    "(species, EventId, PrimaryTrackId); EventId = Geant4 EventID + "
    "job * PER_JOB in a full-simulation merge, or the injected GroupKey in a "
    "post-MAGI run; species files are never mixed at the Geant4 level"
)


def _coarse_bin(log10_e, edges):
    """Bin index on `edges`, with under/overflow folded into the end bins."""
    idx = np.searchsorted(edges, log10_e, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def build_multiplicity_table(crossings, run_summary, type_to_idx, rebin=10,
                             max_k_class=3, n_energy_quantiles=65):
    """
    Measure P(k | primary, E) and P(crossing type | primary, E, k) from a
    full simulation.

    Parameters
    ----------
    crossings : pd.DataFrame
        Primary-truth crossing table (EventId, ParticleName, PrimaryPDG,
        PrimaryEnergy), restricted to the crossing types the model was
        trained on - k then counts exactly the crossings MAGI can generate.

    run_summary : dict
        Result of magi.data.io.load_run_summaries for the same runs.

    type_to_idx : dict[str, int]
        The model's crossing-type map.

    rebin : int
        Fine run_summary bins merged per coarse E bin. The RunAction grid is
        0.05 dex, so the default 10 gives 0.5 dex bins.

    max_k_class : int
        Crossing types are tabulated per k class min(k, max_k_class): in a
        k = 1 shower the crossing is mostly the primary itself, in larger
        ones mostly secondaries.

    n_energy_quantiles : int
        Quantiles of log10 E_prim stored per (coarse bin, k class) for the
        primaries that cross ("e_quantiles"). P(k > 0) can change steeply inside
        a coarse bin: on DM1.2 a muon of 31.6-100 MeV reaches the sphere far
        more often at 90 MeV than at 35 MeV, and drawing E_prim from the
        spectrum and k from the bin put the crossings at the wrong E_prim
        (four times too many below 50 MeV, which then stop inside the sphere).
        `resample_primary_energy` redraws E_prim of the crossing primaries from
        these quantiles. The crossing types are also tabulated on the fine grid
        of the run summary merged in pairs (0.1 dex, "type_counts_fine"), since
        their mix changes inside a coarse bin as well (on DM1.2, 7.5 % of the
        crossing muon primaries of 31.6-40 MeV have a muon crossing against 73 %
        at 80-100 MeV); `sample_group_types` draws them for the resampled E_prim.

    Returns
    -------
    dict
        JSON-serializable: "log10E_edges", "type_names" (index order),
        "max_k_class", "join_key_rule" and, per primary PDG (str key),
        "n_thrown", "thrown" (per coarse bin), "k_counts" (bins x kmax+1, k=0
        included), "type_counts" (bins x max_k_class x n_types) and
        "e_quantiles" (bins x max_k_class lists of log10 E_prim quantiles of the
        crossing primaries, empty where none crossed), "log10E_edges_fine"
        and "type_counts_fine" (fine bins x max_k_class x n_types).
    """
    fine = np.asarray(run_summary["log10E_edges"], dtype=np.float64)
    n_fine = len(fine) - 1
    if n_fine % rebin:
        raise ValueError(f"rebin={rebin} does not divide {n_fine} fine bins")
    edges = fine[::rebin]
    n_bins = len(edges) - 1

    names = [None] * len(type_to_idx)
    for t, i in type_to_idx.items():
        names[int(i)] = t
    n_types = len(names)

    df = crossings[crossings["ParticleName"].isin(type_to_idx)]
    table = {
        "log10E_edges": edges.tolist(),
        "type_names": names,
        "max_k_class": int(max_k_class),
        "join_key_rule": PRIMARY_JOIN_KEY_RULE,
        "species": {},
    }

    table["log10E_edges_fine"] = fine[::2].tolist()
    for pdg, thrown_fine in run_summary["thrown"].items():
        thrown_fine = np.asarray(thrown_fine, dtype=np.int64)
        # [underflow, n_fine bins, overflow] -> coarse, end bins absorbing both.
        inner = thrown_fine[1:-1].reshape(n_bins, rebin).sum(axis=1)
        inner[0] += thrown_fine[0]
        inner[-1] += thrown_fine[-1]

        sub = df[df["PrimaryPDG"] == int(pdg)]
        grp = sub.groupby("EventId")
        k = grp.size().to_numpy()
        e_bin = _coarse_bin(np.log10(grp["PrimaryEnergy"].first().to_numpy()), edges)

        k_max = int(k.max()) if len(k) else 0
        k_counts = np.zeros((n_bins, k_max + 1), dtype=np.int64)
        np.add.at(k_counts, (e_bin, k), 1)
        # Primaries with no kept crossing. Energies in the text tables are
        # printed to 6 significant digits, so a primary right on a coarse
        # edge can land one bin over; clip rather than go negative.
        k_counts[:, 0] = np.maximum(inner - k_counts[:, 1:].sum(axis=1), 0)

        k_of_group = dict(zip(grp.size().index.to_numpy(), k))
        row_k = np.minimum(sub["EventId"].map(k_of_group).to_numpy(), max_k_class)
        row_bin = _coarse_bin(np.log10(sub["PrimaryEnergy"].to_numpy()), edges)
        row_type = sub["ParticleName"].map(type_to_idx).to_numpy()
        type_counts = np.zeros((n_bins, max_k_class, n_types), dtype=np.int64)
        np.add.at(type_counts, (row_bin, row_k - 1, row_type), 1)

        # log10 E_prim of the crossing primaries per (coarse bin, k class)
        g_loge = np.log10(grp["PrimaryEnergy"].first().to_numpy())
        g_kc = np.minimum(k, max_k_class) - 1
        qs = np.linspace(0.0, 1.0, int(n_energy_quantiles))
        fine2 = fine[::2]
        row_fbin = _coarse_bin(np.log10(sub["PrimaryEnergy"].to_numpy()), fine2)
        type_counts_fine = np.zeros((len(fine2) - 1, max_k_class, n_types), dtype=np.int64)
        np.add.at(type_counts_fine, (row_fbin, row_k - 1, row_type), 1)
        e_quantiles = [[[] for _ in range(max_k_class)] for _ in range(n_bins)]
        for b_i in range(n_bins):
            for c in range(max_k_class):
                v = g_loge[(e_bin == b_i) & (g_kc == c)]
                if len(v):
                    e_quantiles[b_i][c] = [round(float(x), 6) for x in np.quantile(v, qs)]

        table["species"][str(int(pdg))] = {
            "n_thrown": int(thrown_fine.sum()),
            "thrown": inner.tolist(),
            "k_counts": k_counts.tolist(),
            "type_counts": type_counts.tolist(),
            "e_quantiles": e_quantiles,
            "type_counts_fine": type_counts_fine.tolist(),
        }
    return table


def sample_multiplicity(table, pdg, log10_e, rng):
    """
    Draw k and the crossing types for a batch of primaries of one species.

    Parameters
    ----------
    table : dict
        Result of build_multiplicity_table (or its JSON round trip).

    pdg : int
        Primary PDG code, a key of table["species"].

    log10_e : np.ndarray
        log10(E_prim / MeV) per primary.

    rng : np.random.Generator

    Returns
    -------
    k : np.ndarray of int
        Multiplicity per primary, 0 included.

    crossing_type_idx : np.ndarray of int
        sum(k) crossing-type indices, grouped per primary in input order
        (np.repeat(np.arange(n), k) gives the owning primary of each).
    """
    sp = table["species"][str(int(pdg))]
    edges = np.asarray(table["log10E_edges"])
    k_counts = np.asarray(sp["k_counts"], dtype=np.float64)
    type_counts = np.asarray(sp["type_counts"], dtype=np.float64)
    max_k_class = int(table["max_k_class"])
    n_bins = k_counts.shape[0]

    # A bin nothing was thrown into cannot say anything: borrow the nearest
    # bin that has data. Only reachable if the generation spectrum differs
    # from the simulated one.
    has = k_counts.sum(axis=1) > 0
    if not has.any():
        raise ValueError(f"multiplicity table for PDG {pdg} is empty")
    filled = np.flatnonzero(has)
    nearest = filled[np.abs(np.arange(n_bins)[:, None] - filled[None, :]).argmin(axis=1)]

    b = nearest[_coarse_bin(np.asarray(log10_e, dtype=np.float64), edges)]
    n = len(b)

    # Two stages: nearly every primary has k = 0, so test that first and only
    # build the full per-row CDF for the few that cross.
    with np.errstate(invalid="ignore"):  # empty bins: NaN rows, never indexed
        p_k = k_counts / k_counts.sum(axis=1, keepdims=True)
    k = np.zeros(n, dtype=np.int64)
    crosses = rng.random(n) >= p_k[b, 0]
    if crosses.any():
        bc = b[crosses]
        tail = p_k[bc, 1:]
        cdf = np.cumsum(tail / tail.sum(axis=1, keepdims=True), axis=1)
        u = rng.random(len(bc))[:, None]
        k[crosses] = 1 + np.minimum((u > cdf).sum(axis=1), cdf.shape[1] - 1)

    row_b = np.repeat(b, k)
    row_kc = np.minimum(np.repeat(k, k), max_k_class) - 1
    p_t = type_counts[row_b, row_kc]
    # Empty (bin, k class) cell: fall back to the bin pooled over k classes.
    empty = p_t.sum(axis=1) == 0
    if empty.any():
        p_t[empty] = type_counts[row_b[empty]].sum(axis=1)
    cdf = np.cumsum(p_t / p_t.sum(axis=1, keepdims=True), axis=1)
    u = rng.random(len(row_b))[:, None]
    types = np.minimum((u > cdf).sum(axis=1), cdf.shape[1] - 1)
    return k, types.astype(np.int32)


def resample_primary_energy(table, pdg, log10_e, k, rng):
    """
    Redraw log10 E_prim of the primaries that cross, inside their coarse bin.

    `sample_multiplicity` draws k from the coarse bin of E_prim, so the E_prim
    of a crossing primary keeps the shape of the generation spectrum inside
    the bin, while in the simulation it follows the spectrum times P(k | E_prim),
    which can rise steeply inside a bin. This draws E_prim of every primary with
    k > 0 from the stored quantiles of the crossing primaries of the same
    coarse bin and k class, so that the joint (E_prim, k) of the groups is that
    of the simulation. The coarse bin, and therefore the normalisation by the
    thrown primaries, is unchanged; primaries with k = 0 are not touched.

    Parameters
    ----------
    table : dict
        Result of build_multiplicity_table.

    pdg : int
        Primary PDG code.

    log10_e : np.ndarray
        log10(E_prim / MeV) per primary, as drawn from the spectrum.

    k : np.ndarray of int
        Multiplicity per primary, from sample_multiplicity.

    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        log10 E_prim per primary. A table without "e_quantiles" (built before
        v1.0.0) returns the input unchanged.
    """
    sp = table["species"][str(int(pdg))]
    out = np.array(log10_e, dtype=np.float64, copy=True)
    if "e_quantiles" not in sp:
        return out
    edges = np.asarray(table["log10E_edges"])
    max_k_class = int(table["max_k_class"])
    b = _coarse_bin(out, edges)
    kc = np.minimum(np.asarray(k), max_k_class) - 1
    for i in np.flatnonzero(np.asarray(k) > 0):
        q = sp["e_quantiles"][b[i]][kc[i]]
        if not q:                                   # empty cell: pool the k classes of the bin
            pooled = [x for c in sp["e_quantiles"][b[i]] for x in c]
            if not pooled:
                continue
            q = sorted(pooled)
        qa = np.asarray(q)
        out[i] = np.interp(rng.random(), np.linspace(0.0, 1.0, len(qa)), qa)
    return out


def sample_group_types(table, pdg, log10_e, k, rng):
    """
    Draw the crossing types of a batch of groups from the fine E_prim grid.

    Use after `resample_primary_energy`: the type mix of the crossings changes
    inside a coarse bin, so the types are drawn from "type_counts_fine" at the
    resampled E_prim and the k class of the group (falling back to the k classes
    pooled, then to the coarse table, where a fine cell is empty).

    Parameters
    ----------
    table : dict
        Result of build_multiplicity_table.

    pdg : int
        Primary PDG code.

    log10_e : np.ndarray
        log10(E_prim / MeV) per primary.

    k : np.ndarray of int
        Multiplicity per primary.

    rng : np.random.Generator

    Returns
    -------
    np.ndarray of int or None
        sum(k) crossing-type indices in primary order, as sample_multiplicity
        returns them, or None for a table without the fine grid.
    """
    sp = table["species"][str(int(pdg))]
    if "type_counts_fine" not in sp:
        return None
    tf_ = np.asarray(sp["type_counts_fine"], dtype=np.float64)
    tc = np.asarray(sp["type_counts"], dtype=np.float64)
    max_k_class = int(table["max_k_class"])
    fb = np.repeat(_coarse_bin(np.asarray(log10_e, dtype=np.float64), np.asarray(table["log10E_edges_fine"])), k)
    cb = np.repeat(_coarse_bin(np.asarray(log10_e, dtype=np.float64), np.asarray(table["log10E_edges"])), k)
    kc = np.minimum(np.repeat(np.asarray(k), k), max_k_class) - 1
    p = tf_[fb, kc]
    e1 = p.sum(axis=1) == 0
    if e1.any():
        p[e1] = tf_[fb[e1]].sum(axis=1)
    e2 = p.sum(axis=1) == 0
    if e2.any():
        p[e2] = tc[cb[e2], kc[e2]]
    e3 = p.sum(axis=1) == 0
    if e3.any():
        p[e3] = tc[cb[e3]].sum(axis=1)
    cdf = np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1)
    u = rng.random(len(fb))[:, None]
    return np.minimum((u > cdf).sum(axis=1), cdf.shape[1] - 1).astype(np.int32)
