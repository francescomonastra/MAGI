"""
Aggregate "network usage" circuit for the v0.8 mixture-energy model
(CVAE_MixEnergy_ContPhi_TaskAdaptive and compatible heads).

For each particle type, this traces how heavily every unit in the network is
used across many real held-out events of that type - not just one event's
trace. The usage metric is Expected Conductance: Conductance (Dhamdhere,
Sundararajan & Yan, ICLR 2019, "How Important Is A Neuron?") is the correct
generalization of Integrated Gradients to *internal* units (plain IG only
attributes importance to input features), computed here as a trapezoidal-rule
path integral of gradient x activation-delta along the straight line, in
continuous-feature space, from a real baseline event to a real target event
(the categorical one-hot type conditioning is held fixed along that path,
since interpolating a one-hot is meaningless). Averaging over a pool of real
baseline events (Erion et al. 2021, "expected gradients") avoids picking an
arbitrary, physically meaningless zero baseline - zero has no special meaning
in this model's quantile-transformed feature space. Averaging the absolute
value of that over many sampled target events of a type gives a stable
per-unit "usage" score for that type. An extra `"__all__"` entry pools usage
across every type into one whole-network view.

Two layers:

- `compute_full_circuit_trace` runs the model and returns plain
  Python/NumPy data - no HTML, no plotting. Useful on its own for
  notebook inspection.
- `render_full_circuit_html` turns that data (plus a real marginal energy
  spectrum) into a self-contained, interactive HTML document: one column of
  usage-colored units per layer (a continuous magma scale, log-transformed
  since usage is heavy-tailed), one button per particle type plus an "All
  types" button, and a terminal panel showing the real combined energy
  spectrum with the selected type's own contribution shaded on top.

`save_full_circuit` is the convenience wrapper for the common case: point
it at a trained checkpoint plus the real detector table it was trained on,
and it rebuilds the held-out test split, runs the trace for every type (plus
the pooled "All types" view), and writes the HTML.

Self-contained: the returned HTML embeds its own light/dark palette
(toggled by a button, not by `prefers-color-scheme` - some embedding
contexts do not report it reliably) rather than depending on a host
stylesheet, so the file opens correctly in a plain browser.
"""
import json
import math
import os
import re

import numpy as np
import tensorflow as tf


_DEFAULT_PALETTE = [
    "#7F77DD", "#1D9E75", "#378ADD", "#BA7517",
    "#E24B4A", "#639922", "#888780", "#0F6E56",
]
_ALL_TYPES_COLOR = "#5f5e5a"

# How many units per stage get wired to the next stage in the rendered
# diagram. The real dense weight matrix is not drawable - two consecutive
# 128-unit layers alone are 16,384 lines - so each stage contributes its
# most-used units and the wires show how the busy part of one layer feeds
# the busy part of the next.
_WIRE_TOP_N = 10

_PREFIX_SUBLABEL = {
    "enc": "encoder", "trunk": "deep_trunk", "eb": "energy_branch",
    "pos": "position_branch", "dir": "direction_branch",
    "z": "latent z", "stem": "decoder_stem", "gate": "energy_gate_head",
}
_PREFIX_SHORT = {
    "enc": "enc", "trunk": "trunk", "eb": "energy", "pos": "pos", "dir": "dir",
}


def _walk(container, x, limit=None):
    """Feed `x` through every non-Input layer of `container`, keeping each
    layer's output. `limit` stops early (used to skip head layers - e.g.
    z_mean/z_logvar - that live inside the same encoder container)."""
    h = x
    acts = []
    layers = [l for l in container.layers if l.__class__.__name__ != "InputLayer"]
    if limit is not None:
        layers = layers[:limit]
    for layer in layers:
        h = layer(h)
        acts.append(h)
    return acts


def _stratified_sample(rng, pool_idx, zones, n_zones, n_target):
    """Sample up to `n_target` indices from `pool_idx`, spreading picks
    across gate zones (continuum + each pinned line) first, so a dominant
    continuum population doesn't drown out rare line events in the sample.
    If some zones don't have enough events to fill their even share, the
    shortfall is topped up from whichever zones have spare events. Returns
    fewer than `n_target` only if `pool_idx` itself is smaller."""
    n_target = min(n_target, pool_idx.size)
    if n_target == 0:
        return np.array([], dtype=np.int64)

    buckets = [pool_idx[zones == z] for z in range(n_zones)]
    present = [b for b in buckets if b.size > 0]
    if not present:
        return np.array([], dtype=np.int64)

    shares = [n_target // len(present)] * len(present)
    for i in range(n_target - sum(shares)):
        shares[i % len(present)] += 1

    picked, leftover_pool, shortfall = [], [], 0
    for bucket, share in zip(present, shares):
        take = min(share, bucket.size)
        chosen = rng.choice(bucket, size=take, replace=False) if take else np.array([], dtype=np.int64)
        picked.append(chosen)
        shortfall += share - take
        leftover_pool.append(np.setdiff1d(bucket, chosen, assume_unique=True))

    if shortfall > 0:
        spare = np.concatenate(leftover_pool) if leftover_pool else np.array([], dtype=np.int64)
        extra = min(shortfall, spare.size)
        if extra:
            picked.append(rng.choice(spare, size=extra, replace=False))

    return np.concatenate(picked) if picked else np.array([], dtype=np.int64)


def _sample_baselines(rng, pool_idx, n_baselines):
    """Plain random sample (no replacement) of up to `n_baselines` real
    events from `pool_idx` to use as the baseline pool for one type."""
    n = min(n_baselines, pool_idx.size)
    if n == 0:
        return np.array([], dtype=np.int64)
    return rng.choice(pool_idx, size=n, replace=False)


def _expected_conductance_for_type(model, n_hidden, target_rows, cond_row,
                                    baseline_rows, n_steps, chunk_size=32):
    """Expected Conductance for one type's target events, one baseline pool,
    one fixed one-hot `cond_row`.

    For every (target, baseline) pair, walks `n_steps + 1` points along the
    straight line in continuous-feature space from the baseline (alpha=0) to
    the target (alpha=1), with `cond_row` held fixed at every point (only
    `target_rows`/`baseline_rows` - the `y_cont` block - is interpolated).
    At each point, runs the same forward pass `compute_full_circuit_trace`
    used to trace one event, gets every stage's activation and the model's
    own per-event reconstruction-loss target, and accumulates per-unit
    conductance via the trapezoidal-rule Riemann sum

        conductance_j = sum_k 0.5*(grad_j(x_k-1)+grad_j(x_k)) * (a_j(x_k)-a_j(x_k-1))

    which reduces to plain gradient x activation (against an implicit zero
    baseline) when `n_steps=1`. Averages over the baseline pool (Expected
    Gradients). Chunks over `target_rows` to bound peak memory - the whole
    `(M*B*(n_steps+1), width)` activation tensor never needs to exist for a
    large M at once.

    Parameters
    ----------
    model : CVAE_MixEnergy_ContPhi_TaskAdaptive
        A loaded (or freshly built) v0.8 mixture model - same requirements
        as `compute_full_circuit_trace`.

    n_hidden : int
        Number of encoder hidden layers (`len(model_config["hidden"])`).

    target_rows : np.ndarray, shape (M, 5 + n_zones)
        `y_cont` rows (the `_reconstruction_terms`-shaped continuous block,
        no `cond`) for the M sampled target events.

    cond_row : np.ndarray, shape (n_types,)
        One-hot type vector, fixed for every point on every path in this
        call.

    baseline_rows : np.ndarray, shape (B, 5 + n_zones)
        `y_cont` rows for the B baseline events, same type as `cond_row`.

    n_steps : int
        Number of interpolation steps K (K+1 points are evaluated per pair,
        including both endpoints).

    chunk_size : int
        Max number of target events processed in one batched forward pass;
        recurses over chunks of `target_rows` above this.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, int]]
        `conductance[stage_name]` has shape `(M, width)` - signed, per-event
        (already averaged over the B baselines, not yet made absolute or
        averaged over events - callers combine multiple types' arrays
        before doing that, for the pooled "__all__" view).
        `widths[stage_name]` is that stage's unit count.
    """
    M = target_rows.shape[0]
    if M > chunk_size:
        parts, widths = [], None
        for start in range(0, M, chunk_size):
            chunk_conductance, widths = _expected_conductance_for_type(
                model, n_hidden, target_rows[start:start + chunk_size], cond_row,
                baseline_rows, n_steps, chunk_size=chunk_size)
            parts.append(chunk_conductance)
        merged = {name: np.concatenate([p[name] for p in parts], axis=0) for name in widths}
        return merged, widths

    B = baseline_rows.shape[0]
    K = n_steps
    D = target_rows.shape[1]

    alphas = np.linspace(0.0, 1.0, K + 1, dtype=np.float32)
    interp = (
        baseline_rows[None, :, None, :]
        + alphas[None, None, :, None] * (target_rows[:, None, None, :] - baseline_rows[None, :, None, :])
    ).astype(np.float32)
    rows = M * B * (K + 1)
    y_cont_flat = interp.reshape(rows, D)
    cond_flat = np.tile(cond_row.astype(np.float32), (rows, 1))
    y_true = tf.constant(y_cont_flat)
    cond_tf = tf.constant(cond_flat)
    x_in = tf.concat([y_true, cond_tf], axis=1)

    with tf.GradientTape(persistent=True) as tape:
        enc_full = _walk(model.encoder, x_in, limit=2 * n_hidden)
        enc_post = enc_full[1::2]
        for a in enc_post:
            tape.watch(a)
        z_mean = model.encoder.get_layer("z_mean")(enc_full[-1])
        z = z_mean
        tape.watch(z)

        base = tf.concat([z, cond_tf], axis=1)

        stem_full = _walk(model.decoder_stem, base)
        stem = stem_full[-1]
        tape.watch(stem)

        trunk_full = _walk(model.decoder_deep_trunk, stem)
        trunk_post = trunk_full[1::2]
        deep = trunk_full[-1]
        for a in trunk_post:
            tape.watch(a)

        eb_full = _walk(model.energy_branch, stem)
        eb_post = eb_full[1::2]
        energy_feat = eb_full[-1]
        for a in eb_post:
            tape.watch(a)

        pos_full = _walk(model.position_branch, deep)
        pos_post = pos_full[1::2]
        pos_feat = pos_full[-1]
        for a in pos_post:
            tape.watch(a)

        dir_full = _walk(model.direction_branch, deep)
        dir_post = dir_full[1::2]
        dir_feat = dir_full[-1]
        for a in dir_post:
            tape.watch(a)

        flow_cond = model.energy_cont_head(energy_feat, training=False)
        gate_logits = model.energy_gate_head(energy_feat)
        tape.watch(gate_logits)

        ur_feat = model.ur_head(pos_feat, training=False)
        ur_mu = model.ur_mu_head(ur_feat)
        ur_logsigma = tf.clip_by_value(
            model.ur_logsigma_head(ur_feat), model.min_log_sigma, model.max_log_sigma)

        phi_r_feat = model.phi_r_head(pos_feat, training=False)
        phi_r_mu = model.phi_r_mu_head(phi_r_feat)
        phi_r_logsigma = tf.clip_by_value(
            model.phi_r_logsigma_head(phi_r_feat), model.min_log_sigma, model.max_log_sigma)

        uv_feat = model.uv_head(dir_feat, training=False)
        uv_mu = model.uv_mu_head(uv_feat)
        uv_logsigma = tf.clip_by_value(
            model.uv_logsigma_head(uv_feat), model.min_log_sigma, model.max_log_sigma)

        phi_v_feat = model.phi_v_head(dir_feat, training=False)
        phi_v_mu = model.phi_v_mu_head(phi_v_feat)
        phi_v_logsigma = tf.clip_by_value(
            model.phi_v_logsigma_head(phi_v_feat), model.min_log_sigma, model.max_log_sigma)

        params = {
            "energy_gate_logits": gate_logits, "flow_cond": flow_cond,
            "ur_mu": ur_mu, "ur_logsigma": ur_logsigma,
            "uv_mu": uv_mu, "uv_logsigma": uv_logsigma,
            "phi_r_mu": phi_r_mu, "phi_r_logsigma": phi_r_logsigma,
            "phi_v_mu": phi_v_mu, "phi_v_logsigma": phi_v_logsigma,
        }
        rec_per, _pieces = model._reconstruction_terms(y_true, params)
        target_sum = tf.reduce_sum(-rec_per)

    stage_acts = {}
    for i, a in enumerate(enc_post):
        stage_acts[f"enc{i + 1}"] = a
    stage_acts["z"] = z
    stage_acts["stem"] = stem
    for i, a in enumerate(trunk_post):
        stage_acts[f"trunk{i + 1}"] = a
    for i, a in enumerate(eb_post):
        stage_acts[f"eb{i + 1}"] = a
    for i, a in enumerate(pos_post):
        stage_acts[f"pos{i + 1}"] = a
    for i, a in enumerate(dir_post):
        stage_acts[f"dir{i + 1}"] = a
    stage_acts["gate"] = gate_logits

    conductance, widths = {}, {}
    for name, act in stage_acts.items():
        grad = tape.gradient(target_sum, act)
        a_np = act.numpy()
        g_np = grad.numpy() if grad is not None else np.zeros_like(a_np)
        width = a_np.shape[-1]
        widths[name] = width
        a4 = a_np.reshape(M, B, K + 1, width)
        g4 = g_np.reshape(M, B, K + 1, width)
        grad_trap = 0.5 * (g4[:, :, :-1, :] + g4[:, :, 1:, :])
        act_diff = a4[:, :, 1:, :] - a4[:, :, :-1, :]
        cond_mb = np.sum(grad_trap * act_diff, axis=2)
        conductance[name] = cond_mb.mean(axis=1)
    del tape

    return conductance, widths


def compute_full_circuit_trace(model, model_config, X_cont_test, y_type_test, idx_to_type, n_zones,
                                n_events_per_type=128, n_baselines=16, n_steps=32, random_state=42):
    """Aggregate "network usage" trace: Expected Conductance per unit, per
    particle type, across many real held-out events - plus one pooled
    `"__all__"` entry across every type.

    For each type, samples up to `n_events_per_type` held-out events
    (stratified across gate zones - continuum + each pinned line - so rare
    line events aren't drowned out by the dominant continuum population)
    and pairs each against a shared pool of up to `n_baselines` other
    held-out events of the *same* type (real data, never a synthetic zero
    vector - zero has no physical meaning in this model's
    quantile-transformed feature space). See `_expected_conductance_for_type`
    for the path-integral computation itself. Taking the absolute value of
    each event's Expected Conductance and averaging over the sampled events
    gives a stable per-unit "usage" score for that type (abs before the
    average, so a unit that matters with different signs on different real
    events still reads as "used," not as canceling to ~0).

    The `"__all__"` entry pools every type's already-computed per-event
    conductance together (no extra model passes - it's a free by-product of
    computing the per-type entries) before taking the absolute value and
    averaging, giving one whole-network usage view.

    Parameters
    ----------
    model : CVAE_MixEnergy_ContPhi_TaskAdaptive
        A loaded (or freshly built) v0.8 mixture model. Must expose
        `encoder`, `decoder_stem`, `decoder_deep_trunk`, `energy_branch`,
        `position_branch`, `direction_branch`, the per-quantity heads
        (`energy_gate_head`, `energy_cont_head`, `ur_head`, `phi_r_head`,
        `uv_head`, `phi_v_head` and their `_mu_head`/`_logsigma_head`
        pairs), and `_reconstruction_terms`.

    model_config : dict
        The model's `to_generation_config()`-shaped config (only
        `"hidden"` is read, to know where the encoder's hidden stack ends).

    X_cont_test : np.ndarray, shape (n_test, 5 + n_zones + 1)
        Held-out continuous features, column layout
        `[u_r_q, u_v_q, phi_r_q, phi_v_q, energy_y, gate_target_0..n_zones-1,
        energy_mev_physical]` - the same layout `_reconstruction_terms`
        expects, plus one trailing physical-energy column not used here.

    y_type_test : np.ndarray, shape (n_test,)
        Integer type index per row, aligned with `X_cont_test`.

    idx_to_type : dict[int, str]
        Type index -> name, e.g. `{0: "gamma", 1: "e-"}`.

    n_zones : int
        1 + number of pinned lines (continuum + lines).

    n_events_per_type : int
        Max number of target events sampled per type, capped to what's
        actually held out for that type.

    n_baselines : int
        Max number of baseline events sampled per type, capped similarly.

    n_steps : int
        Number of path-interpolation steps K between each baseline and
        target event (see `_expected_conductance_for_type`).

    random_state : int
        Seeds the event/baseline sampling, for reproducibility.

    Returns
    -------
    dict[str, dict]
        One entry per type that had at least one held-out event, plus one
        `"__all__"` entry: `{"stages": {stage_name: {"width": int,
        "usage": [float, ...]}}, "n_sampled": int, "n_available": int,
        "n_baselines": int}` (`"__all__"` additionally has
        `"n_types_pooled": int`). `stages` is ordered encoder -> z -> stem
        -> trunk -> branches -> gate, matching the model's actual data flow.
    """
    n_hidden = len(model_config["hidden"])
    n_types = len(idx_to_type)
    rng = np.random.default_rng(random_state)

    where_by_type, zones_by_type = {}, {}
    for t in range(n_types):
        where = np.nonzero(y_type_test == t)[0]
        where_by_type[t] = where
        if where.size:
            zones_by_type[t] = np.argmax(X_cont_test[where, 5:5 + n_zones], axis=1)

    per_type_sample = {}
    for t in range(n_types):
        where = where_by_type[t]
        if where.size == 0:
            continue
        target_idx = _stratified_sample(rng, where, zones_by_type[t], n_zones, n_events_per_type)
        baseline_idx = _sample_baselines(rng, where, n_baselines)
        if target_idx.size == 0 or baseline_idx.size == 0:
            continue
        per_type_sample[t] = (target_idx, baseline_idx)

    results = {}
    pooled_conductance, pooled_widths = {}, {}
    pooled_m, n_types_pooled = 0, 0

    for t, (target_idx, baseline_idx) in per_type_sample.items():
        cond_row = np.eye(n_types, dtype=np.float32)[t]
        target_rows = X_cont_test[target_idx, :5 + n_zones].astype(np.float32)
        baseline_rows = X_cont_test[baseline_idx, :5 + n_zones].astype(np.float32)

        conductance, widths = _expected_conductance_for_type(
            model, n_hidden, target_rows, cond_row, baseline_rows, n_steps)

        results[idx_to_type[t]] = {
            "stages": {
                name: {"width": widths[name], "usage": np.abs(arr).mean(axis=0).tolist()}
                for name, arr in conductance.items()
            },
            "n_sampled": int(target_idx.size),
            "n_available": int(where_by_type[t].size),
            "n_baselines": int(baseline_idx.size),
        }

        for name, arr in conductance.items():
            pooled_conductance.setdefault(name, []).append(arr)
            pooled_widths[name] = widths[name]
        pooled_m += target_idx.size
        n_types_pooled += 1

    if pooled_conductance:
        results["__all__"] = {
            "stages": {
                name: {
                    "width": pooled_widths[name],
                    "usage": np.abs(np.concatenate(arrs, axis=0)).mean(axis=0).tolist(),
                }
                for name, arrs in pooled_conductance.items()
            },
            "n_sampled": int(pooled_m),
            "n_available": int(sum(w.size for w in where_by_type.values())),
            "n_baselines": n_baselines,
            "n_types_pooled": n_types_pooled,
        }

    return results


def _stage_groups(stage_names):
    """Split stage names like 'enc1', 'trunk3' into (prefix, index) and
    group by prefix, sorted by index - so the renderer's layout adapts to
    whatever encoder/trunk/branch depth the model actually has."""
    groups = {}
    for name in stage_names:
        m = re.match(r"^([a-zA-Z]+)(\d*)$", name)
        prefix, num = m.group(1), m.group(2)
        groups.setdefault(prefix, []).append((int(num) if num else 0, name))
    for prefix in groups:
        groups[prefix].sort()
        groups[prefix] = [name for _, name in groups[prefix]]
    return groups


def render_full_circuit_html(types_data, zone_labels, lines, spectrum, type_order,
                              source_label="", type_colors=None, title=None):
    """Render a `compute_full_circuit_trace` result as a standalone,
    interactive HTML document.

    Layout: encoder -> latent z -> decoder stem -> deep trunk, then three
    parallel lanes branching off (energy branches off the stem directly;
    position and direction branch off the end of the trunk), ending in an
    energy-gate node and a real marginal energy spectrum panel. Each stage
    is drawn as a column of unit circles colored by that unit's *usage*
    (Expected Conductance, see `compute_full_circuit_trace`) on one shared,
    log-scaled color scale across the whole network - so parts of the
    network barely used by a type show up dim and heavily used parts show
    up bright, comparable across stages. One button per particle type plus
    an "All types" button (pooled usage across every type) switches the
    view live - no server, no rebuild.

    Parameters
    ----------
    types_data : dict[str, dict]
        Output of `compute_full_circuit_trace`. Must include a `"__all__"`
        entry.

    zone_labels : list[str]
        Gate zone names, `["continuum", <line 1 label>, ...]`, in gate
        column order.

    lines : list[dict]
        Pinned line markers for the spectrum panel, each
        `{"label": str, "energy_mev": float}`.

    spectrum : dict
        `{"log_edges": [float, ...], "combined": [int, ...],
        "by_type": {type_name: [int, ...]}}` - a log10(E/MeV) histogram of
        the real marginal spectrum (combined) and each type's own
        sub-histogram, same bin edges. Typically built directly from the
        real detector table with `numpy.histogram`.

    type_order : list[str]
        Real particle-type names in the order/selection the per-type
        buttons should show, e.g. `[t for t in types_data if t != "__all__"]`.
        The "All types" button is added automatically and does not belong
        in this list.

    source_label : str
        Short name shown in the title and spectrum caption, e.g. "CR" or
        "K-40 (Small)".

    type_colors : dict[str, str] or None
        Optional `{type_name: "#rrggbb"}` override. Types not in the dict
        (or when the dict is None) get an accent color from a built-in
        8-color palette, cycling by position in `type_order`. `"__all__"`
        defaults to a neutral gray unless overridden.

    title : str or None
        Optional heading override. Defaults to
        "Network usage across real {source_label} events".

    Returns
    -------
    str
        A complete, self-contained HTML fragment (own `<style>` and
        `<script>`, no external requests, no host stylesheet dependency).

    Raises
    ------
    ValueError
        If `type_order` is empty, references a type missing from
        `types_data`, or `types_data` has no `"__all__"` entry.
    """
    if not type_order:
        raise ValueError("type_order must be non-empty.")
    missing = [t for t in type_order if t not in types_data]
    if missing:
        raise ValueError(f"type_order references types missing from types_data: {missing}")
    if "__all__" not in types_data:
        raise ValueError(
            'types_data must include an "__all__" pooled entry (from compute_full_circuit_trace).')

    type_colors = dict(type_colors or {})
    for i, t in enumerate(type_order):
        type_colors.setdefault(t, _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)])
    type_colors.setdefault("__all__", _ALL_TYPES_COLOR)

    stage_names = list(types_data[type_order[0]]["stages"])
    groups = _stage_groups(stage_names)

    enc_chain = groups.get("enc", [])
    trunk_chain = groups.get("trunk", [])
    eb_chain = groups.get("eb", [])
    pos_chain = groups.get("pos", [])
    dir_chain = groups.get("dir", [])

    main_chain = enc_chain + ["z", "stem"] + trunk_chain
    energy_lane = ["stem"] + eb_chain + ["gate"]
    position_lane = ([trunk_chain[-1]] if trunk_chain else ["stem"]) + pos_chain
    direction_lane = ([trunk_chain[-1]] if trunk_chain else ["stem"]) + dir_chain

    edges = (
        list(zip(main_chain, main_chain[1:]))
        + list(zip(energy_lane, energy_lane[1:]))
        + list(zip(position_lane, position_lane[1:]))
        + list(zip(direction_lane, direction_lane[1:]))
    )

    # 350px apart, not 300: pitch_for floors at 2.0px, so a 128-unit column
    # is ~254px tall rather than the nominal max_col_h, and at 300px spacing
    # one lane's labels landed on the next lane's columns.
    lane_y = {"top": 210, "mid": 560, "bot": 910}
    # Wide enough that a stage's "trunk 1"-style label clears its neighbours;
    # the block sub-label ("deep_trunk") is drawn once per block rather than
    # once per column, which is what used to collide at close spacing.
    x_spacing = 116
    x0 = 120
    x = {name: x0 + i * x_spacing for i, name in enumerate(main_chain)}
    branch_x0 = x[main_chain[-1]] + x_spacing
    for i, name in enumerate(eb_chain):
        x[name] = branch_x0 + i * x_spacing
    gate_x = branch_x0 + len(eb_chain) * x_spacing
    x["gate"] = gate_x
    for i, name in enumerate(pos_chain):
        x[name] = branch_x0 + i * x_spacing
    for i, name in enumerate(dir_chain):
        x[name] = branch_x0 + i * x_spacing

    lane_of = {name: "mid" for name in main_chain}
    for name in eb_chain + ["gate"]:
        lane_of[name] = "top"
    for name in pos_chain:
        lane_of[name] = "mid"
    for name in dir_chain:
        lane_of[name] = "bot"

    n_zones = len(zone_labels)
    widths = {name: types_data[type_order[0]]["stages"][name]["width"] for name in stage_names}
    widths["gate"] = n_zones

    max_col_h = 200

    def pitch_for(width):
        return max(2.0, min(26.0, max_col_h / max(width, 1)))

    def unit_positions(width, cx, cy):
        p = pitch_for(width)
        h = p * (width - 1)
        y0 = cy - h / 2.0
        return [(cx, y0 + i * p) for i in range(width)], p

    stage_units = {}
    for name in list(stage_names) + ["gate"]:
        pts, pitch = unit_positions(widths[name], x[name], lane_y[lane_of[name]])
        stage_units[name] = {"x": x[name], "lane": lane_of[name], "width": widths[name],
                              "pitch": pitch, "pts": pts}

    svg_w = max(x.values()) + 550
    svg_h = 1120

    def short_label(name):
        m = re.match(r"^([a-zA-Z]+)(\d*)$", name)
        prefix, num = m.group(1), m.group(2)
        if prefix in ("z", "stem", "gate"):
            return {"z": "z", "stem": "stem", "gate": "gate"}[prefix]
        return f"{_PREFIX_SHORT.get(prefix, prefix)} {num}"

    def sub_label(name):
        prefix = re.match(r"^([a-zA-Z]+)", name).group(1)
        return _PREFIX_SUBLABEL.get(prefix, prefix)

    parts = []
    parts.append(
        f'<svg id="circuit-svg" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-labelledby="circuit-title">'
    )
    parts.append(
        f'<title id="circuit-title">MAGI network-usage circuit for {source_label}: encoder, decoder '
        f'trunk, and all three branches, aggregated over real held-out events</title>'
    )

    for y in lane_y.values():
        parts.append(
            f'<line x1="60" y1="{y}" x2="{svg_w - 40}" y2="{y}" stroke="var(--border)" '
            f'stroke-width="1" stroke-dasharray="2 4" opacity="0.5"/>'
        )

    def backbone(x1, y1, x2, y2):
        if y1 == y2:
            return f'<path class="backbone" d="M {x1} {y1} L {x2} {y2}"/>'
        mx = (x1 + x2) / 2.0
        return f'<path class="backbone" d="M {x1} {y1} C {mx} {y1} {mx} {y2} {x2} {y2}"/>'

    bb = []
    for a, b in edges:
        xa, ya = stage_units[a]["x"], lane_y[stage_units[a]["lane"]]
        xb, yb = stage_units[b]["x"], lane_y[stage_units[b]["lane"]]
        bb.append(backbone(xa, ya, xb, yb))
    parts.append(f'<g fill="none" stroke="var(--text-muted)" stroke-width="1.75">{"".join(bb)}</g>')

    # The block sub-label ("encoder", "deep_trunk", ...) is drawn once, over
    # the first column of each block, and spans the block's full width - the
    # old per-column repeat both collided with its neighbours and said the
    # same word three times in a row.
    block_start = {}
    for name in stage_units:
        prefix = re.match(r"^([a-zA-Z]+)", name).group(1)
        block_start.setdefault(prefix, []).append(name)

    for name, info in stage_units.items():
        label = "gate" if name == "gate" else short_label(name)
        top_y = min(p[1] for p in info["pts"]) - 20
        parts.append(f'<text x="{info["x"]}" y="{top_y}" text-anchor="middle" class="stage-label">{label}</text>')
        parts.append(
            f'<text x="{info["x"]}" y="{max(p[1] for p in info["pts"]) + 18}" '
            f'text-anchor="middle" class="stage-width">{info["width"]} units</text>'
        )

    for prefix, names in block_start.items():
        first, last = names[0], names[-1]
        sub = "energy_gate_head" if prefix == "gate" else _PREFIX_SUBLABEL.get(prefix, prefix)
        info_f, info_l = stage_units[first], stage_units[last]
        cx = (info_f["x"] + info_l["x"]) / 2.0
        top_y = min(min(p[1] for p in stage_units[n]["pts"]) for n in names) - 20
        parts.append(
            f'<text x="{cx:.1f}" y="{top_y - 13}" text-anchor="middle" class="stage-sub">{sub}</text>'
        )

    # Lane titles clear whatever the tallest column in that lane actually
    # turned out to be - measured, not assumed: pitch_for floors at 2.0px, so
    # a wide layer overshoots max_col_h and a fixed offset lands the title on
    # top of the stage labels.
    lane_label_top = {}
    for name, info in stage_units.items():
        top = min(p[1] for p in info["pts"]) - 20 - 13
        lane = info["lane"]
        lane_label_top[lane] = min(lane_label_top.get(lane, top), top)

    for lane, text in (("top", "energy path"),
                       ("mid", "encoder -&gt; z -&gt; decoder trunk -&gt; position path"),
                       ("bot", "direction path")):
        y = lane_label_top.get(lane, lane_y[lane] - 150) - 18
        parts.append(f'<text x="60" y="{y:.0f}" class="lane-title">{text}</text>')

    parts.append(
        f'<g class="c-gray">'
        f'<rect x="20" y="{lane_y["mid"] - 24}" width="60" height="48" rx="8"/>'
        f'<text x="50" y="{lane_y["mid"] - 4}" text-anchor="middle" class="t">event</text>'
        f'<text x="50" y="{lane_y["mid"] + 14}" text-anchor="middle" class="ts">+ one-hot</text>'
        f'</g>'
    )
    first_stage_x = stage_units[main_chain[0]]["x"] if main_chain else stage_units["stem"]["x"]
    parts.append(
        f'<line x1="80" y1="{lane_y["mid"]}" x2="{first_stage_x - 14}" y2="{lane_y["mid"]}" '
        f'stroke="var(--text-muted)" stroke-width="1.75"/>'
    )

    # Drawn before the unit circles so the wires pass behind them, and
    # populated by renderType() - the wire set depends on which units are
    # most used by the currently selected type.
    parts.append('<g id="connectors" fill="none"></g>')

    for name, info in stage_units.items():
        g = [f'<g id="units-{name}">']
        for i, (px, py) in enumerate(info["pts"]):
            r = max(1.1, min(5.0, info["pitch"] * 0.42))
            g.append(
                f'<circle id="u-{name}-{i}" cx="{px:.2f}" cy="{py:.2f}" r="{r:.2f}" '
                f'class="unit" fill="#333"/>'
            )
        g.append("</g>")
        parts.append("".join(g))

    gate_pts = stage_units["gate"]["pts"]
    for i, lbl in enumerate(zone_labels):
        px, py = gate_pts[i]
        # There is ~230px of clear space between the gate column and the
        # spectrum panel, so zone names can be spelled out rather than cut
        # to "e+e- annihilati.".
        short = lbl if len(lbl) <= 26 else lbl[:25] + "."
        parts.append(f'<text x="{px + 10}" y="{py + 3}" class="zone-tick">{short}</text>')

    spec_x0, spec_x1 = gate_x + 230, svg_w - 40
    spec_y0, spec_y1 = 170, 900
    log_edges = spectrum["log_edges"]
    log_min, log_max = log_edges[0], log_edges[-1]
    combined = spectrum["combined"]
    max_count = max(combined) if combined else 1
    y_log_max = math.log10(max_count + 1)

    def sx(loge):
        return spec_x0 + (loge - log_min) / (log_max - log_min) * (spec_x1 - spec_x0)

    def sy(count):
        v = math.log10(count + 1) / y_log_max if y_log_max > 0 else 0
        return spec_y1 - v * (spec_y1 - spec_y0)

    def area_path(hist, edges):
        pts = []
        for i in range(len(hist)):
            y = sy(hist[i])
            pts.append((sx(edges[i]), y))
            pts.append((sx(edges[i + 1]), y))
        path = f"M {pts[0][0]:.1f} {spec_y1:.1f} "
        for (px, py) in pts:
            path += f"L {px:.1f} {py:.1f} "
        path += f"L {pts[-1][0]:.1f} {spec_y1:.1f} Z"
        return path

    parts.append(
        f'<g><text x="{(spec_x0 + spec_x1) / 2:.0f}" y="{spec_y0 - 74}" text-anchor="middle" '
        f'class="stage-label">real {source_label} energy spectrum</text>'
        f'<text x="{(spec_x0 + spec_x1) / 2:.0f}" y="{spec_y0 - 58}" text-anchor="middle" '
        f'class="stage-sub">all {len(type_order)} types, {sum(combined):,} events - selected type shaded</text></g>'
    )
    decades = []
    c = 1
    while c <= max_count * 1.3:
        decades.append(c)
        c *= 10
    for c in decades:
        y = sy(c)
        if y < spec_y0 - 2 or y > spec_y1 + 2:
            continue
        parts.append(f'<line x1="{spec_x0}" y1="{y:.1f}" x2="{spec_x1}" y2="{y:.1f}" stroke="var(--border)" stroke-width="1"/>')
        label = f"{c:,}" if c < 1000 else f"{c:.0e}".replace("e+0", "e").replace("e+", "e")
        parts.append(f'<text x="{spec_x0 - 8}" y="{y + 3:.1f}" text-anchor="end" class="axis-tick">{label}</text>')
    tick_defs = [(-3, "1 keV"), (-2, "10 keV"), (-1, "100 keV"), (0, "1 MeV"),
                 (1, "10 MeV"), (2, "100 MeV"), (3, "1 GeV"), (4, "10 GeV"), (5, "100 GeV")]
    # Every decade gets a tick, but a label only if it clears the previous
    # one. Over CR's ~9 decades in a fixed-width panel the decade labels are
    # about 45px wide and ~35px apart, so labelling all of them renders
    # "1 keV 10 keV100 keVMeV10 MeV" as one smear.
    last_label_x = None
    min_label_gap = 52.0
    for logv, label in tick_defs:
        if logv < log_min or logv > log_max:
            continue
        tx = sx(logv)
        parts.append(f'<line x1="{tx:.1f}" y1="{spec_y1}" x2="{tx:.1f}" y2="{spec_y1 + 6}" stroke="var(--text-muted)" stroke-width="1"/>')
        if last_label_x is None or (tx - last_label_x) >= min_label_gap:
            parts.append(f'<text x="{tx:.1f}" y="{spec_y1 + 20}" text-anchor="middle" class="axis-tick">{label}</text>')
            last_label_x = tx
    for ln in lines:
        logv = math.log10(ln["energy_mev"])
        if logv < log_min or logv > log_max:
            continue
        tx = sx(logv)
        parts.append(f'<line x1="{tx:.1f}" y1="{spec_y0 - 6}" x2="{tx:.1f}" y2="{spec_y0}" stroke="var(--text-muted)" stroke-width="1"/>')
        parts.append(
            f'<text x="{tx:.1f}" y="{spec_y0 - 9}" text-anchor="start" class="line-tick" '
            f'transform="rotate(-38 {tx:.1f} {spec_y0 - 9})">{ln["label"]}</text>'
        )

    parts.append(f'<path id="spec-combined" d="{area_path(combined, log_edges)}"/>')
    parts.append('<path id="spec-type" d=""/>')
    parts.append(
        f'<rect x="{spec_x0}" y="{spec_y0}" width="{spec_x1 - spec_x0}" height="{spec_y1 - spec_y0}" '
        f'fill="none" stroke="var(--border-strong)" stroke-width="1"/>'
    )
    parts.append("</svg>")
    svg_body = "".join(parts)

    # Color scale: percentile rank within the distribution pooled over every
    # group and every stage. Usage is heavy-tailed (most units contribute
    # near-nothing, a few dominate), so a linear - or even log - min-max
    # parks the bulk of the units in magma's near-black lower third and the
    # whole diagram reads as uniformly dark. Ranking spreads the units evenly
    # across the full color range instead. Pooling across groups (rather than
    # ranking each type separately) keeps the scale comparable between
    # buttons: a type that uses the network less overall stays visibly dimmer
    # everywhere, instead of being re-stretched to look identical to a heavy
    # user of the network.
    pooled = np.sort(np.abs(np.asarray(
        [v for rec in types_data.values()
         for st in rec["stages"].values() for v in st["usage"]],
        dtype=float,
    )))
    if pooled.size == 0:
        pooled = np.zeros(1)

    def to_pct_rank(values):
        idx = np.searchsorted(pooled, np.abs(np.asarray(values, dtype=float)), side="right")
        return [int(round(i / pooled.size * 100)) for i in idx]

    quantized = {}
    for tname, rec in types_data.items():
        stages_q = {
            name: {"usage100": to_pct_rank(stage["usage"])}
            for name, stage in rec["stages"].items()
        }
        rec_q = {
            "stages": stages_q,
            "n_sampled": rec["n_sampled"],
            "n_available": rec["n_available"],
            "n_baselines": rec["n_baselines"],
        }
        if "n_types_pooled" in rec:
            rec_q["n_types_pooled"] = rec["n_types_pooled"]
        quantized[tname] = rec_q

    # Absolute-magnitude companion to the (rank-based) node colors. The color
    # scale deliberately spreads units over its full range, which makes the
    # diagram readable but means a dark unit is only "low-ranked", not
    # "unused" - by construction some units are always dark. These are the
    # raw conductance magnitudes, which is what a question like "is this
    # layer wider than it needs to be?" actually has to be answered from.
    abs_stats = {}
    for tname, rec in types_data.items():
        totals = {
            name: float(np.abs(np.asarray(st["usage"], dtype=float)).sum())
            for name, st in rec["stages"].items()
        }
        net_total = sum(totals.values()) or 1e-30
        net_units = sum(len(st["usage"]) for st in rec["stages"].values()) or 1
        rows = []
        for name, st in rec["stages"].items():
            u = np.sort(np.abs(np.asarray(st["usage"], dtype=float)))[::-1]
            tot = totals[name]
            k = max(1, int(round(0.2 * u.size)))
            share = tot / net_total
            rows.append({
                "stage": name,
                "width": int(u.size),
                "total": tot,
                "mean": tot / max(u.size, 1),
                "share": share,
                # work relative to width: share of the network's conductance
                # divided by share of its units. 1.0 = the layer pulls exactly
                # its weight; 0.2 = it holds five times more units than its
                # contribution justifies. This, not concentration alone, is
                # what separates an oversized layer from a merely peaky one.
                "load": share / (u.size / net_units) if u.size else 0.0,
                # concentration: how much of this layer's own work its busiest
                # fifth does. ~0.2 means the layer is used evenly; near 1.0
                # means a handful of units carry it and the rest are padding.
                "top20": float(u[:k].sum() / tot) if tot > 0 else 0.0,
                # units contributing under 1% of the layer's own busiest unit
                "faint": float(np.mean(u < 0.01 * u[0])) if u[0] > 0 else 1.0,
            })
        abs_stats[tname] = rows

    data = {
        "zone_labels": zone_labels,
        "types": quantized,
        "abs_stats": abs_stats,
        "spectrum": spectrum,
        "type_order": type_order,
    }
    data_json = json.dumps(data, separators=(",", ":"))
    layout_json = json.dumps({
        "stage_units": {k: {"x": v["x"], "lane": v["lane"], "width": v["width"]} for k, v in stage_units.items()},
        "edges": [[a, b] for a, b in edges],
        "type_color": type_colors,
        "spec": {"x0": spec_x0, "x1": spec_x1, "y0": spec_y0, "y1": spec_y1,
                 "log_min": log_min, "log_max": log_max, "y_log_max": y_log_max},
    }, separators=(",", ":"))

    buttons = "".join(
        f'<button class="type-btn" data-type="{t}" style="--tc:{type_colors[t]}" onclick="renderType(\'{t}\')">{t}</button>'
        for t in type_order
    )
    buttons += (
        f'<button class="type-btn" data-type="__all__" style="--tc:{type_colors["__all__"]}" '
        f'onclick="renderType(\'__all__\')">All types</button>'
    )
    heading = title or f"Network usage across real {source_label} events"

    html_doc = f"""
<style>
:root {{
  --text-primary: #17171a; --text-secondary: #52514e; --text-muted: #6b6a63;
  --text-danger: #A32D2D; --text-success: #2f5c11;
  --border: rgba(15,15,13,0.16); --border-strong: rgba(15,15,13,0.32);
  --surface-1: #efede6; --surface-2: #ffffff; --radius: 8px;
  --page-bg: #fbfbf9;
}}
.circuit-wrap.theme-dark {{
  --text-primary: #f5f5f2; --text-secondary: #d3d1c6; --text-muted: #a3a196;
  --text-danger: #f0a3a3; --text-success: #a6d97f;
  --border: rgba(255,255,255,0.18); --border-strong: rgba(255,255,255,0.34);
  --surface-1: #292925; --surface-2: #1b1b19; --page-bg: #131312;
}}
html, body {{ background: var(--page-bg); margin: 0; }}
.circuit-wrap {{ font-family: -apple-system, "Segoe UI", sans-serif; color: var(--text-primary); background: var(--page-bg); padding: 18px 20px; border-radius: 14px; }}
.circuit-wrap h1 {{ font-size: 18px; font-weight: 500; margin: 0 0 4px; }}
.circuit-wrap .subtitle {{ font-size: 13px; color: var(--text-secondary); margin: 0 0 14px; line-height: 1.5; }}
.top-row {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.type-btn, .theme-toggle {{
  background: transparent; border: 0.5px solid var(--border-strong); border-radius: var(--radius, 8px);
  padding: 6px 12px; font-size: 13px; color: var(--text-secondary); cursor: pointer;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}}
.type-btn:hover, .theme-toggle:hover {{ background: var(--surface-1); }}
.type-btn.active {{ background: var(--tc); color: #fff; border-color: var(--tc); font-weight: 500; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 18px; align-items: center; margin: 4px 0 14px; font-size: 12px; color: var(--text-secondary); }}
.grad-bar {{ display:inline-block; width:90px; height:9px; border-radius:4px; vertical-align:-1px; margin: 0 6px; background: linear-gradient(90deg,#1a1523,#5b3a7a,#b8434f,#e8912a,#f7e56a); }}
.svg-scroll {{ overflow-x: auto; background: var(--surface-1); border-radius: 12px; border: 0.5px solid var(--border); padding: 10px 4px; }}
#circuit-svg {{ display: block; margin: 0 auto; min-width: {svg_w}px; }}
/* No `fill` here on purpose: renderType() sets each unit's fill as a
   presentation attribute, and a CSS rule would silently override it (CSS
   beats presentation attributes in SVG), leaving every unit one flat color
   no matter its usage. The initial fill is set on the <circle> itself. */
#circuit-svg .unit {{ stroke: none; }}
#circuit-svg .stage-label {{ font-size: 12px; font-weight: 500; fill: var(--text-primary); }}
#circuit-svg .stage-sub, #circuit-svg .stage-width {{ font-size: 9.5px; fill: var(--text-muted); }}
#circuit-svg .lane-title {{ font-size: 11px; fill: var(--text-secondary); font-weight: 500; }}
#circuit-svg .zone-tick {{ font-size: 9.5px; fill: var(--text-secondary); dominant-baseline: middle; }}
#circuit-svg .axis-tick {{ font-size: 10px; fill: var(--text-muted); }}
#circuit-svg .line-tick {{ font-size: 9.5px; fill: var(--text-secondary); }}
#circuit-svg .backbone {{ opacity: 0.9; }}
#circuit-svg .c-gray rect {{ fill: var(--surface-1); stroke: var(--border-strong); stroke-width: 1; }}
#circuit-svg .c-gray text {{ fill: var(--text-primary); }}
#circuit-svg .t {{ font-size: 12px; font-weight: 500; }}
#circuit-svg .ts {{ font-size: 9.5px; fill: var(--text-secondary); }}
#spec-combined {{ fill: var(--border-strong); opacity: 0.55; stroke: var(--text-muted); stroke-width: 1; }}
#spec-type {{ opacity: 0.5; stroke-width: 1.6; }}
.stats-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
.stat-card {{ background: var(--surface-1); border-radius: var(--radius, 8px); padding: 10px 14px; min-width: 140px; }}
.stat-card .label {{ font-size: 11px; color: var(--text-muted); margin-bottom: 3px; }}
.stat-card .value {{ font-size: 15px; font-weight: 500; }}
.abs-panel {{ margin-top: 16px; border: 0.5px solid var(--border); border-radius: var(--radius, 8px); background: var(--surface-1); padding: 10px 14px; }}
.abs-panel summary {{ cursor: pointer; font-size: 13px; font-weight: 500; color: var(--text-primary); }}
.abs-note {{ font-size: 12px; color: var(--text-secondary); line-height: 1.55; margin: 10px 0 12px; }}
.abs-scroll {{ overflow-x: auto; }}
.abs-table {{ border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; width: 100%; }}
.abs-table th, .abs-table td {{ text-align: right; padding: 4px 10px; white-space: nowrap; border-bottom: 0.5px solid var(--border); }}
.abs-table th {{ color: var(--text-muted); font-weight: 500; }}
.abs-table td.stage-cell, .abs-table th.stage-cell {{ text-align: left; color: var(--text-primary); }}
.abs-table tr.flagged td {{ color: var(--text-danger); }}
.abs-table tr.flagged td.stage-cell::after {{ content: " - width-reduction candidate"; font-size: 10.5px; }}
.sr-only {{ position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); }}
</style>
<div class="circuit-wrap">
  <h1>{heading}</h1>
  <p class="subtitle">Aggregate network usage - Expected Conductance (the generalization of Integrated
  Gradients to internal units) averaged over real held-out events per type, against a pool of other real
  held-out events of the same type as the baseline (never a synthetic zero vector). Circle fill = that
  unit's usage; wires join the {_WIRE_TOP_N} most-used units of each stage to the next and take the color of their
  weaker end, so a link between two busy units glows and a link into a quiet one stays dark. Both use one
  shared color scale (percentile rank across the <i>whole</i> network and every type), so parts of the
  network barely used by a type show up dim and heavily used parts show up bright, comparably across
  stages and across buttons. "All types" pools every type's sampled events into one whole-network view.</p>
  <div class="top-row">
    <div class="controls" role="group" aria-label="select particle type">{buttons}</div>
    <button class="theme-toggle" type="button" onclick="toggleTheme()" id="theme-toggle-btn">Dark mode</button>
  </div>
  <div class="legend">
    <span><span class="grad-bar" role="img" aria-label="usage color scale from low to high"></span>usage: low to high (percentile rank, whole network)</span>
  </div>
  <div class="svg-scroll">{svg_body}</div>
  <div class="stats-row" id="stats-row"></div>
  <details class="abs-panel">
    <summary>Per-layer absolute usage (raw conductance, not rank)</summary>
    <p class="abs-note">The colors above are <em>percentile ranks</em>, which spread units evenly over the
    scale by construction - some units always look dark, whatever the real distribution is. These are the
    unnormalized magnitudes, for questions the color scale cannot answer, like whether a layer is wider
    than it needs to be. <b>share</b> is the layer's fraction of the whole network's conductance;
    <b>load</b> is that share divided by the layer's share of the network's units - 1.0x means it pulls
    exactly its weight, 0.2x means it holds five times more units than its contribution justifies;
    <b>top-20%</b> is how much of the layer's own total its busiest fifth carries (0.2 = used evenly,
    near 1.0 = a few units do the work); <b>faint</b> is the fraction of units under 1% of that layer's
    busiest unit. Rows flagged in red have <b>load</b> under 0.5x <em>and</em> a real population of faint
    units - concentration on its own is not enough, since a layer can be peaky and still carry its weight.
    A flag is a place to run an experiment, not a verdict: attribution cannot prove a unit is removable,
    only an ablation and a re-scored run can.</p>
    <div class="abs-scroll"><table class="abs-table" id="abs-table"></table></div>
  </details>
  <p class="sr-only" id="sr-summary" aria-live="polite"></p>
</div>
<script>
const DATA = {data_json};
const LAYOUT = {layout_json};

function magma(t) {{
  t = Math.max(0, Math.min(1, t));
  const stops = [[0.10,0.08,0.14],[0.36,0.23,0.48],[0.72,0.26,0.31],[0.91,0.57,0.16],[0.97,0.90,0.41]];
  const n = stops.length - 1;
  const seg = Math.min(n - 1, Math.floor(t * n));
  const localT = t * n - seg;
  const a = stops[seg], b = stops[seg + 1];
  const r = Math.round(255 * (a[0] + (b[0]-a[0]) * localT));
  const g = Math.round(255 * (a[1] + (b[1]-a[1]) * localT));
  const bl = Math.round(255 * (a[2] + (b[2]-a[2]) * localT));
  return `rgb(${{r}},${{g}},${{bl}})`;
}}

let currentType = DATA.type_order[0];

const WIRE_TOP_N = {_WIRE_TOP_N};

function clearGroup(g) {{ while (g.firstChild) g.removeChild(g.firstChild); }}

function unitPos(stage, i) {{
  const el = document.getElementById(`u-${{stage}}-${{i}}`);
  if (!el) return null;
  return [parseFloat(el.getAttribute('cx')), parseFloat(el.getAttribute('cy'))];
}}

function topUnits(stage, n) {{
  const vals = stage.usage100;
  const idx = vals.map((v, i) => i);
  idx.sort((a, b) => vals[b] - vals[a]);
  return idx.slice(0, Math.min(n, idx.length));
}}

function drawWire(g, x1, y1, x2, y2, color, width, opacity) {{
  const mx = (x1 + x2) / 2;
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', `M ${{x1}} ${{y1}} C ${{mx}} ${{y1}} ${{mx}} ${{y2}} ${{x2}} ${{y2}}`);
  p.setAttribute('stroke', color);
  p.setAttribute('stroke-width', width);
  p.setAttribute('opacity', opacity);
  g.appendChild(p);
}}

function toggleTheme() {{
  const wrap = document.querySelector('.circuit-wrap');
  const dark = wrap.classList.toggle('theme-dark');
  document.getElementById('theme-toggle-btn').textContent = dark ? 'Light mode' : 'Dark mode';
}}

function sci(v) {{
  if (v === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(v)));
  return (v / Math.pow(10, e)).toFixed(2) + 'e' + e;
}}

function renderAbsTable(typeName) {{
  const rows = DATA.abs_stats[typeName] || [];
  let html =
    '<thead><tr>' +
    '<th class="stage-cell">layer</th><th>units</th><th>total |C|</th><th>mean/unit</th>' +
    '<th>share</th><th>load</th><th>top-20%</th><th>faint</th>' +
    '</tr></thead><tbody>';
  for (const r of rows) {{
    // Oversized means "does much less work than its width would imply AND
    // has a real population of near-silent units". Concentration alone is
    // not enough - a layer can be peaky and still be carrying its weight.
    const flagged = r.load < 0.5 && r.faint >= 0.15;
    html +=
      `<tr class="${{flagged ? 'flagged' : ''}}">` +
      `<td class="stage-cell">${{r.stage}}</td>` +
      `<td>${{r.width}}</td>` +
      `<td>${{sci(r.total)}}</td>` +
      `<td>${{sci(r.mean)}}</td>` +
      `<td>${{(r.share * 100).toFixed(1)}}%</td>` +
      `<td>${{r.load.toFixed(2)}}x</td>` +
      `<td>${{(r.top20 * 100).toFixed(0)}}%</td>` +
      `<td>${{(r.faint * 100).toFixed(0)}}%</td>` +
      '</tr>';
  }}
  document.getElementById('abs-table').innerHTML = html + '</tbody>';
}}

function renderType(typeName) {{
  currentType = typeName;
  const rec = DATA.types[typeName];
  const color = LAYOUT.type_color[typeName];

  document.querySelectorAll('.type-btn').forEach(b => b.classList.toggle('active', b.dataset.type === typeName));

  for (const [name, info] of Object.entries(LAYOUT.stage_units)) {{
    const stage = rec.stages[name];
    const vals = stage.usage100;
    for (let i = 0; i < vals.length; i++) {{
      const el = document.getElementById(`u-${{name}}-${{i}}`);
      if (!el) continue;
      el.setAttribute('fill', magma(vals[i] / 100));
    }}
  }}

  // Per-neuron wires: the most-used units of each stage fanned into the
  // most-used units of the next. Each wire takes the color of the weaker of
  // the two endpoints it joins (a link is only as used as its dimmer end),
  // so a wire between two heavily used units glows and a wire into a barely
  // used unit stays dark - and the whole wire set re-colors per particle
  // type, including the pooled "All types" view.
  const connG = document.getElementById('connectors');
  clearGroup(connG);
  for (const [a, b] of LAYOUT.edges) {{
    const stageA = rec.stages[a], stageB = rec.stages[b];
    if (!stageA || !stageB) continue;
    const srcTop = topUnits(stageA, WIRE_TOP_N);
    const dstTop = topUnits(stageB, WIRE_TOP_N);
    for (const si of srcTop) {{
      const pa = unitPos(a, si);
      if (!pa) continue;
      for (const ti of dstTop) {{
        const pb = unitPos(b, ti);
        if (!pb) continue;
        const strength = Math.min(stageA.usage100[si], stageB.usage100[ti]) / 100;
        drawWire(connG, pa[0], pa[1], pb[0], pb[1], magma(strength),
                 0.4 + 0.9 * strength, 0.12 + 0.42 * strength);
      }}
    }}
  }}

  renderAbsTable(typeName);

  const spec = LAYOUT.spec;
  function sx(logE) {{ return spec.x0 + (logE - spec.log_min) / (spec.log_max - spec.log_min) * (spec.x1 - spec.x0); }}
  function sy(count) {{ const v = spec.y_log_max > 0 ? Math.log10(count + 1) / spec.y_log_max : 0; return spec.y1 - v * (spec.y1 - spec.y0); }}
  const specType = document.getElementById('spec-type');
  if (typeName === '__all__') {{
    specType.setAttribute('d', '');
  }} else {{
    const hist = DATA.spectrum.by_type[typeName] || DATA.spectrum.combined.map(() => 0);
    const edges = DATA.spectrum.log_edges;
    let d = `M ${{sx(edges[0]).toFixed(1)}} ${{spec.y1}} `;
    for (let i = 0; i < hist.length; i++) {{
      const y = sy(hist[i]);
      d += `L ${{sx(edges[i]).toFixed(1)}} ${{y.toFixed(1)}} L ${{sx(edges[i+1]).toFixed(1)}} ${{y.toFixed(1)}} `;
    }}
    d += `L ${{sx(edges[edges.length-1]).toFixed(1)}} ${{spec.y1}} Z`;
    specType.setAttribute('d', d);
    specType.setAttribute('fill', color);
    specType.setAttribute('stroke', color);
  }}

  const cards = [
    `<div class="stat-card"><div class="label">events sampled</div><div class="value">${{rec.n_sampled}}</div></div>`,
    `<div class="stat-card"><div class="label">held-out available</div><div class="value">${{rec.n_available}}</div></div>`,
    `<div class="stat-card"><div class="label">baseline pool</div><div class="value">${{rec.n_baselines}}</div></div>`,
  ];
  if (rec.n_types_pooled !== undefined) {{
    cards.push(`<div class="stat-card"><div class="label">types pooled</div><div class="value">${{rec.n_types_pooled}}</div></div>`);
  }}
  document.getElementById('stats-row').innerHTML = cards.join('');
  const label = typeName === '__all__' ? 'all types pooled' : typeName;
  document.getElementById('sr-summary').textContent =
    `Showing ${{label}}: usage aggregated over ${{rec.n_sampled}} sampled events.`;
}}

renderType(currentType);
</script>
"""
    return html_doc


def save_full_circuit(save_dir, model_name, training_data_filepath, candidate_lines_file,
                       center, radius, resolution_ev=4.0, output_html=None, random_state=42,
                       energy_bins=512, n_quantiles=10000, spectrum_bins=140,
                       spectrum_log_range=(-3.5, 5.9), n_events_per_type=128,
                       n_baselines=16, n_steps=32):
    """Load a v0.8 mixture checkpoint plus the real detector table it was
    trained on, rebuild its held-out test split, and write the aggregate
    network-usage circuit (every type, plus "All types", every stage) as a
    standalone HTML file.

    This re-derives the training-time preprocessing (line matching, gate
    targets, quantile/log10 transforms, the train/test split) from the raw
    detector table rather than trusting anything cached in the checkpoint,
    since the checkpoint only stores the trained weights and config - not a
    held-out sample to trace. Uses the same `random_state` as
    `tools/run_v0_8_real.py` by default, both for the held-out split and for
    the event/baseline sampling inside `compute_full_circuit_trace`.

    Parameters
    ----------
    save_dir : str
        Run directory holding `<model_name>_config.json` and
        `<model_name>_metadata.json` plus the saved weights.

    model_name : str
        Filename stem used when the run was saved.

    training_data_filepath : str
        Path to the raw detector table (the same file the run was trained
        on - e.g. the CR or Small `.dat` file).

    candidate_lines_file : str
        Path to the candidate energy lines JSON used at training time
        (see `load_candidate_energy_lines`).

    center : tuple[float, float, float]
        Detector-sphere center used at training time.

    radius : float
        Detector-sphere radius used at training time.

    resolution_ev : float
        Detector energy resolution (FWHM, eV) used at training time for
        line matching and gate-target bandwidth. Default 4.0 (this
        project's X-IFU resolution).

    output_html : str or None
        Destination path. None (default) writes to
        `<save_dir>/<model_name>_full_circuit.html`.

    random_state : int
        Passed to `build_feature_dataframe`'s quantile transform, to
        `split_feature_data`, and to `compute_full_circuit_trace`'s event/
        baseline sampling, so the whole run is reproducible.

    energy_bins, n_quantiles : int
        Passed through to `build_feature_dataframe`.

    spectrum_bins : int
        Number of log10(E/MeV) bins for the terminal spectrum panel.

    spectrum_log_range : tuple[float, float]
        (min, max) log10(E/MeV) range for the spectrum panel.

    n_events_per_type, n_baselines, n_steps : int
        Passed through to `compute_full_circuit_trace` - see there for what
        each controls (sampled target events, baseline pool size, and
        path-interpolation steps per Expected Conductance estimate).

    Returns
    -------
    str
        The path written.

    Raises
    ------
    TypeError
        If the loaded checkpoint is not a v0.8 mixture-energy model (no
        `decoder_deep_trunk`/`position_branch` to trace).
    """
    from ..data.io import load_detector_table, load_candidate_energy_lines
    from ..data.preprocessing import (
        build_physical_features, detect_energy_lines, measure_line_centroid,
        build_feature_dataframe, build_gate_targets,
    )
    from ..data.dataset import filter_particle_types_continuous_geometry, split_feature_data
    from ..training.checkpointing import load_json, load_task_adaptive_model_for_generation

    model_config = load_json(os.path.join(save_dir, f"{model_name}_config.json"))
    line_positions_y = model_config["line_positions_y"]
    candidate_lines = load_candidate_energy_lines(candidate_lines_file)["lines"]

    df = load_detector_table(filepath=training_data_filepath, sep=r"\s+")
    prep = build_physical_features(df, center=center, radius=radius)
    E = prep["features"]["Energy"].to_numpy()

    res = detect_energy_lines(
        E, binning_mode="log_fixed_count", n_bins=1024, prominence_factor=3.0, window=5,
        candidate_lines=candidate_lines, refine_bin_width_mev=resolution_ev * 1e-6,
    )
    coarse_by_label = {m["label"]: m for m in res["matched_lines"]}
    E_sorted = np.sort(E)
    matched = []
    for y in np.asarray(line_positions_y, dtype=np.float64).reshape(-1):
        e_c = float(10.0 ** y)
        cand = min(candidate_lines, key=lambda c: abs(float(c["energy_mev"]) - e_c))
        m = coarse_by_label.get(cand["label"])
        if m is None:
            r = measure_line_centroid(
                E_sorted, float(cand["energy_mev"]),
                [(c["label"], float(c["energy_mev"])) for c in candidate_lines], resolution_ev,
            )
            count = float(r["n_line"]) if r["verdict"] == "ok" else 0.0
            m = {"label": cand["label"], "origin": cand.get("origin", ""),
                 "candidate_energy_mev": float(cand["energy_mev"]), "count": count}
        matched.append(m)

    log_edges = np.linspace(spectrum_log_range[0], spectrum_log_range[1], spectrum_bins + 1)
    logE_all = np.log10(E)
    particle_names = prep["features"]["ParticleName"].to_numpy()
    spectrum = {
        "log_edges": log_edges.tolist(),
        "combined": np.histogram(logE_all, bins=log_edges)[0].tolist(),
        "by_type": {
            str(name): np.histogram(logE_all[particle_names == name], bins=log_edges)[0].tolist()
            for name in np.unique(particle_names)
        },
    }

    feature_pack = build_feature_dataframe(
        prep, energy_binning_mode="log_fixed_count", n_bins=energy_bins,
        geometry_transform="quantile_u_r_u_v_phi_r_phi_v", n_quantiles=n_quantiles,
        random_state=random_state, energy_transform="log10",
    )
    E_full = feature_pack["filtered_prep"]["features"]["Energy"].to_numpy()
    gate_targets = build_gate_targets(
        E_full, feature_pack["energy_bins"], matched,
        bandwidth_mode="resolution", bandwidth_fwhm_mev=resolution_ev * 1e-6,
    )
    feat = feature_pack["feat"].copy()
    for j in range(gate_targets.shape[1]):
        feat[f"gate_target_{j}"] = gate_targets[:, j]
    feat["energy_mev_physical"] = E_full
    cont_cols = ("u_r_q", "u_v_q", "phi_r_q", "phi_v_q", "energy_y") + tuple(
        f"gate_target_{j}" for j in range(gate_targets.shape[1])) + ("energy_mev_physical",)
    dataset_pack = filter_particle_types_continuous_geometry(
        feat=feat, prob_threshold=1e-5, cont_cols=cont_cols)
    split_pack = split_feature_data(dataset_pack, random_state=random_state)

    n_types = dataset_pack["n_types"]
    idx_to_type = dataset_pack["idx_to_type"]
    zone_labels = ["continuum"] + [m["label"] for m in matched]

    model = load_task_adaptive_model_for_generation(
        save_dir=save_dir, model_name=model_name, model_config=model_config,
        n_types=n_types, type_weights=None, radius=radius, verbose=0)
    if not hasattr(model, "decoder_deep_trunk") or not hasattr(model, "position_branch"):
        raise TypeError(
            f"{model_name} does not look like a v0.8 mixture-energy model "
            "(no decoder_deep_trunk/position_branch to trace)."
        )

    idx_test = split_pack["idx_test"]
    X_cont_test = dataset_pack["X_cont_raw"][idx_test]
    y_type_test = dataset_pack["y_type"][idx_test]

    types_data = compute_full_circuit_trace(
        model, model_config, X_cont_test, y_type_test, idx_to_type, n_zones=len(zone_labels),
        n_events_per_type=n_events_per_type, n_baselines=n_baselines, n_steps=n_steps,
        random_state=random_state,
    )
    lines = [{"label": m["label"], "energy_mev": m["candidate_energy_mev"]} for m in matched]
    type_order = [idx_to_type[t] for t in range(n_types) if idx_to_type[t] in types_data]

    html_doc = render_full_circuit_html(
        types_data, zone_labels, lines, spectrum, type_order, source_label=model_name)

    if output_html is None:
        output_html = os.path.join(save_dir, f"{model_name}_full_circuit.html")
    outdir = os.path.dirname(output_html)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(output_html, "w") as f:
        f.write(html_doc)

    print(f"Saved full circuit trace to: {output_html}")
    return output_html
