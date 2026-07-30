"""
Gradient-attribution "circuit" trace for the v0.8 mixture-energy model
(CVAE_MixEnergy_ContPhi_TaskAdaptive and compatible heads).

For one real held-out event per particle type, this traces
gradient x activation attribution through every stage the event actually
passes through: the encoder, the latent z, the decoder stem, the deep
trunk, and all three heads (energy, position, direction) - not just the
final gate logits. The attribution target is the model's own per-event
reconstruction loss (`model._reconstruction_terms`), so every branch gets
a genuine, nonzero attribution instead of only the branch that happens to
feed the term you picked.

Two layers:

- `compute_full_circuit_trace` runs the model and returns plain
  Python/NumPy data - no HTML, no plotting. Useful on its own for
  notebook inspection.
- `render_full_circuit_html` turns that data (plus a real marginal energy
  spectrum) into a self-contained, interactive HTML document: one column
  of activation-colored units per layer, gradient-ranked "top-k" rings and
  wires with an adjustable percentile, and a terminal panel showing the
  real combined energy spectrum with the selected type's own contribution
  shaded on top.

`save_full_circuit` is the convenience wrapper for the common case: point
it at a trained checkpoint plus the real detector table it was trained on,
and it rebuilds the held-out test split, runs the trace for every type,
and writes the HTML.

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


def compute_full_circuit_trace(model, model_config, X_cont_test, y_type_test, idx_to_type, n_zones):
    """Trace one real held-out event per type through the whole v0.8 mixture
    model, with gradient x activation attribution at every stage.

    For each type with at least one held-out event, the representative
    event is the first one whose true gate zone is a pinned line if any
    exist for that type (so a real line hit is shown, not just the
    continuum that dominates every type's population), otherwise the
    first held-out event of that type. `z = z_mean` (posterior mean, no
    reparameterization noise), so the trace is a deterministic
    reconstruction, not a generation sample.

    The attribution target is `model._reconstruction_terms`'s per-event
    loss - the model's own training objective - not just the gate logit,
    so the position and direction branches (which have no gradient path
    to the gate at all: `energy_branch` reads the decoder stem directly,
    bypassing the deep trunk that feeds them) get genuine, nonzero
    attribution too.

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
        `"hidden"` is read, to know where the encoder's hidden stack ends
        and its z_mean/z_logvar heads begin).

    X_cont_test : np.ndarray, shape (n_test, 5 + n_zones + 1)
        Held-out continuous features, column layout
        `[u_r_q, u_v_q, phi_r_q, phi_v_q, energy_y, gate_target_0..n_zones-1,
        energy_mev_physical]` - the same layout `_reconstruction_terms`
        expects, plus one trailing column with the real physical energy in
        MeV (not fed to the model) for the spectrum marker.

    y_type_test : np.ndarray, shape (n_test,)
        Integer type index per row, aligned with `X_cont_test`.

    idx_to_type : dict[int, str]
        Type index -> name, e.g. `{0: "gamma", 1: "e-"}`.

    n_zones : int
        1 + number of pinned lines (continuum + lines).

    Returns
    -------
    dict[str, dict]
        One entry per type that had at least one held-out event:
        `{"stages": {stage_name: {"width": int, "activation": [float, ...],
        "rank": [int, ...]}}, "gate_probs": [float, ...], "true_zone": int,
        "pred_zone": int, "rec_loss": float, "ur_true", "ur_pred", "uv_true",
        "uv_pred", "energy_y_true", "energy_mev_true",
        "log10_energy_mev_true"}`. `stages` is ordered encoder -> z -> stem
        -> trunk -> branches, matching the model's actual data flow.
        `"rank"` is the full permutation of unit indices, highest
        |activation x gradient| first - a percentile-based "top-k" set can
        be recovered as `rank[:k]` for any k, which is what
        `render_full_circuit_html`'s slider does.
    """
    n_hidden = len(model_config["hidden"])
    n_types = len(idx_to_type)

    results = {}
    for t in sorted(idx_to_type):
        where = np.nonzero(y_type_test == t)[0]
        if where.size == 0:
            continue
        true_zones_all = np.argmax(X_cont_test[where, 5:5 + n_zones], axis=1)
        line_where = where[true_zones_all != 0]
        pick = line_where[0] if line_where.size > 0 else where[0]

        row = X_cont_test[pick:pick + 1].astype(np.float32)
        cond = tf.one_hot([t], n_types)
        y_true = tf.constant(row[:, :5 + n_zones])
        x_in = tf.concat([y_true, cond], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            enc_full = _walk(model.encoder, x_in, limit=2 * n_hidden)
            enc_post = enc_full[1::2]
            for a in enc_post:
                tape.watch(a)
            z_mean = model.encoder.get_layer("z_mean")(enc_full[-1])
            z = z_mean
            tape.watch(z)

            base = tf.concat([z, cond], axis=1)

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
            target = -rec_per[0]

        def grad_attr(act):
            g = tape.gradient(target, act)
            a = act[0].numpy()
            gg = g[0].numpy() if g is not None else np.zeros_like(a)
            return a, a * gg

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

        stage_data = {}
        for name, act in stage_acts.items():
            a_np, attr = grad_attr(act)
            rank = np.argsort(attr)[::-1]
            stage_data[name] = {
                "width": int(a_np.size),
                "activation": [round(float(v), 4) for v in a_np],
                "rank": [int(i) for i in rank],
            }

        gate_probs = tf.nn.softmax(gate_logits)[0].numpy()
        true_zone = int(np.argmax(row[0, 5:5 + n_zones]))
        pred_zone = int(np.argmax(gate_probs))
        e_phys = float(row[0, 5 + n_zones])

        results[idx_to_type[t]] = {
            "stages": stage_data,
            "gate_probs": [round(float(x), 6) for x in gate_probs],
            "true_zone": true_zone, "pred_zone": pred_zone,
            "rec_loss": round(float(rec_per[0].numpy()), 4),
            "ur_true": round(float(row[0, 0]), 4), "ur_pred": round(float(ur_mu[0, 0].numpy()), 4),
            "uv_true": round(float(row[0, 1]), 4), "uv_pred": round(float(uv_mu[0, 0].numpy()), 4),
            "energy_y_true": round(float(row[0, 4]), 4),
            "energy_mev_true": e_phys,
            "log10_energy_mev_true": round(float(np.log10(max(e_phys, 1e-12))), 4),
        }
        del tape

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
    is drawn as a column of unit circles colored by activation strength;
    an adjustable "highlight top N%" slider rings the units the event's
    loss depends on most and draws attribution wires between consecutive
    stages' highlighted sets. The gate node additionally rings the true
    and predicted zone. Switching particle type (buttons) or the slider
    re-renders live from the embedded data - no server, no rebuild.

    Parameters
    ----------
    types_data : dict[str, dict]
        Output of `compute_full_circuit_trace`.

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
        Type names in the order/selection the type buttons should show,
        e.g. `list(types_data)`.

    source_label : str
        Short name shown in the title and spectrum caption, e.g. "CR" or
        "K-40 (Small)".

    type_colors : dict[str, str] or None
        Optional `{type_name: "#rrggbb"}` override. Types not in the dict
        (or when the dict is None) get an accent color from a built-in
        8-color palette, cycling by position in `type_order`.

    title : str or None
        Optional heading override. Defaults to
        "The circuit behind one real {source_label} event".

    Returns
    -------
    str
        A complete, self-contained HTML fragment (own `<style>` and
        `<script>`, no external requests, no host stylesheet dependency).

    Raises
    ------
    ValueError
        If `type_order` is empty, or references a type missing from
        `types_data`.
    """
    if not type_order:
        raise ValueError("type_order must be non-empty.")
    missing = [t for t in type_order if t not in types_data]
    if missing:
        raise ValueError(f"type_order references types missing from types_data: {missing}")

    type_colors = dict(type_colors or {})
    for i, t in enumerate(type_order):
        type_colors.setdefault(t, _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)])

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

    lane_y = {"top": 200, "mid": 500, "bot": 800}
    x_spacing = 100
    x0 = 110
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
    svg_h = 1000

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
        f'<title id="circuit-title">MAGI full-circuit trace for {source_label}: encoder, decoder '
        f'trunk, and all three branches for one real held-out event</title>'
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
    parts.append(f'<g fill="none" stroke="var(--border-strong)" stroke-width="1.5">{"".join(bb)}</g>')

    for name, info in stage_units.items():
        label = "gate" if name == "gate" else short_label(name)
        sub = "energy_gate_head" if name == "gate" else sub_label(name)
        top_y = min(p[1] for p in info["pts"]) - 20
        parts.append(f'<text x="{info["x"]}" y="{top_y - 12}" text-anchor="middle" class="stage-sub">{sub}</text>')
        parts.append(f'<text x="{info["x"]}" y="{top_y}" text-anchor="middle" class="stage-label">{label}</text>')
        parts.append(
            f'<text x="{info["x"]}" y="{max(p[1] for p in info["pts"]) + 18}" '
            f'text-anchor="middle" class="stage-width">{info["width"]} units</text>'
        )

    parts.append(f'<text x="60" y="{lane_y["top"] - 38}" class="lane-title">energy path</text>')
    parts.append(f'<text x="60" y="{lane_y["mid"] - 38}" class="lane-title">encoder -&gt; z -&gt; decoder trunk -&gt; position path</text>')
    parts.append(f'<text x="60" y="{lane_y["bot"] - 38}" class="lane-title">direction path</text>')

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
        f'stroke="var(--border-strong)" stroke-width="1.5"/>'
    )

    for name, info in stage_units.items():
        g = [f'<g id="units-{name}">']
        for i, (px, py) in enumerate(info["pts"]):
            r = max(1.1, min(5.0, info["pitch"] * 0.42))
            g.append(f'<circle id="u-{name}-{i}" cx="{px:.2f}" cy="{py:.2f}" r="{r:.2f}" class="unit"/>')
        g.append("</g>")
        parts.append("".join(g))

    parts.append('<g id="connectors" fill="none"></g>')
    if pos_chain:
        pos_last = stage_units[pos_chain[-1]]
        pos_bottom = max(p[1] for p in pos_last["pts"]) + 40
        parts.append(f'<g id="pos-head" transform="translate({pos_last["x"]},{pos_bottom})"></g>')
    if dir_chain:
        dir_last = stage_units[dir_chain[-1]]
        dir_bottom = max(p[1] for p in dir_last["pts"]) + 40
        parts.append(f'<g id="dir-head" transform="translate({dir_last["x"]},{dir_bottom})"></g>')

    gate_pts = stage_units["gate"]["pts"]
    for i, lbl in enumerate(zone_labels):
        px, py = gate_pts[i]
        short = lbl if len(lbl) <= 16 else lbl[:15] + "."
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
        f'<g><text x="{(spec_x0 + spec_x1) / 2:.0f}" y="{spec_y0 - 30}" text-anchor="middle" '
        f'class="stage-label">real {source_label} energy spectrum</text>'
        f'<text x="{(spec_x0 + spec_x1) / 2:.0f}" y="{spec_y0 - 14}" text-anchor="middle" '
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
    for logv, label in tick_defs:
        if logv < log_min or logv > log_max:
            continue
        tx = sx(logv)
        parts.append(f'<line x1="{tx:.1f}" y1="{spec_y1}" x2="{tx:.1f}" y2="{spec_y1 + 6}" stroke="var(--text-muted)" stroke-width="1"/>')
        parts.append(f'<text x="{tx:.1f}" y="{spec_y1 + 20}" text-anchor="middle" class="axis-tick">{label}</text>')
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
    parts.append(f'<line id="spec-marker" x1="0" y1="{spec_y0}" x2="0" y2="{spec_y1}" stroke-width="2" stroke-dasharray="5 4" visibility="hidden"/>')
    parts.append('<text id="spec-marker-label" class="marker-label" visibility="hidden"></text>')
    parts.append(
        f'<rect x="{spec_x0}" y="{spec_y0}" width="{spec_x1 - spec_x0}" height="{spec_y1 - spec_y0}" '
        f'fill="none" stroke="var(--border-strong)" stroke-width="1"/>'
    )
    parts.append("</svg>")
    svg_body = "".join(parts)

    quantized = {}
    for tname, rec in types_data.items():
        stages_q = {}
        for name, stage in rec["stages"].items():
            acts = stage["activation"]
            lo, hi = min(acts), max(acts)
            span = hi - lo if hi > lo else 1e-9
            stages_q[name] = {
                "act100": [round((a - lo) / span * 100) for a in acts],
                "rank": stage["rank"],
            }
        rec_q = dict(rec)
        rec_q["stages"] = stages_q
        quantized[tname] = rec_q

    data = {
        "zone_labels": zone_labels,
        "types": quantized,
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
    heading = title or f"The circuit behind one real {source_label} event"

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
.legend .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }}
.grad-bar {{ display:inline-block; width:90px; height:9px; border-radius:4px; vertical-align:-1px; margin: 0 6px; background: linear-gradient(90deg,#1a1523,#5b3a7a,#b8434f,#e8912a,#f7e56a); }}
.pct-control {{ display: flex; align-items: center; gap: 8px; }}
.pct-control input[type=range] {{ width: 140px; }}
.pct-control .pct-val {{ font-weight: 500; color: var(--text-primary); min-width: 34px; display: inline-block; }}
.svg-scroll {{ overflow-x: auto; background: var(--surface-1); border-radius: 12px; border: 0.5px solid var(--border); padding: 10px 4px; }}
#circuit-svg {{ display: block; margin: 0 auto; min-width: {svg_w}px; }}
#circuit-svg .unit {{ fill: #333; stroke: none; }}
#circuit-svg .unit.top {{ stroke: #E24B4A; stroke-width: 1.4px; }}
#circuit-svg .stage-label {{ font-size: 12px; font-weight: 500; fill: var(--text-primary); }}
#circuit-svg .stage-sub, #circuit-svg .stage-width {{ font-size: 9.5px; fill: var(--text-muted); }}
#circuit-svg .lane-title {{ font-size: 11px; fill: var(--text-secondary); font-weight: 500; }}
#circuit-svg .zone-tick {{ font-size: 9.5px; fill: var(--text-secondary); dominant-baseline: middle; }}
#circuit-svg .axis-tick {{ font-size: 10px; fill: var(--text-muted); }}
#circuit-svg .line-tick {{ font-size: 9.5px; fill: var(--text-secondary); }}
#circuit-svg .marker-label {{ font-size: 11px; font-weight: 500; }}
#circuit-svg .backbone {{ opacity: 0.9; }}
#circuit-svg .t {{ font-size: 12px; font-weight: 500; }}
#circuit-svg .ts {{ font-size: 9.5px; }}
#circuit-svg .head-row {{ font-size: 11px; }}
#circuit-svg .head-label {{ fill: var(--text-muted); }}
#circuit-svg .head-val {{ fill: var(--text-primary); font-weight: 500; }}
#spec-combined {{ fill: var(--border-strong); opacity: 0.55; stroke: var(--text-muted); stroke-width: 1; }}
#spec-type {{ opacity: 0.5; stroke-width: 1.6; }}
.stats-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
.stat-card {{ background: var(--surface-1); border-radius: var(--radius, 8px); padding: 10px 14px; min-width: 140px; }}
.stat-card .label {{ font-size: 11px; color: var(--text-muted); margin-bottom: 3px; }}
.stat-card .value {{ font-size: 15px; font-weight: 500; }}
.stat-card .value.warn {{ color: var(--text-danger); }}
.stat-card .value.ok {{ color: var(--text-success); }}
.sr-only {{ position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); }}
</style>
<div class="circuit-wrap">
  <h1>{heading}</h1>
  <p class="subtitle">Gradient x activation attribution through every stage - encoder, latent <i>z</i>, decoder stem,
  deep trunk, and all three heads (energy, position, direction) - for one real held-out test event per particle type.
  z = z<sub>mean</sub> (posterior mean, deterministic): this is reconstruction, not blind generation. Circle fill =
  activation strength within that layer; red rings + red wires = the top N% of units by |activation x gradient| on the
  model's own reconstruction loss (N adjustable below), i.e. the units this event's loss actually depends on.</p>
  <div class="top-row">
    <div class="controls" role="group" aria-label="select particle type">{buttons}</div>
    <button class="theme-toggle" type="button" onclick="toggleTheme()" id="theme-toggle-btn">Dark mode</button>
  </div>
  <div class="legend">
    <span><span class="grad-bar" role="img" aria-label="activation color scale from low to high"></span>activation: low to high (per layer)</span>
    <span><span class="swatch" style="background:#E24B4A;border:1.4px solid #E24B4A"></span><span id="legend-top-label">top-5% unit</span> (drives this event's loss)</span>
    <span><span style="display:inline-block;width:22px;height:2px;background:#E24B4A;vertical-align:2px;margin-right:5px"></span>attribution wire</span>
    <span class="pct-control"><label for="pct-slider">highlight top</label><input type="range" id="pct-slider" min="1" max="30" value="5" step="1" oninput="setPct(this.value)"><span class="pct-val" id="pct-val">5%</span></span>
  </div>
  <div class="svg-scroll">{svg_body}</div>
  <div class="stats-row" id="stats-row"></div>
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

function clearGroup(g) {{ while (g.firstChild) g.removeChild(g.firstChild); }}

function unitPos(stage, i) {{
  const el = document.getElementById(`u-${{stage}}-${{i}}`);
  return [parseFloat(el.getAttribute('cx')), parseFloat(el.getAttribute('cy'))];
}}

function drawBezier(g, x1, y1, x2, y2, color, width, opacity) {{
  const mx = (x1 + x2) / 2;
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', `M ${{x1}} ${{y1}} C ${{mx}} ${{y1}} ${{mx}} ${{y2}} ${{x2}} ${{y2}}`);
  p.setAttribute('stroke', color);
  p.setAttribute('stroke-width', width);
  p.setAttribute('opacity', opacity);
  g.appendChild(p);
}}

let currentType = DATA.type_order[0];
let currentPct = 5;

function toggleTheme() {{
  const wrap = document.querySelector('.circuit-wrap');
  const dark = wrap.classList.toggle('theme-dark');
  document.getElementById('theme-toggle-btn').textContent = dark ? 'Light mode' : 'Dark mode';
}}

function setPct(v) {{
  currentPct = parseFloat(v);
  document.getElementById('pct-val').textContent = `${{v}}%`;
  document.getElementById('legend-top-label').textContent = `top-${{v}}% unit`;
  renderType(currentType);
}}

function topKSet(stage) {{
  const k = Math.max(1, Math.round(stage.rank.length * currentPct / 100));
  return new Set(stage.rank.slice(0, k));
}}

function renderType(typeName) {{
  currentType = typeName;
  const rec = DATA.types[typeName];
  const color = LAYOUT.type_color[typeName];

  document.querySelectorAll('.type-btn').forEach(b => b.classList.toggle('active', b.dataset.type === typeName));

  for (const [name, info] of Object.entries(LAYOUT.stage_units)) {{
    if (name === 'gate') continue;
    const stage = rec.stages[name];
    const acts = stage.act100;
    const topSet = topKSet(stage);
    for (let i = 0; i < acts.length; i++) {{
      const el = document.getElementById(`u-${{name}}-${{i}}`);
      if (!el) continue;
      el.setAttribute('fill', magma(acts[i] / 100));
      el.classList.toggle('top', topSet.has(i));
    }}
  }}
  {{
    const probs = rec.gate_probs;
    const hi = Math.max(...probs);
    for (let i = 0; i < probs.length; i++) {{
      const el = document.getElementById(`u-gate-${{i}}`);
      if (!el) continue;
      el.setAttribute('fill', magma(hi > 0 ? probs[i] / hi : 0));
      el.classList.toggle('top', i === rec.pred_zone || i === rec.true_zone);
      el.setAttribute('stroke', i === rec.true_zone && i !== rec.pred_zone ? '#BA7517' : (i === rec.pred_zone ? '#E24B4A' : 'none'));
      el.setAttribute('stroke-width', i === rec.true_zone || i === rec.pred_zone ? '2' : '0');
      el.setAttribute('stroke-dasharray', i === rec.true_zone && i !== rec.pred_zone ? '2 2' : 'none');
    }}
  }}

  const connG = document.getElementById('connectors');
  clearGroup(connG);
  for (const [a, b] of LAYOUT.edges) {{
    if (b === 'gate') {{
      const srcTop = topKSet(rec.stages[a]);
      const targets = new Set([rec.pred_zone, rec.true_zone]);
      for (const si of srcTop) {{
        for (const ti of targets) {{
          const [x1, y1] = unitPos(a, si), [x2, y2] = unitPos('gate', ti);
          drawBezier(connG, x1, y1, x2, y2, ti === rec.true_zone && ti !== rec.pred_zone ? '#BA7517' : '#E24B4A', 1, 0.45);
        }}
      }}
      continue;
    }}
    const srcTop = topKSet(rec.stages[a]), dstTop = topKSet(rec.stages[b]);
    for (const si of srcTop) {{
      for (const ti of dstTop) {{
        const [x1, y1] = unitPos(a, si), [x2, y2] = unitPos(b, ti);
        drawBezier(connG, x1, y1, x2, y2, '#E24B4A', 1, 0.4);
      }}
    }}
  }}

  const posHead = document.getElementById('pos-head');
  if (posHead) {{
    posHead.innerHTML =
      `<text x="0" y="0" text-anchor="middle" class="head-row"><tspan class="head-label">u_r head</tspan></text>` +
      `<text x="0" y="15" text-anchor="middle" class="head-row"><tspan class="head-label">true </tspan><tspan class="head-val">${{rec.ur_true.toFixed(3)}}</tspan><tspan class="head-label"> pred </tspan><tspan class="head-val">${{rec.ur_pred.toFixed(3)}}</tspan></text>`;
  }}
  const dirHead = document.getElementById('dir-head');
  if (dirHead) {{
    dirHead.innerHTML =
      `<text x="0" y="0" text-anchor="middle" class="head-row"><tspan class="head-label">u_v head</tspan></text>` +
      `<text x="0" y="15" text-anchor="middle" class="head-row"><tspan class="head-label">true </tspan><tspan class="head-val">${{rec.uv_true.toFixed(3)}}</tspan><tspan class="head-label"> pred </tspan><tspan class="head-val">${{rec.uv_pred.toFixed(3)}}</tspan></text>`;
  }}

  const spec = LAYOUT.spec;
  function sx(logE) {{ return spec.x0 + (logE - spec.log_min) / (spec.log_max - spec.log_min) * (spec.x1 - spec.x0); }}
  function sy(count) {{ const v = spec.y_log_max > 0 ? Math.log10(count + 1) / spec.y_log_max : 0; return spec.y1 - v * (spec.y1 - spec.y0); }}
  const hist = DATA.spectrum.by_type[typeName] || DATA.spectrum.combined.map(() => 0);
  const edges = DATA.spectrum.log_edges;
  let d = `M ${{sx(edges[0]).toFixed(1)}} ${{spec.y1}} `;
  for (let i = 0; i < hist.length; i++) {{
    const y = sy(hist[i]);
    d += `L ${{sx(edges[i]).toFixed(1)}} ${{y.toFixed(1)}} L ${{sx(edges[i+1]).toFixed(1)}} ${{y.toFixed(1)}} `;
  }}
  d += `L ${{sx(edges[edges.length-1]).toFixed(1)}} ${{spec.y1}} Z`;
  const specType = document.getElementById('spec-type');
  specType.setAttribute('d', d);
  specType.setAttribute('fill', color);
  specType.setAttribute('stroke', color);

  const marker = document.getElementById('spec-marker');
  const mx = sx(rec.log10_energy_mev_true);
  marker.setAttribute('x1', mx.toFixed(1));
  marker.setAttribute('x2', mx.toFixed(1));
  marker.setAttribute('stroke', color);
  marker.setAttribute('visibility', 'visible');
  const mlabel = document.getElementById('spec-marker-label');
  const eStr = rec.energy_mev_true < 1e-2 ? (rec.energy_mev_true*1000).toFixed(1) + ' keV' : rec.energy_mev_true.toFixed(3) + ' MeV';
  mlabel.setAttribute('x', (mx + 6).toFixed(1));
  mlabel.setAttribute('y', spec.y0 + 14);
  mlabel.setAttribute('fill', color);
  mlabel.setAttribute('visibility', 'visible');
  mlabel.textContent = `this event: ${{eStr}}`;

  const trueZoneLabel = DATA.zone_labels[rec.true_zone];
  const predZoneLabel = DATA.zone_labels[rec.pred_zone];
  const misrouted = rec.true_zone !== rec.pred_zone;
  document.getElementById('stats-row').innerHTML = `
    <div class="stat-card"><div class="label">true zone</div><div class="value">${{trueZoneLabel}}</div></div>
    <div class="stat-card"><div class="label">predicted zone</div><div class="value ${{misrouted ? 'warn' : 'ok'}}">${{predZoneLabel}}${{misrouted ? ' (misrouted)' : ' (correct)'}}</div></div>
    <div class="stat-card"><div class="label">gate confidence (true zone)</div><div class="value">${{(rec.gate_probs[rec.true_zone]*100).toFixed(1)}}%</div></div>
    <div class="stat-card"><div class="label">reconstruction loss (this event)</div><div class="value">${{rec.rec_loss.toFixed(2)}}</div></div>
  `;
  document.getElementById('sr-summary').textContent =
    `Showing ${{typeName}}: true zone ${{trueZoneLabel}}, predicted ${{predZoneLabel}}, event energy ${{eStr}}.`;
}}

renderType(currentType);
</script>
"""
    return html_doc


def save_full_circuit(save_dir, model_name, training_data_filepath, candidate_lines_file,
                       center, radius, resolution_ev=4.0, output_html=None, random_state=42,
                       energy_bins=512, n_quantiles=10000, spectrum_bins=140,
                       spectrum_log_range=(-3.5, 5.9)):
    """Load a v0.8 mixture checkpoint plus the real detector table it was
    trained on, rebuild its held-out test split, and write the full-circuit
    trace (every type, every stage) as a standalone HTML file.

    This re-derives the training-time preprocessing (line matching, gate
    targets, quantile/log10 transforms, the train/test split) from the raw
    detector table rather than trusting anything cached in the checkpoint,
    since the checkpoint only stores the trained weights and config - not a
    held-out sample to trace. Uses the same `random_state` as
    `tools/run_v0_8_real.py` by default so the picked events are
    reproducible against that run's own artifacts.

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
        Passed to `build_feature_dataframe`'s quantile transform and to
        `split_feature_data`, so the held-out split (and therefore which
        event gets picked per type) is reproducible.

    energy_bins, n_quantiles : int
        Passed through to `build_feature_dataframe`.

    spectrum_bins : int
        Number of log10(E/MeV) bins for the terminal spectrum panel.

    spectrum_log_range : tuple[float, float]
        (min, max) log10(E/MeV) range for the spectrum panel.

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
        model, model_config, X_cont_test, y_type_test, idx_to_type, n_zones=len(zone_labels))
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
