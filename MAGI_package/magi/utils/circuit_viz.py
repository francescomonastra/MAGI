"""
Interactive HTML "circuit" visualizations of the v0.8 mixture head's
conditional routing, for the physical picture that a static plot cannot
convey well: which particle types the model actually sends toward the
continuum vs. a pinned spectral line, and how sharply that differs by type.

Self-contained: the returned HTML embeds its own <style> block (CSS custom
properties + the handful of color/text classes used below) rather than
depending on any host stylesheet, so the file opens correctly in a plain
browser, not only inside a tool that injects design tokens.
"""
import html
import json
import os


_STYLE = """
:root {
  --circuit-text: #1a1a18; --circuit-text-2: #5f5e5a; --circuit-text-3: #888780;
  --circuit-border: #d3d1c7; --circuit-border-strong: #b4b2a9;
  --circuit-surface-1: #f1efe8; --circuit-accent: #7f77dd;
  --circuit-teal-50: #e1f5ee; --circuit-teal-600: #0f6e56; --circuit-teal-800: #085041;
  --circuit-purple-50: #eeedfe; --circuit-purple-400: #7f77dd; --circuit-purple-800: #3c3489;
  --circuit-gray-50: #f1efe8;
}
@media (prefers-color-scheme: dark) {
  :root {
    --circuit-text: #f1efe8; --circuit-text-2: #b4b2a9; --circuit-text-3: #888780;
    --circuit-border: #444441; --circuit-border-strong: #5f5e5a;
    --circuit-surface-1: #2c2c2a; --circuit-accent: #afa9ec;
    --circuit-teal-50: #085041; --circuit-teal-600: #5dcaa5; --circuit-teal-800: #e1f5ee;
    --circuit-purple-50: #26215c; --circuit-purple-400: #afa9ec; --circuit-purple-800: #cecbf6;
    --circuit-gray-50: #2c2c2a;
  }
}
.magi-circuit { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 720px;
  margin: 0 auto; color: var(--circuit-text); }
.magi-circuit svg text { fill: var(--circuit-text); font-size: 12px; }
.magi-circuit svg text.mc-t2 { fill: var(--circuit-text-2); font-size: 11px; }
.magi-circuit .mc-teal rect, .magi-circuit .mc-teal circle { fill: var(--circuit-teal-50); stroke: var(--circuit-teal-600); }
.magi-circuit .mc-teal text { fill: var(--circuit-teal-800); }
.magi-circuit .mc-purple rect { fill: var(--circuit-purple-50); stroke: var(--circuit-purple-400); }
.magi-circuit .mc-purple text { fill: var(--circuit-purple-800); }
.magi-circuit .mc-gray rect { fill: var(--circuit-gray-50); stroke: var(--circuit-border-strong); }
.magi-circuit button.mc-btn { display: flex; flex-direction: column; align-items: center;
  gap: 2px; padding: 6px 14px; font: inherit; font-size: 13px; font-weight: 500;
  color: var(--circuit-text); background: transparent; cursor: pointer;
  border: 0.5px solid var(--circuit-border); border-radius: 8px; }
.magi-circuit button.mc-btn:hover { background: var(--circuit-surface-1); }
.magi-circuit button.mc-btn span.mc-sub { font-size: 10px; font-weight: 400; color: var(--circuit-text-2); }
.magi-circuit .mc-row { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; margin-bottom: 1.25rem; }
.magi-circuit .mc-legend { display: flex; gap: 16px; justify-content: center; font-size: 11px;
  color: var(--circuit-text-2); margin-top: 0.5rem; }
"""


def _short_label(name, max_len=10):
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def render_routing_circuit_html(
    idx_to_type,
    type_probs,
    zone_probs,
    zone_labels,
    title="MAGI conditional routing circuit",
):
    """Build a self-contained interactive HTML "circuit" diagram of how the
    v0.8 mixture head routes each particle type between the continuum and its
    pinned spectral lines.

    Renders the generative path -- one-hot condition -> learned prior p(z|c)
    -> z -> decoder -> gate -> {continuum, line_1..line_L} -- as an SVG
    circuit, with a button per particle type that swaps in that type's real
    routing fractions and highlights the corresponding one-hot wire. This
    plots `zone_probs`, the per-type empirical routing frequency measured
    from training data and used by generate() to sample the mixture
    component at generation time (see
    CVAE_MixEnergy_ContPhi_TaskAdaptive.generate) -- it does not run the
    model itself, so it works from a checkpoint's saved config alone.

    Parameters
    ----------
    idx_to_type : dict[int, str]
        Type index to particle name, e.g. `preprocessing_metadata["idx_to_type"]`.

    type_probs : array-like of float, length n_types
        Marginal population fraction of each type, e.g.
        `preprocessing_metadata["type_probs"]`.

    zone_probs : array-like of float, shape (n_types, n_lines + 1)
        Per-type routing distribution over [continuum, line_1, ..., line_L],
        e.g. `model_config["zone_probs"]` from a checkpoint trained with
        `prior_zone_conditioning=True`.

    zone_labels : list[str], length n_lines + 1
        Display label per zone column, first entry "continuum" followed by
        one label per matched line (energy and transition, e.g.
        "Cu Kα1 · 8.01 keV"). Must be in the same column order as
        `zone_probs`.

    title : str
        Accessible title embedded in the SVG `<title>` and the page `<title>`.

    Returns
    -------
    str
        A complete standalone HTML document (with its own `<style>`, no
        external stylesheet or script dependency) that can be written to
        disk and opened directly in a browser, or embedded in a notebook via
        `IPython.display.HTML`.

    Raises
    ------
    ValueError
        If `zone_probs` is not shaped (n_types, len(zone_labels)), or
        `idx_to_type`/`type_probs` disagree on the number of types.
    """
    n_types = len(idx_to_type)
    if len(type_probs) != n_types:
        raise ValueError(
            f"type_probs has {len(type_probs)} entries but idx_to_type has "
            f"{n_types} types."
        )
    n_zones = len(zone_labels)
    if len(zone_probs) != n_types or any(len(r) != n_zones for r in zone_probs):
        raise ValueError(
            f"zone_probs must have shape ({n_types}, {n_zones}) to match "
            f"idx_to_type and zone_labels; got {len(zone_probs)} rows of "
            f"length {[len(r) for r in zone_probs][:1]}."
        )

    types = [idx_to_type[i] for i in range(n_types)]

    cell_w, cell_h, cell_gap = 74, 36, 8
    row_w = n_types * cell_w + (n_types - 1) * cell_gap
    width = max(680, row_w + 40)
    cx0 = (width - row_w) / 2
    cell_x = [cx0 + i * (cell_w + cell_gap) for i in range(n_types)]
    cell_y = 22
    junction_x, junction_y = width / 2, cell_y + cell_h + 28

    leaf_w, leaf_h, leaf_gap, per_row = 140, 52, 15, 3
    n_rows = -(-n_zones // per_row)
    leaf_top = junction_y + 250
    gate_y = junction_y + 182

    cells_svg, wires_svg = [], []
    for i, t in enumerate(types):
        x = cell_x[i]
        cells_svg.append(
            f'<g id="cell-{i}" class="mc-purple">'
            f'<rect x="{x:.1f}" y="{cell_y}" width="{cell_w}" height="{cell_h}" rx="4"></rect>'
            f'<text x="{x + cell_w / 2:.1f}" y="{cell_y + cell_h / 2 + 4:.1f}" text-anchor="middle">'
            f"{html.escape(_short_label(t))}</text></g>"
        )
        wires_svg.append(
            f'<path id="wire-{i}" d="M{x + cell_w / 2:.1f},{cell_y + cell_h} '
            f"L{x + cell_w / 2:.1f},{cell_y + cell_h + 18} "
            f'L{junction_x:.1f},{junction_y}" fill="none" '
            f'stroke="var(--circuit-border)" stroke-width="1"></path>'
        )

    leaves_svg, fan_svg = [], []
    for j in range(n_zones):
        row, col = divmod(j, per_row)
        this_row_n = min(per_row, n_zones - row * per_row)
        row_w2 = this_row_n * leaf_w + (this_row_n - 1) * leaf_gap
        lx0 = (width - row_w2) / 2
        x = lx0 + col * (leaf_w + leaf_gap)
        y = leaf_top + row * (leaf_h + 14)
        cls = "mc-gray" if j == 0 else "mc-purple"
        leaves_svg.append(
            f'<g id="leaf-{j}" class="{cls}">'
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{leaf_w}" height="{leaf_h}" rx="4"></rect>'
            f'<text x="{x + leaf_w / 2:.1f}" y="{y + 20:.1f}" text-anchor="middle">'
            f"{html.escape(_short_label(zone_labels[j], 20))}</text>"
            f'<text id="pct-{j}" x="{x + leaf_w / 2:.1f}" y="{y + 38:.1f}" '
            f'text-anchor="middle" class="mc-t2">-</text></g>'
        )
        fan_svg.append(
            f'<path d="M{width / 2:.1f},{gate_y + 40} '
            f"C{width / 2:.1f},{gate_y + 64} {x + leaf_w / 2:.1f},{gate_y + 64} "
            f'{x + leaf_w / 2:.1f},{y:.1f}" fill="none" '
            f'stroke="var(--circuit-border-strong)" stroke-width="1" '
            f'marker-end="url(#mc-arr)"></path>'
        )

    total_height = leaf_top + n_rows * (leaf_h + 14) + 30

    geometry_arrow = (
        f'<path d="M{junction_x - 60:.1f},{junction_y + 153} '
        f"C{junction_x - 60:.1f},{junction_y + 176} {junction_x - 170:.1f},{junction_y + 176} "
        f'{junction_x - 170:.1f},{junction_y + 196}" fill="none" '
        f'stroke="var(--circuit-border-strong)" stroke-width="1" marker-end="url(#mc-arr)"></path>'
    )
    gate_arrow = (
        f'<path d="M{junction_x + 60:.1f},{junction_y + 153} '
        f"C{junction_x + 60:.1f},{junction_y + 176} {junction_x + 60:.1f},{junction_y + 176} "
        f'{junction_x + 60:.1f},{gate_y}" fill="none" '
        f'stroke="var(--circuit-border-strong)" stroke-width="1" marker-end="url(#mc-arr)"></path>'
    )

    svg = f"""
<svg viewBox="0 0 {width:.0f} {total_height:.0f}" role="img" style="width:100%; height:auto;">
<title>{html.escape(title)}</title>
<desc>One-hot particle type feeds the learned coupling prior and the decoder;
the decoder's gate routes each event to the continuum or a pinned spectral
line, with routing weights that differ by particle type.</desc>
<defs><marker id="mc-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6"
orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--circuit-text-3)"></path></marker></defs>
<text x="{width / 2:.1f}" y="14" class="mc-t2" text-anchor="middle">one-hot condition c</text>
{''.join(cells_svg)}
{''.join(wires_svg)}
<path d="M{junction_x:.1f},{junction_y} L{junction_x:.1f},{junction_y + 10}" fill="none"
stroke="var(--circuit-border-strong)" stroke-width="1" marker-end="url(#mc-arr)"></path>
<g class="mc-teal"><rect x="{junction_x - 80:.1f}" y="{junction_y + 10}" width="160" height="46" rx="4"></rect>
<text x="{junction_x:.1f}" y="{junction_y + 29:.1f}" text-anchor="middle">prior</text>
<text x="{junction_x:.1f}" y="{junction_y + 45:.1f}" text-anchor="middle" class="mc-t2">p(z | c, zone)</text></g>
<path d="M{junction_x:.1f},{junction_y + 56} L{junction_x:.1f},{junction_y + 68}" fill="none"
stroke="var(--circuit-border-strong)" stroke-width="1" marker-end="url(#mc-arr)"></path>
<g class="mc-teal"><circle cx="{junction_x:.1f}" cy="{junction_y + 84:.1f}" r="15"></circle>
<text x="{junction_x:.1f}" y="{junction_y + 89:.1f}" text-anchor="middle">z</text></g>
<path d="M{junction_x:.1f},{junction_y + 99} L{junction_x:.1f},{junction_y + 111}" fill="none"
stroke="var(--circuit-border-strong)" stroke-width="1" marker-end="url(#mc-arr)"></path>
<g class="mc-teal"><rect x="{junction_x - 90:.1f}" y="{junction_y + 111}" width="180" height="42" rx="4"></rect>
<text x="{junction_x:.1f}" y="{junction_y + 129:.1f}" text-anchor="middle">decoder</text>
<text x="{junction_x:.1f}" y="{junction_y + 145:.1f}" text-anchor="middle" class="mc-t2">[z, c] → outputs</text></g>
{geometry_arrow}
{gate_arrow}
<g class="mc-gray"><rect x="{junction_x - 260:.1f}" y="{junction_y + 196}" width="180" height="44" rx="4"></rect>
<text x="{junction_x - 170:.1f}" y="{junction_y + 214:.1f}" text-anchor="middle">geometry heads</text>
<text x="{junction_x - 170:.1f}" y="{junction_y + 230:.1f}" text-anchor="middle" class="mc-t2">u_r, u_v, φ_r, φ_v</text></g>
<g class="mc-teal"><rect x="{junction_x:.1f}" y="{gate_y}" width="120" height="40" rx="4"></rect>
<text x="{junction_x + 60:.1f}" y="{gate_y + 24:.1f}" text-anchor="middle">gate g(z,c)</text></g>
{''.join(fan_svg)}
{''.join(leaves_svg)}
</svg>
"""

    data = {
        "types": types,
        "typeProbs": [round(float(p) * 100, 4) for p in type_probs],
        "zones": [[round(float(x) * 100, 6) for x in row] for row in zone_probs],
    }

    buttons = "".join(
        f'<button class="mc-btn" data-idx="{i}"><span>{html.escape(_short_label(t))}</span>'
        f'<span class="mc-sub">{data["typeProbs"][i]:.2f}% of events</span></button>'
        for i, t in enumerate(types)
    )

    script = f"""
<script>
(function() {{
  const DATA = {json.dumps(data)};
  function fmt(p) {{
    if (p >= 1) return p.toFixed(2) + '%';
    if (p >= 0.01) return p.toFixed(3) + '%';
    if (p > 0) return p.toFixed(4) + '%';
    return '0%';
  }}
  function select(idx) {{
    const row = DATA.zones[idx];
    row.forEach((v, j) => {{
      const el = document.getElementById('pct-' + j);
      if (el) el.textContent = fmt(v);
      const leaf = document.getElementById('leaf-' + j);
      if (leaf && j > 0) leaf.style.opacity = v > 0.02 ? 1 : 0.4;
    }});
    DATA.types.forEach((t, i) => {{
      const cell = document.getElementById('cell-' + i);
      const wire = document.getElementById('wire-' + i);
      const active = i === idx;
      if (cell) cell.style.opacity = active ? 1 : 0.35;
      if (wire) {{
        wire.setAttribute('stroke', active ? 'var(--circuit-accent)' : 'var(--circuit-border)');
        wire.setAttribute('stroke-width', active ? 2.5 : 1);
      }}
    }});
    document.querySelectorAll('.mc-btn').forEach(b => {{
      const active = Number(b.dataset.idx) === idx;
      b.style.borderColor = active ? 'var(--circuit-accent)' : '';
      b.style.background = active ? 'var(--circuit-surface-1)' : '';
    }});
  }}
  document.querySelectorAll('.mc-btn').forEach(b =>
    b.addEventListener('click', () => select(Number(b.dataset.idx))));
  select(0);
}})();
</script>
"""

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{_STYLE}</style></head>
<body>
<div class="magi-circuit">
<div class="mc-row">{buttons}</div>
{svg}
<div class="mc-legend"><span>zone_probs from the loaded checkpoint's config</span></div>
</div>
{script}
</body></html>
"""


def save_routing_circuit(save_dir, model_name, output_html=None):
    """Load a v0.8 mixture checkpoint's saved config/metadata and write its
    conditional routing circuit as a standalone HTML file.

    Convenience wrapper around render_routing_circuit_html for the common
    case of visualizing an already-trained run without re-deriving its
    zone_probs by hand. Requires the checkpoint to have been trained with
    `prior_zone_conditioning=True` -- config_version 1 checkpoints (or any
    run with prior_zone_conditioning=False) have no zone_probs to plot.

    Parameters
    ----------
    save_dir : str
        Run directory holding `<model_name>_config.json` and
        `<model_name>_metadata.json`.

    model_name : str
        Filename stem used when the run was saved.

    output_html : str or None
        Destination path. None (default) writes to
        `<save_dir>/<model_name>_routing_circuit.html`.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    KeyError
        If the checkpoint's config has no "zone_probs" (prior_zone_conditioning
        was not enabled for this run).
    """
    from ..training.checkpointing import load_json

    model_config = load_json(os.path.join(save_dir, f"{model_name}_config.json"))
    metadata = load_json(os.path.join(save_dir, f"{model_name}_metadata.json"))
    pp = metadata["preprocessing_metadata"]

    if not model_config.get("zone_probs"):
        raise KeyError(
            f"{model_name}_config.json has no zone_probs -- this checkpoint "
            "was not trained with prior_zone_conditioning=True, so there is "
            "no per-type routing distribution to plot."
        )

    idx_to_type = {int(k): v for k, v in pp["idx_to_type"].items()}
    matched = pp.get("matched_lines") or []
    n_lines = len(model_config["zone_probs"][0]) - 1

    if len(matched) == n_lines:
        zone_labels = ["continuum"] + [m["label"] for m in matched]
    else:
        # preprocessing_metadata does not (yet) always persist matched_lines
        # (see tools/run_v0_8_real.py), so the transition name (Cu Kα1 vs.
        # Cu Kβ) can be unrecoverable from a checkpoint alone. line_positions_y
        # IS always saved -- fall back to the pinned energy, which is exact,
        # rather than a meaningless "line 1/2/3".
        import numpy as np

        energies_mev = 10.0 ** np.asarray(model_config["line_positions_y"], dtype=float)
        zone_labels = ["continuum"] + [
            f"line @ {e * 1000:.2f} keV" for e in energies_mev
        ]

    html_doc = render_routing_circuit_html(
        idx_to_type=idx_to_type,
        type_probs=pp["type_probs"],
        zone_probs=model_config["zone_probs"],
        zone_labels=zone_labels,
        title=f"MAGI routing circuit -- {model_name}",
    )

    if output_html is None:
        output_html = os.path.join(save_dir, f"{model_name}_routing_circuit.html")
    outdir = os.path.dirname(output_html)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(output_html, "w") as f:
        f.write(html_doc)

    print(f"Saved routing circuit to: {output_html}")
    return output_html
