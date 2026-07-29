"""Phase A4 (docs/v0.8.2_RoadmapForAdoption.md S5.2b, S6): fraction-vs-error
curve across CR + Small + Torio, to test the "minimum learnable component
fraction" hypothesis and break the CR/Small confound (7 points isn't enough
to tell a fraction effect from a source effect - Torio has yet another
event count and type count, so a genuine fraction trend should continue
through its points too).

Purely a compiler over already-produced tools/acceptance_v0_8.py JSON reports
(no model access, no training) - point it at one JSON per source.

Usage:
  python tools/fraction_vs_error_report.py \
      --reports CR=docs/_data/a4_acceptance_CR.json \
                Small=docs/_data/a4_acceptance_SmallTorio.json \
                Torio=docs/_data/a4_acceptance_SmallTorio.json
"""
import argparse, json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--reports", nargs="+", required=True,
                    help="SOURCE=path/to/acceptance_report.json, repeatable. "
                         "If a JSON covers multiple sources (tools/acceptance_v0_8.py "
                         "--sources A B in one run), pass the same path once per "
                         "source name it contains.")
parser.add_argument("--min-significance", type=float, default=5.0,
                    help="Poisson-sigma floor (docs/v0.8.2_RoadmapForAdoption.md "
                         "S4's restated bar) - lines below this are reported "
                         "separately, not mixed into the trend.")
args = parser.parse_args()

rows = []
for item in args.reports:
    src, _, path = item.partition("=")
    with open(path) as f:
        d = json.load(f)
    s = d["sources"].get(src)
    if s is None:
        raise SystemExit(f"{path} has no source '{src}' (has: {list(d['sources'])})")
    for r in s["recovery"]:
        rec = r.get("recovery_ratio")
        comp_rec = r.get("component_recovery")
        rows.append({
            "source": src, "label": r["label"],
            "real_fraction_line": r.get("real_fraction_line"),
            "recovery_ratio": rec, "component_recovery": comp_rec,
            "significance": r.get("line_significance"),
            "overlaps": r.get("overlaps_lines"),
            "sideband_contaminated": r.get("sideband_contaminated"),
        })

rows = [r for r in rows if r["real_fraction_line"] and r["real_fraction_line"] > 0]
rows.sort(key=lambda r: r["real_fraction_line"])

print(f"{'source':8s} {'line':22s} {'real_fraction':>13s} {'sig':>7s} "
      f"{'recovery':>9s} {'comp_rec':>9s}  note")
for r in rows:
    err = r["recovery_ratio"] if r["recovery_ratio"] is not None else r["component_recovery"]
    err_s = f"{err:9.3f}" if err is not None else "      n/a"
    cr_s = f"{r['component_recovery']:9.3f}" if r["component_recovery"] is not None else "      n/a"
    sig_s = f"{r['significance']:6.1f}s" if r["significance"] is not None else "    n/a"
    note = []
    if r["significance"] is not None and r["significance"] < args.min_significance:
        note.append(f"BELOW {args.min_significance:g}sigma FLOOR")
    if r["sideband_contaminated"]:
        note.append(f"CONTAMINATED by {r['overlaps']}")
    print(f"{r['source']:8s} {r['label']:22s} {r['real_fraction_line']:13.3e} {sig_s} "
          f"{err_s} {cr_s}  {' '.join(note)}")

usable = [r for r in rows
          if (r["significance"] is None or r["significance"] >= args.min_significance)
          and not r["sideband_contaminated"]
          and (r["recovery_ratio"] is not None or r["component_recovery"] is not None)]
if len(usable) >= 3:
    fracs = np.log10([r["real_fraction_line"] for r in usable])
    errs = np.log10([abs(np.log10(r["recovery_ratio"] if r["recovery_ratio"] is not None
                                   else r["component_recovery"])) + 1e-6 for r in usable])
    corr = float(np.corrcoef(fracs, errs)[0, 1])
    print(f"\nCorrelation(log10 real_fraction, log10 |log10(recovery)|) over "
          f"{len(usable)} lines >= {args.min_significance:g}sigma, "
          f"uncontaminated: {corr:+.3f}")
    print("(more negative => stronger support for a minimum-learnable-fraction "
          "trend: rarer lines have larger |log recovery error|)")
    print("\nPer-source breakdown (does the trend hold WITHIN a source, or "
          "only BETWEEN sources - the latter is the confound):")
    for src in sorted(set(r["source"] for r in usable)):
        sub = [r for r in usable if r["source"] == src]
        if len(sub) < 2:
            print(f"  {src}: only {len(sub)} usable line(s) - can't test within-source")
            continue
        f2 = np.log10([r["real_fraction_line"] for r in sub])
        e2 = np.log10([abs(np.log10(r["recovery_ratio"] if r["recovery_ratio"] is not None
                                     else r["component_recovery"])) + 1e-6 for r in sub])
        c2 = float(np.corrcoef(f2, e2)[0, 1]) if len(sub) > 1 else float("nan")
        frac_strs = [f"{r['real_fraction_line']:.2e}" for r in sub]
        print(f"  {src}: n={len(sub)} corr={c2:+.3f} fractions={frac_strs}")
else:
    print(f"\nOnly {len(usable)} usable line(s) (>= {args.min_significance:g}sigma, "
          "uncontaminated) - not enough to fit a trend.")
