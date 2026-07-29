"""Audit the pinned line positions and widths against the real spectrum.

For every line the v0.8 mixture head would pin, this measures - directly from
the raw events, at eV resolution - where the line actually is and how wide it
actually is, and compares that with the catalogue value that would be pinned:

    delta = catalogue - measured centroid,  reported in units of the pinned FWHM

A line pinned more than ~1 FWHM from its real position cannot be reproduced at
all: the generated 4 eV Gaussian and the real peak simply do not overlap. That
was the state of v0.8 before the EADL rebuild (CryoSphere-CR was 4-10 FWHM off
on every fluorescence line), and the wide bin-width recovery window hid it.

Also usable as the confirmation gate for speculative candidates (escape peaks):
pass --candidates <file> --all-candidates to test every catalogue entry, not
just the detected ones, and read the CONFIRMED/absent verdict.

Usage:
  python tools/line_centroid_audit.py --sources CR Small
  python tools/line_centroid_audit.py --sources CR \
      --candidates CandidateLines/..._EADL.json --all-candidates --json logs/audit.json
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse, json, time
import numpy as np
import magi

DEFAULT_CANDIDATES = ("/Volumes/X10Pro/MAGI/CandidateLines/"
    "CANDIDATE_ENERGY_LINES_SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed_EADL.json")

parser = argparse.ArgumentParser()
parser.add_argument("--sources", nargs="+", default=["CR", "Small"])
parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
parser.add_argument("--resolution-ev", type=float, default=4.0,
                    help="detector FWHM the mixture head pins the line widths to")
parser.add_argument("--all-candidates", action="store_true",
                    help="audit every catalogue entry, not only the detected ones "
                         "(use to confirm/reject speculative candidates)")
parser.add_argument("--min-count", type=int, default=100,
                    help="line strength below which a detected line is not fed to "
                         "the mixture head (matches the run scripts' filter)")
parser.add_argument("--fail-fwhm", type=float, default=1.0,
                    help="|catalogue - measured| above this many FWHM is a FAIL")
parser.add_argument("--search-fwhm", type=float, default=20.0,
                    help="half-width of the search window around the catalogue "
                         "energy, in FWHM. Must cover a plausible mis-pin (the "
                         "Bearden/EADL offsets are 4-11 FWHM) without reaching a "
                         "neighbouring line; a centroid that turns out to be closer "
                         "to another catalogue entry is reported as 'shadowed'.")
parser.add_argument("--json", default="")
args = parser.parse_args()

magi.initialize_environment(seed=42, cpu_only=True, quiet=True)

TRAINING_DATA_DIR = "/Volumes/X10Pro/MAGI/TrainingData"
SOURCE_FILES = {
    "CR": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereCR.dat",
    "Small": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereSmall.dat",
    "Torio": f"{TRAINING_DATA_DIR}/alloutputDSCryoSphereTorio.dat",
}
center = (0.0, 0.0, -507.66); R = 100.0

FWHM_MEV = args.resolution_ev * 1e-6

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


def audit_line(E_sorted, E_c, label):
    """Measure the real peak near E_c: centroid, width, line and continuum counts.

    Thin wrapper around magi.measure_line_centroid (promoted from this
    function so the audit tool and the training pipeline's
    confirm_unresolved_candidate_lines share one implementation, including
    the search-window clip that stops a doublet member being reported
    "shadowed" by its own stronger neighbour - see that function's docstring
    for the Cu Kalpha2/Kalpha1 case that motivated it).

    Renames "energy_mev" -> "candidate_mev" and re-derives "strong" as a
    count-only threshold (independent of position accuracy) to keep this
    script's PASS/FAIL logic - which checks position accuracy separately via
    delta_fwhm - unchanged.
    """
    r = magi.measure_line_centroid(
        E_sorted, E_c, candidate_energies, args.resolution_ev,
        search_fwhm=args.search_fwhm, min_count=args.min_count,
        fail_fwhm=args.fail_fwhm,
    )
    out = {"label": label, "candidate_mev": r.pop("energy_mev"), **r}
    if out["verdict"] == "ok":
        out["strong"] = bool(out["n_line"] >= args.min_count)
    return out


candidates = magi.load_candidate_energy_lines(args.candidates)["lines"]
candidate_energies = [(c["label"], float(c["energy_mev"])) for c in candidates]
report = {"candidates_file": args.candidates, "resolution_ev": args.resolution_ev,
          "sources": {}}
overall = True

for name in args.sources:
    log("=" * 62)
    log(f"SOURCE: {name}   (candidates: {os.path.basename(args.candidates)})")
    df = magi.load_detector_table(filepath=SOURCE_FILES[name], sep=r"\s+")
    E = magi.build_physical_features(df, center=center, radius=R)["features"]["Energy"].to_numpy()
    E_sorted = np.sort(E)

    if args.all_candidates:
        targets = [(c["label"], float(c["energy_mev"]), c.get("origin", ""), 0.0)
                   for c in candidates]
    else:
        res = magi.detect_energy_lines(
            E, binning_mode="log_fixed_count", n_bins=1024, prominence_factor=3.0,
            window=5, candidate_lines=candidates, refine_bin_width_mev=FWHM_MEV)
        targets = [(m["label"], float(m["candidate_energy_mev"]), m["origin"],
                    float(m["count"]))
                   for m in res["matched_lines"]]
        log(f"  {len(targets)} matched lines to audit")

    rows = []
    print(f"\n  {'line':20s} {'catalogue':>11s} {'measured':>11s} {'delta':>9s} "
          f"{'FWHM':>7s} {'n_line':>9s} {'width':>8s}  verdict")
    print(f"  {'':20s} {'[keV]':>11s} {'[keV]':>11s} {'[eV]':>9s} {'off':>7s} "
          f"{'':>9s} {'[eV]':>8s}")
    for label, E_c, origin, det_count in targets:
        r = audit_line(E_sorted, E_c, label)
        r["origin"] = origin
        r["detected_count"] = det_count
        # Only lines the pipeline would actually feed to the mixture head
        # (detected with count >= --min-count) can fail the audit; weaker ones
        # are dropped upstream anyway and are reported for information.
        fed_to_model = det_count >= args.min_count

        if r["verdict"] == "absent":
            if fed_to_model:
                overall = False
            if not args.all_candidates:
                print(f"  {label:20s} {E_c*1e3:11.4f} {'-':>11s} {'-':>9s} {'-':>7s} "
                      f"{'-':>9s} {'-':>8s}  ABSENT ({r['reason']})")
            rows.append(r)
            continue

        if r["verdict"] == "shadowed":
            if fed_to_model:
                overall = False
            print(f"  {label:20s} {E_c*1e3:11.4f} {r['measured_centroid_mev']*1e3:11.4f} "
                  f"{r['delta_ev']:+9.1f} {r['delta_fwhm']:7.1f} {'-':>9s} {'-':>8s}  "
                  f"SHADOWED by {r['shadowed_by']} "
                  f"({r['shadow_delta_ev']:+.1f} eV) - no separate peak here")
            rows.append(r)
            continue

        ok = r["delta_fwhm"] <= args.fail_fwhm
        strong = r["strong"] and fed_to_model
        v = "PASS" if ok else "FAIL"
        if strong and not ok:
            overall = False
        tag = "" if strong else "  (weak, below --min-count)"
        r["pass"] = bool(ok)
        print(f"  {label:20s} {E_c*1e3:11.4f} {r['measured_centroid_mev']*1e3:11.4f} "
              f"{r['delta_ev']:+9.1f} {r['delta_fwhm']:7.1f} {r['n_line']:9.0f} "
              f"{r['measured_fwhm_ev']:8.1f}  {v}{tag}")
        rows.append(r)

    if args.all_candidates:
        found = [r for r in rows if r["verdict"] == "ok" and r.get("strong")]
        print(f"\n  CONFIRMED (>= {args.min_count} line events, within "
              f"{args.fail_fwhm} FWHM): {len(found)} / {len(rows)} candidates")
        for r in found:
            print(f"    {r['label']:20s} {r['candidate_mev']*1e3:10.4f} keV  "
                  f"n_line={r['n_line']:.0f}  delta={r['delta_ev']:+.1f} eV  [{r['origin']}]")

    report["sources"][name] = rows
    print()

print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
if args.json:
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(report, f, indent=2, default=float)
    log(f"report -> {args.json}")
