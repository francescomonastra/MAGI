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
}
center = (0.0, 0.0, -507.66); R = 100.0

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493
SIGMA_MEV = args.resolution_ev * 1e-6 * FWHM_TO_SIGMA
FWHM_MEV = args.resolution_ev * 1e-6

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)


def audit_line(E_sorted, E_c, label):
    """Measure the real peak near E_c: centroid, width, line and continuum counts.

    The search window is wide (default +/-60 FWHM) so a mis-pinned line is found
    rather than silently reported as missing; the continuum level is taken as the
    median density across the window, and the line region is the contiguous run of
    eV-scale bins around the mode that stands above it.
    """
    half = args.search_fwhm * FWHM_MEV
    lo, hi = E_c - half, E_c + half
    i0, i1 = np.searchsorted(E_sorted, [lo, hi])
    sub = E_sorted[i0:i1]
    out = {"label": label, "candidate_mev": float(E_c), "n_window": int(sub.size)}
    if sub.size < 10:
        out.update(verdict="absent", reason="fewer than 10 events in the search window")
        return out

    # eV-scale histogram across the window
    step = FWHM_MEV / 2.0
    nb = max(int(round((hi - lo) / step)), 8)
    hist, edges = np.histogram(sub, bins=nb, range=(lo, hi))
    med = float(np.median(hist))
    k = int(np.argmax(hist))
    peak_e = 0.5 * (edges[k] + edges[k + 1])

    # a line must stand clearly above the local continuum density
    if hist[k] < max(10.0, 5.0 * max(med, 1.0)):
        out.update(verdict="absent", peak_energy_mev=float(peak_e),
                   continuum_per_bin=med, peak_bin_count=int(hist[k]),
                   reason="no bin stands 5x above the local continuum")
        return out

    # contiguous run above the continuum, centred on the mode
    thr = max(med * 2.0, 1.0)
    a = k
    while a > 0 and hist[a - 1] > thr:
        a -= 1
    b = k
    while b < nb - 1 and hist[b + 1] > thr:
        b += 1
    core = sub[(sub >= edges[a]) & (sub < edges[b + 1])]
    n_core = core.size
    n_cont = med * (b - a + 1)
    n_line = max(n_core - n_cont, 0.0)

    centroid = float(core.mean())
    std = float(core.std(ddof=1)) if n_core > 1 else 0.0
    err = std / np.sqrt(n_core) if n_core > 1 else float("nan")
    delta = E_c - centroid

    # A weak line sitting near a much stronger one can have the strong line's
    # peak fall inside its search window (CR: Al K-beta at 1.5450 keV "finding"
    # Al K-alpha at 1.4690 keV). If some other catalogue entry is closer to the
    # measured centroid than this one, this is not a measurement of this line.
    # Only a *resolvable* neighbour can shadow this line: Al K-alpha1/K-alpha2 are
    # 0.47 eV apart in EADL, far inside one 4 eV detector FWHM, so which of the
    # two the centroid lands nearer to is meaningless.
    nearest = min((c for c in candidate_energies if abs(c[1] - E_c) > FWHM_MEV),
                  key=lambda c: abs(c[1] - centroid), default=None)
    if nearest is not None and abs(nearest[1] - centroid) < abs(delta) - 1e-12:
        out.update(verdict="shadowed", measured_centroid_mev=centroid,
                   shadowed_by=nearest[0], shadow_delta_ev=float((nearest[1] - centroid) * 1e6),
                   delta_ev=float(delta * 1e6), delta_fwhm=float(abs(delta) / FWHM_MEV),
                   n_core=int(n_core))
        return out

    out.update(
        verdict="ok",
        measured_centroid_mev=centroid,
        centroid_err_ev=float(err * 1e6),
        measured_fwhm_ev=float(std * 2.3548200450309493 * 1e6),
        n_core=int(n_core), n_continuum=float(n_cont), n_line=float(n_line),
        delta_ev=float(delta * 1e6),
        delta_fwhm=float(abs(delta) / FWHM_MEV),
        strong=bool(n_line >= args.min_count),
    )
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
