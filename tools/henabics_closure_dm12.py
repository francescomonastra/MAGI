#!/usr/bin/env python
"""HENABICS closure test on DM1.2: does primary truth survive MAGI, and does
the post-MAGI simulation reproduce the full one per primary?

One species per invocation, because the species are flux-normalized
separately. The MAGI file may hold several species (generated from a
multi-species GPS macro): its rows and deposits are then cut to this species
after the plumbing checks, and N_primaries_sampled comes from the sidecar's
per-species record. The full arm is divided by the primaries thrown
(run_summary.json); the MAGI arm by the primaries *sampled*
(N_primaries_sampled in the generation sidecar), not by beamOn - primaries
with k = 0 never become a Geant4 event but still count.

    1. plumbing   every post-MAGI crossing/deposit row carries a key that
                  exists in the injected file, with that key's primary PDG and
                  energy; one (PDG, E) per key; deposit keys join to crossings.
    2. crossings  P(k >= 1) and <k | k >= 1> per E_prim bin; a classifier
                  two-sample AUC on (crossing, E_prim), MAGI vs full.
    3. detector   per-primary deposit rate R = MAGI / full, total and per
                  E_prim bin, plus the rank-1 style test: the Edep x E_prim
                  table per thrown primary, cell by cell. That joint is what the
                  cINN learns.
    4. counting   deposit rows and Edep per triggering primary, MAGI vs full,
                  bootstrapped over primaries, so a shower split into several
                  events (or merged) shows up.

Crossings inside one MAGI shower are independent given the primary (the MVP
approximation); step 4 and the Edep x E_prim table are where that would show.

    python tools/henabics_closure_dm12.py --species mu \\
        --metadata trained_models/.../mix_DM1_2_primtruth_metadata.json \\
        --full-crossings .../allDSCryoSphere_DM1_2_primtruth_mu.dat \\
        --full-deposits  .../allOutputGDML_DM1_2_primtruth_mu.dat \\
        --full-summaries .../run_summaries_mu \\
        --magi-file gen_mu.gntbin \\
        --post-deposits  postmagi_mu/job*/outputGDML.dat \\
        --post-crossings postmagi_mu/job*/outputDSCryoSphere.dat \\
        --out-dir closure_mu
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

import magi
from magi.data.primary_truth import _coarse_bin
from magi.generation.export import PARTICLE_TO_PDG

SPECIES_PDG = {"mu": 13, "gamma": 22, "e-": 11}
# outputGDML.dat, 13 columns (S1GDML_DM1.2/src/EventAction.cc)
COLS_DEP = ["EventId", "ParticleId", "ParticleInt", "EPreStep_keV", "Edep_keV",
            "x", "y", "z", "StartE_MeV", "GlobalTime",
            "PrimaryTrackId", "PrimaryPDG", "PrimaryEnergy"]
EDEP_EDGES_KEV = np.logspace(-1, 5, 25)


def poisson_ratio_ci(n_m, n_r, expo_m, expo_r, sigma=1.0):
    """Exact CI on (n_m/expo_m)/(n_r/expo_r); as in Paper/scripts/make_fig1.py."""
    n_m = np.asarray(n_m, float)
    n_r = np.asarray(n_r, float)
    alpha = 2 * (1 - norm.cdf(sigma))
    T = n_m + n_r
    with np.errstate(divide="ignore", invalid="ignore"):
        p_lo = np.where(n_m > 0, beta.ppf(alpha / 2, n_m, T - n_m + 1), 0.0)
        p_hi = np.where(n_m < T, beta.ppf(1 - alpha / 2, n_m + 1, T - n_m), 1.0)
        scale = expo_r / expo_m
        r = (n_m / expo_m) / (n_r / expo_r)
        lo = np.clip(r - p_lo / (1 - p_lo) * scale, 0, None)
        hi = np.where(p_hi < 1, p_hi / (1 - p_hi), np.inf) * scale - r
    return r, lo, hi


def poisson_pull(n_m, n_r, expo_m, expo_r):
    """Signed-root likelihood-ratio pull; as in Paper/scripts/make_fig1.py."""
    n_m = np.asarray(n_m, float)
    n_r = np.asarray(n_r, float)
    T = n_m + n_r
    p0 = expo_m / (expo_m + expo_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_m = np.where(n_m > 0, n_m * np.log(n_m / (T * p0)), 0.0)
        t_r = np.where(n_r > 0, n_r * np.log(n_r / (T * (1 - p0))), 0.0)
    z = np.sign(n_m - T * p0) * np.sqrt(np.clip(2 * (t_m + t_r), 0, None))
    return np.where(T > 0, z, np.nan)


def per_primary_deposits(paths, chunksize=5_000_000):
    """Aggregate deposit files per key without holding every step in memory.

    Returns one row per key: n_rows, Edep_keV (sum), and min/max of the
    carried PrimaryPDG and PrimaryEnergy (equal when the key has one truth).
    """
    parts = []
    for path in paths:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        for chunk in pd.read_csv(path, sep=r"\s+", header=None, names=COLS_DEP,
                                 usecols=["EventId", "Edep_keV", "PrimaryPDG",
                                          "PrimaryEnergy"],
                                 chunksize=chunksize):
            parts.append(chunk.groupby("EventId").agg(
                n_rows=("Edep_keV", "size"), Edep_keV=("Edep_keV", "sum"),
                pdg_min=("PrimaryPDG", "min"), pdg_max=("PrimaryPDG", "max"),
                e_min=("PrimaryEnergy", "min"), e_max=("PrimaryEnergy", "max"),
            ))
    if not parts:
        return pd.DataFrame(columns=["n_rows", "Edep_keV", "pdg_min", "pdg_max",
                                     "e_min", "e_max"])
    # A chunk boundary can split one event: combine the partial aggregates.
    return pd.concat(parts).groupby(level=0).agg(
        n_rows=("n_rows", "sum"), Edep_keV=("Edep_keV", "sum"),
        pdg_min=("pdg_min", "min"), pdg_max=("pdg_max", "max"),
        e_min=("e_min", "min"), e_max=("e_max", "max"),
    )


def thrown_per_bin(run_summary, pdg, edges):
    """Full-arm thrown primaries on the coarse multiplicity-table bins."""
    fine = np.asarray(run_summary["log10E_edges"])
    counts = np.asarray(run_summary["thrown"][pdg], dtype=np.int64)
    rebin = (len(fine) - 1) // (len(edges) - 1)
    inner = counts[1:-1].reshape(len(edges) - 1, rebin).sum(axis=1)
    inner[0] += counts[0]
    inner[-1] += counts[-1]
    return inner


def truth_consistent(dep, rel_tol=1e-5):
    same_pdg = dep["pdg_min"] == dep["pdg_max"]
    same_e = (dep["e_max"] - dep["e_min"]) <= rel_tol * dep["e_max"].abs()
    return same_pdg & same_e


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--species", required=True, choices=sorted(SPECIES_PDG))
    ap.add_argument("--metadata", required=True, help="checkpoint *_metadata.json")
    ap.add_argument("--full-crossings", nargs="+", required=True)
    ap.add_argument("--full-deposits", nargs="+", required=True)
    ap.add_argument("--full-summaries", nargs="+", required=True)
    ap.add_argument("--magi-file", required=True, help="injected .gntbin (v2)")
    ap.add_argument("--post-deposits", nargs="+", required=True)
    ap.add_argument("--post-crossings", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--auc-max-rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pdg = SPECIES_PDG[args.species]
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    report = {"species": args.species, "pdg": pdg}

    with open(args.metadata) as f:
        prep = json.load(f)["preprocessing_metadata"]
    table = prep["multiplicity_table"]
    edges = np.asarray(table["log10E_edges"])
    type_names = list(table["type_names"])
    type_pdgs = [PARTICLE_TO_PDG[t] for t in type_names]

    # ---- inputs ------------------------------------------------------------
    run_summary = magi.load_run_summaries(args.full_summaries)
    thrown_full = thrown_per_bin(run_summary, pdg, edges)
    N_full = int(thrown_full.sum())

    with open(args.magi_file + ".json") as f:
        side = json.load(f)
    if side.get("log10E_edges") != table["log10E_edges"]:
        raise ValueError("sidecar has no thrown_hist on the table's E bins; regenerate")
    if "per_species" in side:
        if str(pdg) not in side["per_species"]:
            raise ValueError(f"{args.magi_file} has no PDG {pdg}: {side['species_pdgs']}")
        side_sp = side["per_species"][str(pdg)]
    elif side["species_pdg"] == pdg:
        side_sp = side  # single-species sidecar from before per_species
    else:
        raise ValueError(f"{args.magi_file} is PDG {side['species_pdg']}, not {pdg}")
    thrown_magi = np.asarray(side_sp["thrown_hist"], dtype=np.int64)
    N_magi = int(side_sp["N_primaries_sampled"])
    report["N_thrown_full"] = N_full
    report["N_primaries_sampled_magi"] = N_magi

    full_x = pd.concat([magi.load_detector_table(filepath=p, sep=r"\s+")
                        for p in args.full_crossings], ignore_index=True)
    full_x = full_x[(full_x["PrimaryPDG"] == pdg)
                    & full_x["ParticleName"].isin(type_names)]
    version, rec = magi.read_detector_binary(args.magi_file)
    if version != 2:
        raise ValueError(f"{args.magi_file} is GNTBIN v{version}; primary truth needs v2")
    magi_x = pd.DataFrame({
        "EventId": rec["group_key"], "pdg": rec["pdg"], "Energy": rec["energy"],
        "X": rec["x"], "Y": rec["y"], "Z": rec["z"],
        "Vx": rec["vx"], "Vy": rec["vy"], "Vz": rec["vz"],
        "PrimaryPDG": rec["prim_pdg"], "PrimaryEnergy": rec["prim_energy"],
    })
    truth = magi_x.groupby("EventId").agg(pdg=("PrimaryPDG", "first"),
                                         e=("PrimaryEnergy", "first"))

    dep_full = per_primary_deposits(args.full_deposits)
    dep_magi = per_primary_deposits(args.post_deposits)
    dep_full = dep_full[dep_full["pdg_min"] == pdg]

    # ---- 1. plumbing ---------------------------------------------------------
    plumb = {}
    j = truth.reindex(dep_magi.index)
    plumb["deposit_keys_post"] = int(len(dep_magi))
    plumb["deposit_orphan_keys"] = int(j["pdg"].isna().sum())
    plumb["deposit_pdg_mismatch"] = int((j["pdg"].to_numpy() != dep_magi["pdg_min"].to_numpy()).sum())
    plumb["deposit_max_rel_energy_diff"] = float(np.nanmax(
        np.abs(dep_magi["e_min"].to_numpy() / j["e"].to_numpy() - 1), initial=0.0))
    plumb["deposit_keys_with_multiple_truths"] = int((~truth_consistent(dep_magi)).sum())
    plumb["full_deposit_keys_with_multiple_truths"] = int((~truth_consistent(dep_full)).sum())
    plumb["full_deposit_keys_without_crossing"] = int((~dep_full.index.isin(full_x["EventId"])).sum())
    if args.post_crossings:
        post_x = pd.concat([magi.load_detector_table(filepath=p, sep=r"\s+")
                            for p in args.post_crossings], ignore_index=True)
        jx = truth.reindex(post_x["EventId"])
        plumb["crossing_rows_post"] = int(len(post_x))
        plumb["crossing_orphan_rows"] = int(jx["pdg"].isna().sum())
        plumb["crossing_pdg_mismatch"] = int((jx["pdg"].to_numpy() != post_x["PrimaryPDG"].to_numpy()).sum())
        plumb["crossing_max_rel_energy_diff"] = float(np.nanmax(
            np.abs(post_x["PrimaryEnergy"].to_numpy() / jx["e"].to_numpy() - 1), initial=0.0))
        plumb["post_deposit_keys_without_crossing"] = int((~dep_magi.index.isin(post_x["EventId"])).sum())
    tol = 1e-5  # the text tables print 6 significant digits
    plumb["pass"] = bool(
        plumb["deposit_orphan_keys"] == 0 and plumb["deposit_pdg_mismatch"] == 0
        and plumb["deposit_max_rel_energy_diff"] < tol
        and plumb["deposit_keys_with_multiple_truths"] == 0
        and plumb.get("crossing_orphan_rows", 0) == 0
        and plumb.get("crossing_pdg_mismatch", 0) == 0
        and plumb.get("crossing_max_rel_energy_diff", 0.0) < tol
    )
    report["plumbing"] = plumb

    # Plumbing ran on every key in the file; the rest is this species only.
    magi_x = magi_x[magi_x["PrimaryPDG"] == pdg]
    dep_magi = dep_magi[dep_magi["pdg_min"] == pdg]

    # ---- 2. crossings ----------------------------------------------------------
    def k_by_bin(x):
        g = x.groupby("EventId")
        k = g.size().to_numpy()
        b = _coarse_bin(np.log10(g["PrimaryEnergy"].first().to_numpy()), edges)
        n_cross = np.bincount(b, minlength=len(edges) - 1)
        sum_k = np.bincount(b, weights=k, minlength=len(edges) - 1)
        return n_cross, sum_k

    nf, kf = k_by_bin(full_x)
    nm, km = k_by_bin(magi_x)
    rows = []
    for i in np.flatnonzero((thrown_full > 0) & (thrown_magi > 0)):
        r, lo, hi = poisson_ratio_ci(nm[i], nf[i], thrown_magi[i], thrown_full[i])
        rows.append({
            "log10E_lo": edges[i], "log10E_hi": edges[i + 1],
            "thrown_full": int(thrown_full[i]), "thrown_magi": int(thrown_magi[i]),
            "P_cross_full": nf[i] / thrown_full[i], "P_cross_magi": nm[i] / thrown_magi[i],
            "P_cross_ratio": float(r), "ratio_lo": float(lo), "ratio_hi": float(hi),
            "mean_k_full": kf[i] / nf[i] if nf[i] else np.nan,
            "mean_k_magi": km[i] / nm[i] if nm[i] else np.nan,
        })
    cross_bins = pd.DataFrame(rows)
    cross_bins.to_csv(os.path.join(args.out_dir, "crossings_per_Eprim_bin.csv"), index=False)
    r, lo, hi = poisson_ratio_ci(nm.sum(), nf.sum(), N_magi, N_full)
    report["crossings"] = {
        "P_cross_full": float(nf.sum() / N_full), "P_cross_magi": float(nm.sum() / N_magi),
        "P_cross_ratio": [float(r), float(lo), float(hi)],
        "mean_k_full": float(kf.sum() / max(nf.sum(), 1)),
        "mean_k_magi": float(km.sum() / max(nm.sum(), 1)),
    }

    # Two-sample classifier on single crossings with their primary energy.
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    full_x = full_x.assign(pdg=full_x["ParticleName"].map(PARTICLE_TO_PDG))

    def features(x):
        n = min(len(x), args.auc_max_rows)
        x = x.iloc[rng.choice(len(x), size=n, replace=False)]
        return np.column_stack([
            np.log10(x["Energy"].to_numpy()), x[["X", "Y", "Z", "Vx", "Vy", "Vz"]].to_numpy(),
            np.log10(x["PrimaryEnergy"].to_numpy()),
            *(x["pdg"].to_numpy() == p for p in type_pdgs),
        ])

    Xf, Xm = features(full_x), features(magi_x)
    n = min(len(Xf), len(Xm))
    if n >= 50:
        X = np.vstack([Xf[:n], Xm[:n]])
        y = np.r_[np.zeros(n), np.ones(n)]
        p = cross_val_predict(HistGradientBoostingClassifier(random_state=args.seed), X, y,
                              cv=5, method="predict_proba")[:, 1]
        report["crossings"]["classifier_auc"] = float(roc_auc_score(y, p))
        report["crossings"]["classifier_rows_per_arm"] = int(n)

    # ---- 3. detector level -------------------------------------------------------
    trig_f = dep_full[dep_full["Edep_keV"] > 0]
    trig_m = dep_magi[dep_magi["Edep_keV"] > 0]
    r, lo, hi = poisson_ratio_ci(len(trig_m), len(trig_f), N_magi, N_full)
    det = {"triggering_primaries_full": int(len(trig_f)),
           "triggering_primaries_magi": int(len(trig_m)),
           "R": [float(r), float(lo), float(hi)]}

    # Primary-level Poisson bootstrap: each thrown primary gets weight
    # Poisson(1); only triggering ones contribute to any count, so resampling
    # them is enough.
    def boot_R(stat_m, stat_f):
        if not (len(stat_m) and len(stat_f)):
            return float("nan")
        out = []
        for _ in range(args.n_boot):
            wm = rng.poisson(1.0, len(stat_m))
            wf = rng.poisson(1.0, len(stat_f))
            out.append((wm @ stat_m / N_magi) / (wf @ stat_f / N_full))
        return float(np.std(out))
    det["R_boot_sigma"] = boot_R(np.ones(len(trig_m)), np.ones(len(trig_f)))

    def joint(dep):
        b = _coarse_bin(np.log10(dep["e_min"].to_numpy()), edges)
        h, _, _ = np.histogram2d(b, dep["Edep_keV"].to_numpy(),
                                 bins=[np.arange(len(edges)), EDEP_EDGES_KEV])
        return h

    hf, hm = joint(trig_f), joint(trig_m)
    z = poisson_pull(hm, hf, N_magi, N_full)
    ok = (hm + hf) >= 5
    det["joint_Edep_Eprim_chi2"] = float(np.nansum(z[ok] ** 2))
    det["joint_Edep_Eprim_ndf"] = int(ok.sum())
    per_bin = []
    for i in np.flatnonzero(hf.sum(axis=1) + hm.sum(axis=1) > 0):
        rb, lob, hib = poisson_ratio_ci(hm[i].sum(), hf[i].sum(), thrown_magi[i], thrown_full[i])
        per_bin.append({"log10E_lo": edges[i], "log10E_hi": edges[i + 1],
                        "n_full": int(hf[i].sum()), "n_magi": int(hm[i].sum()),
                        "R": float(rb), "R_lo": float(lob), "R_hi": float(hib)})
    pd.DataFrame(per_bin).to_csv(os.path.join(args.out_dir, "R_per_Eprim_bin.csv"), index=False)
    np.savez(os.path.join(args.out_dir, "joint_Edep_Eprim.npz"), full=hf, magi=hm,
             log10E_edges=edges, edep_edges_keV=EDEP_EDGES_KEV,
             N_full=N_full, N_magi=N_magi, pull=z)
    report["detector"] = det

    # ---- 4. counting ---------------------------------------------------------------
    cnt = {}
    for name, col in (("rows_per_triggering_primary", "n_rows"),
                      ("Edep_keV_per_triggering_primary", "Edep_keV")):
        a, b = trig_m[col].to_numpy(float), trig_f[col].to_numpy(float)
        if len(a) and len(b):
            ratio = a.mean() / b.mean()
            boots = [rng.choice(a, len(a)).mean() / rng.choice(b, len(b)).mean()
                     for _ in range(args.n_boot)]
            cnt[name] = {"magi": float(a.mean()), "full": float(b.mean()),
                         "ratio": float(ratio), "ratio_boot_sigma": float(np.std(boots))}
    report["counting"] = cnt

    with open(os.path.join(args.out_dir, "closure_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))
    _plot(args.out_dir, args.species, hf, hm, N_full, N_magi, cross_bins, per_bin)


def _plot(out_dir, species, hf, hm, N_full, N_magi, cross_bins, per_bin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    cen = np.sqrt(EDEP_EDGES_KEV[1:] * EDEP_EDGES_KEV[:-1])
    ax[0].step(cen, hf.sum(0) / N_full, where="mid", label="full Geant4")
    ax[0].step(cen, hm.sum(0) / N_magi, where="mid", ls="--", label="MAGI")
    ax[0].set(xscale="log", yscale="log", xlabel="Edep per primary [keV]",
              ylabel="events / thrown primary")
    ax[0].legend(frameon=False)
    if len(cross_bins):
        c = 0.5 * (cross_bins["log10E_lo"] + cross_bins["log10E_hi"])
        for arm, fmt, label in (("full", "o-", "full"), ("magi", "s--", "MAGI")):
            p = cross_bins[f"P_cross_{arm}"]
            # P(k >= 1) is below 2%, so the binomial error is Poisson on the
            # crossing count.
            err = np.sqrt(p * cross_bins[f"thrown_{arm}"]) / cross_bins[f"thrown_{arm}"]
            ax[1].errorbar(c, p, yerr=err, fmt=fmt, ms=3, capsize=2, label=label)
        ax[1].set(yscale="log", xlabel="log10 E_prim [MeV]", ylabel="P(k >= 1)")
        ax[1].legend(frameon=False)
    if per_bin:
        pb = pd.DataFrame(per_bin)
        c = 0.5 * (pb["log10E_lo"] + pb["log10E_hi"])
        ymax = 2.0
        ok = np.isfinite(pb["R"]) & np.isfinite(pb["R_hi"]) & (pb["R"] <= ymax)
        ax[2].errorbar(c[ok].to_numpy(), pb["R"][ok].to_numpy(),
                       yerr=[pb["R_lo"][ok].to_numpy(), pb["R_hi"][ok].to_numpy()],
                       fmt="o", ms=3, capsize=2)
        # Off-scale (or infinite, no full-sim counts) bins: marker on the top edge.
        off = ~ok
        ax[2].plot(c[off], np.full(off.sum(), ymax * 0.97), "^", ms=5, color="C0")
        # Per-bin counts: with a few triggering primaries per bin the error
        # bars alone hide how little each point rests on.
        for x, nm_, nf_ in zip(c, pb["n_magi"], pb["n_full"]):
            ax[2].annotate(f"{nm_}/{nf_}", (x, 0.04), ha="center", va="bottom",
                           rotation=90, fontsize=7, color="0.3")
        ax[2].axhline(1, color="0.4", ls="--", lw=0.8)
        ax[2].margins(x=0.1)
        ax[2].set(xlabel="log10 E_prim [MeV]",
                  ylabel="R = MAGI / full (deposits)", ylim=(0, ymax))
        ax[2].set_title("labels: n MAGI / n full; ^ = off scale", fontsize=8)
    fig.suptitle(f"HENABICS closure DM1.2, primary {species}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "closure.png"), dpi=150)


if __name__ == "__main__":
    main()
