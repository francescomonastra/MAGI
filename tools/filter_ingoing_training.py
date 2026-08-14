#!/usr/bin/env python3
"""Filter the SRON CryoSphere training sets to ingoing crossings only.

`EventAction.cc` skips outgoing crossings at registration:

    G4ThreeVector toCenter = SphereCenter - Position;
    if (Direction.dot(toCenter) <= 0) continue;   // outgoing -> not recorded

but the SRON training sets predate that cut and still contain them (CR 10.37%,
K-40 22.53%), while any set produced by the current code - DM1.2 included - is
0% outgoing. MAGI does not filter either: there is no ingoing/outgoing selection
in the package, and the features cannot express it directly (u_r = rhat[:,2] and
u_v = vhat[:,2] are absolute lab-frame polar angles), so the model has to learn
the ingoing character from the (u_r,phi_r) x (u_v,phi_v) correlation - and gets
it wrong by ~3 points.

Outgoing crossings are shower products leaving the assembly; re-injecting them
inward from the sphere double-counts them and spends generation budget on
particles that cannot deposit. This script applies the same test the detector
now applies, writes `<name>_ingoing.dat` next to the original, and reports the
`Chi` that must be used with the filtered set.

Chi must be recomputed because it is defined on the very file being filtered:
    Chi = N_crossings / N_thrown
Filtering the numerator without updating Chi would break the normalization.
"""
import argparse
import os

import numpy as np

NEUTRINOS = {"nu_e", "anti_nu_e", "nu_mu", "anti_nu_mu", "nu_tau", "anti_nu_tau"}

SETS = {
    "CR": dict(
        path="/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereCR.dat",
        n_thrown=1e6 * 540,
        centre=(0.0, 0.0, -507.6),
        drop_neutrinos=True,
    ),
    "Small": dict(   # K-40
        path="/Volumes/X10Pro/MAGI/TrainingData/alloutputDSCryoSphereSmall.dat",
        n_thrown=1.53e8 * 540,
        centre=(0.0, 0.0, -507.6),
        drop_neutrinos=True,
    ),
}


def run(name, cfg, dry):
    path = cfg["path"]
    centre = np.array(cfg["centre"])
    out = path.replace(".dat", "_ingoing.dat")

    n_tot = n_nu = n_out = n_keep = 0
    sink = None if dry else open(out, "w")
    try:
        for line in open(path):
            a = line.split()
            if len(a) < 9:
                continue
            n_tot += 1
            if cfg["drop_neutrinos"] and a[1] in NEUTRINOS:
                n_nu += 1
                continue
            pos = np.array([float(a[3]), float(a[4]), float(a[5])])
            dirv = np.array([float(a[6]), float(a[7]), float(a[8])])
            # exactly EventAction's test
            if np.dot(dirv, centre - pos) <= 0:
                n_out += 1
                continue
            n_keep += 1
            if sink:
                sink.write(line)
    finally:
        if sink:
            sink.close()

    chi_old = n_tot / cfg["n_thrown"]
    chi_new = n_keep / cfg["n_thrown"]
    print(f"=== {name}  ({os.path.basename(path)})")
    print(f"    rows in                  {n_tot:>12,}")
    if cfg["drop_neutrinos"]:
        print(f"    dropped, neutrinos       {n_nu:>12,}  ({n_nu/n_tot:6.2%})")
    print(f"    dropped, outgoing        {n_out:>12,}  ({n_out/n_tot:6.2%})")
    print(f"    kept, ingoing            {n_keep:>12,}  ({n_keep/n_tot:6.2%})")
    print(f"    N thrown                 {cfg['n_thrown']:>12,.0f}")
    print(f"    Chi  all crossings       {chi_old:.6e}")
    print(f"    Chi  ingoing only        {chi_new:.6e}   (x{chi_new/chi_old:.4f})")
    if not dry:
        print(f"    wrote                    {out}")
    print()
    return chi_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=sorted(SETS))
    args = ap.parse_args()
    for name, cfg in SETS.items():
        if args.only and name != args.only:
            continue
        run(name, cfg, args.dry_run)
    print("Use the ingoing-only Chi with the ingoing-only training set. Mixing")
    print("them (old Chi with filtered data, or vice versa) biases every flux.")


if __name__ == "__main__":
    main()
