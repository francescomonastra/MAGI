#!/usr/bin/env python3
"""
Build a CANDIDATE_ENERGY_LINES data file for a specific Geant4 mass model.

Purpose
-------
The v0.8 mixture-density energy head (see docs/v0.8_beta_plan.md) needs a
table of physically-expected spectral lines to match against the real
training spectrum. That table must be grounded in this experiment's own
Geant4 data, not literature values recalled from memory:

  - Instrumental fluorescence lines: for every element actually present in
    the GDML mass model, look up its K/L-shell fluorescence transition
    energies in Geant4's own atomic-relaxation dataset (G4EMLOW, the
    "fluor_Bearden" tables - Bearden 1967 reference energies, the same
    dataset Geant4's Livermore/Penelope EM physics can use for de-excitation).
  - Source decay lines: for each radioactive background component (K-40,
    Ra-226 chain, Th-232 chain, Rn-222 chain), look up the dominant gamma
    transitions of the relevant decay-chain daughters in Geant4's own
    PhotonEvaporation dataset (nuclear level scheme + gamma branching data).

Nothing here is destructive: it only reads Geant4 data files and the GDML,
and writes one new JSON file. It does not run Geant4 and has no Geant4
Python bindings dependency - the data files are plain text.

Known limitation (documented, not silently swept under the rug): the
decay-chain gamma extraction (parse_photoevap_lines) uses a "prefer
transitions that decay directly to the ground state" heuristic to pick the
dominant lines per nuclide, since Geant4's PhotonEvaporation "intensity"
column is not on a single consistent scale across a file (see that
function's docstring). This reproduces several well-known reference lines
exactly (K-40: 1460.851 keV; Bi-214: 609.316/1764.515 keV; Pb-214: 295.223
keV; Tl-208: 2614.522 keV - all independently verified against standard
gamma-spectroscopy references) but can miss a chain's other well-known lines
that decay via an intermediate excited level rather than straight to ground
(e.g. Bi-214's 1120 keV, Ac-228's 911 keV do not currently appear). This is
acceptable for the current phase, which only uses the K-40 and instrumental
lines (both extracted correctly) - if a later phase needs the full Ra-226/
Th-232 chain line set, re-check this heuristic's output against literature
first.

Usage
-----
    python build_candidate_lines_from_geant4.py \
        --gdml /Volumes/X10Pro/Geant4-projects/Geant4-10.4.3/S1GDMLSRON_CryoSphereEmission/SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed.gdml \
        --fluor-dir /Volumes/X10Pro/Geant4-system/geant4-v10.4.3-build/data/G4EMLOW7.3/fluor_Bearden \
        --photoevap-dir /Volumes/X10Pro/Geant4-system/geant4-v10.4.3-build/data/PhotonEvaporation5.2 \
        --out-dir /Volumes/X10Pro/MAGI/CandidateLines

Output: CandidateLines/CANDIDATE_ENERGY_LINES_<gdml basename>.json
"""

import argparse
import datetime
import json
import os
import re
import sys

# ----------------------------------------------------------------------------
# Decay chains: which PhotonEvaporation daughter files to read for each
# radioactive background component. This mapping is standard nuclear physics
# (which nuclide decays to which), not something extracted from Geant4's data
# files - those are keyed by nuclide, not by "chain membership".
#
# Naming convention (verified empirically against known reference energies,
# not assumed): a gamma line conventionally cited under isotope X's name is
# physically emitted by X's own excited state immediately after being
# populated by X's PARENT's decay - i.e. the level-scheme file to read is
# X's own (Z, A), not X's decay product. E.g. the "Pb-214" 295/352 keV lines
# are emitted by excited Bi-214 (Pb-214's beta-decay daughter) relaxing to
# Bi-214's own ground state, so the "Pb-214" entry below points at Bi-214's
# file (z83.a214) - confirmed by grep against the raw PhotonEvaporation5.2
# data, not guessed. Same pattern verified for Bi-214->Po-214 (609/1120/1764
# keV), Ac-228->Th-228 (911 keV), Tl-208->Pb-208 (583/2614 keV).
# ----------------------------------------------------------------------------
DECAY_CHAINS = {
    "K-40": [("Ar-40", 18, 40)],
    "Ra-226 chain": [
        ("Rn-222", 86, 222),
        ("Po-218", 84, 218),
        ("Pb-214", 83, 214),   # -> Bi-214's own level file (verified: 295.22/351.93 keV)
        ("Bi-214", 84, 214),   # -> Po-214's own level file (verified: 609.32/1120.29/1764.51 keV)
    ],
    "Rn-222 chain": [
        ("Po-218", 84, 218),
        ("Pb-214", 83, 214),
        ("Bi-214", 84, 214),
    ],
    "Th-232 chain": [
        ("Ra-228", 89, 228),   # -> Ac-228's own level file
        ("Ac-228", 90, 228),   # -> Th-228's own level file (verified: 911.21 keV)
        ("Th-228", 88, 224),   # -> Ra-224's own level file
        ("Ra-224", 86, 220),   # -> Rn-220's own level file
        ("Rn-220", 84, 216),   # -> Po-216's own level file
        ("Po-216", 82, 212),   # -> Pb-212's own level file
        ("Pb-212", 83, 212),   # -> Bi-212's own level file
        ("Tl-208", 82, 208),   # -> Pb-208's own level file (verified: 583.19/2614.52 keV)
    ],
}

# Minimum Z for which L-shell fluorescence is extracted (below this, L lines
# fall well under 1 keV and aren't physically relevant for this detector).
L_LINE_MIN_Z = 40

# Minimum relative gamma intensity (Geant4 PhotonEvaporation "intensity"
# column, roughly percent-branching within that transition) to keep a decay
# line, and the max number of lines kept per daughter nuclide.
MIN_GAMMA_INTENSITY = 5.0
MAX_LINES_PER_NUCLIDE = 3


def parse_gdml_elements(gdml_path):
    """Return sorted [(symbol, Z), ...] for every <element> in the GDML."""
    with open(gdml_path) as f:
        text = f.read()

    elements = {}
    for tag in re.finditer(r"<element\b([^>]*)/?>", text):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag.group(1)))
        if "Z" in attrs and "formula" in attrs:
            elements[attrs["formula"]] = int(attrs["Z"])

    return sorted(elements.items(), key=lambda kv: kv[1])


def parse_fluor_lines(fluor_dir, z):
    """
    Extract K-alpha1, K-alpha2, K-beta (and, for high-Z elements, L-alpha,
    L-beta) transition energies for element Z from a Geant4 fluor_Bearden
    fl-tr-pr-<Z>.commented.dat file.

    Returns a list of (label_suffix, energy_kev) tuples, e.g.
    [("K-alpha1", 8.04778), ("K-alpha2", 8.02765), ("K-beta", 8.90529)].
    """
    path = os.path.join(fluor_dir, f"fl-tr-pr-{z}.commented.dat")
    if not os.path.exists(path):
        return []

    with open(path) as f:
        raw = f.read()

    groups = []
    current_header = None
    current_rows = []
    for line in raw.splitlines():
        tokens = line.split("*", 1)
        data_part = tokens[0].split()
        comment = tokens[1].strip() if len(tokens) > 1 else ""
        if not data_part:
            continue

        nums = [float(x) for x in data_part]
        if len(nums) >= 3 and nums[0] == -1:
            if current_header is not None:
                groups.append((current_header, current_rows))
            current_header = None
            current_rows = []
            continue

        if current_header is None:
            current_header = int(nums[0])
            continue

        if len(nums) >= 3:
            prob, energy_mev = nums[1], nums[2]
            current_rows.append((prob, energy_mev * 1000.0, comment))

    if current_header is not None:
        groups.append((current_header, current_rows))

    lines = []

    # K-shell group (header id 1).
    k_group = next((rows for hdr, rows in groups if hdr == 1), [])
    kalpha1 = [r for r in k_group if "alpha1" in r[2] and "alpha2" not in r[2]]
    kalpha2 = [r for r in k_group if "alpha2" in r[2]]
    kbeta = sorted((r for r in k_group if "beta" in r[2]), key=lambda r: -r[0])
    if kalpha1:
        lines.append(("K-alpha1", kalpha1[0][1]))
    if kalpha2:
        lines.append(("K-alpha2", kalpha2[0][1]))
    if kbeta:
        lines.append(("K-beta", kbeta[0][1]))

    # L-shell groups (header ids 3=LI, 5=LII, 6=LIII), high-Z only.
    if z >= L_LINE_MIN_Z:
        l_rows = [r for hdr, rows in groups if hdr in (3, 5, 6) for r in rows]
        lalpha = sorted((r for r in l_rows if "alpha" in r[2]), key=lambda r: -r[0])
        lbeta = sorted((r for r in l_rows if "beta" in r[2]), key=lambda r: -r[0])
        if lalpha:
            lines.append(("L-alpha", lalpha[0][1]))
        if lbeta:
            lines.append(("L-beta", lbeta[0][1]))

    return lines


def parse_photoevap_lines(photoevap_dir, z, a):
    """
    Extract the dominant gamma decay lines for nuclide (Z, A) from a Geant4
    PhotonEvaporation z<Z>.a<A> level-scheme file.

    Geant4's "intensity" column is a per-transition branching value that is
    NOT on a single consistent scale across a file - most rows report a
    clean 0-100 relative-to-level percentage, but a minority of rows (for
    exceptionally well-measured lines) report values well above 100 on a
    different absolute-yield convention, which would otherwise dominate a
    naive "sort by intensity" ranking. To get a reasonable, defensible
    approximation of the dominant OBSERVED lines without a full cascade-
    population calculation: prefer transitions that decay directly to the
    ground state (final level 0) - these are reliably the terminal, most
    commonly observed step of any decay cascade - within the clean 0-100
    intensity range, then fill any remaining slots with the next-strongest
    non-ground transitions in the same range.

    Returns a list of (energy_kev, intensity) tuples, strongest first.
    """
    path = os.path.join(photoevap_dir, f"z{z}.a{a}")
    if not os.path.exists(path):
        return []

    to_ground = []
    other = []
    with open(path) as f:
        for raw in f:
            tokens = raw.split()
            if len(tokens) < 3:
                continue
            if tokens[1] == "-":
                continue  # level header row, not a gamma line
            try:
                final_level = int(tokens[0])
                energy_kev = float(tokens[1])
                intensity = float(tokens[2])
            except ValueError:
                continue
            if energy_kev <= 0 or not (0 < intensity <= 100):
                continue
            (to_ground if final_level == 0 else other).append((energy_kev, intensity))

    to_ground.sort(key=lambda t: -t[1])
    other.sort(key=lambda t: -t[1])

    kept = []
    seen_energies = set()
    for energy_kev, intensity in to_ground + other:
        if len(kept) >= MAX_LINES_PER_NUCLIDE:
            break
        if energy_kev in seen_energies:
            continue
        seen_energies.add(energy_kev)
        kept.append((energy_kev, intensity))

    return kept


def build_candidate_lines(gdml_path, fluor_dir, photoevap_dir):
    elements = parse_gdml_elements(gdml_path)
    lines = []

    # -- e+/e- annihilation: a fundamental physical constant (twice the
    # electron rest mass, one photon each), not an element- or mass-model-
    # specific fluorescence line, so it isn't in Geant4's fluor_Bearden
    # tables - included here directly since any detector material with
    # enough pair-production/positron activity produces it. --
    lines.append({
        "label": "e+e- annihilation",
        "energy_kev": 510.999,
        "energy_mev": 510.999 / 1000.0,
        "origin": "instrumental",
        "source": "physical constant (electron rest mass)",
    })

    # -- Instrumental fluorescence lines, one entry per element/transition --
    for symbol, z in elements:
        for suffix, energy_kev in parse_fluor_lines(fluor_dir, z):
            lines.append({
                "label": f"{symbol} {suffix}",
                "energy_kev": round(energy_kev, 5),
                "energy_mev": round(energy_kev / 1000.0, 8),
                "origin": "instrumental",
                "source": f"G4EMLOW7.3/fluor_Bearden/fl-tr-pr-{z}.dat",
            })

    # -- Source decay lines, one entry per daughter-nuclide transition --
    for chain_name, daughters in DECAY_CHAINS.items():
        for label, z, a in daughters:
            for energy_kev, intensity in parse_photoevap_lines(photoevap_dir, z, a):
                lines.append({
                    "label": label,
                    "energy_kev": round(energy_kev, 5),
                    "energy_mev": round(energy_kev / 1000.0, 8),
                    "origin": f"source:{chain_name}",
                    "source": f"PhotonEvaporation5.2/z{z}.a{a}",
                    "intensity": intensity,
                })

    return lines


def build_parser():
    p = argparse.ArgumentParser(
        description="Build a CANDIDATE_ENERGY_LINES data file from Geant4's own "
                     "fluorescence and decay databases, for a specific GDML mass model."
    )
    p.add_argument("--gdml", required=True, help="Path to the GDML mass model file.")
    p.add_argument("--fluor-dir", required=True,
                   help="Path to G4EMLOW's fluor_Bearden directory.")
    p.add_argument("--photoevap-dir", required=True,
                   help="Path to Geant4's PhotonEvaporation directory.")
    p.add_argument("--out-dir", default="CandidateLines",
                   help="Output directory for the JSON file.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    lines = build_candidate_lines(args.gdml, args.fluor_dir, args.photoevap_dir)

    mass_model_name = os.path.basename(args.gdml)
    mass_model_basename = os.path.splitext(mass_model_name)[0]

    payload = {
        "mass_model": mass_model_name,
        "mass_model_path": os.path.abspath(args.gdml),
        "sources": list(DECAY_CHAINS.keys()),
        "fluor_dataset_path": os.path.abspath(args.fluor_dir),
        "photoevap_dataset_path": os.path.abspath(args.photoevap_dir),
        "generated_by": "tools/build_candidate_lines_from_geant4.py",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_lines": len(lines),
        "lines": lines,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"CANDIDATE_ENERGY_LINES_{mass_model_basename}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Parsed {len(parse_gdml_elements(args.gdml))} elements from GDML.")
    print(f"Wrote {len(lines)} candidate lines to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
