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
        --gdml /path/to/Geant4-10.4.3/S1GDMLSRON_CryoSphereEmission/SRON_CCNwithXFDM_NoShield_FlowerCryoAC_fixed.gdml \
        --fluor-dir /path/to/geant4-v10.4.3-build/data/G4EMLOW7.3/fluor_Bearden \
        --photoevap-dir /path/to/geant4-v10.4.3-build/data/PhotonEvaporation5.2 \
        --out-dir /path/to/MAGI/CandidateLines

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

# ----------------------------------------------------------------------------
# NIST predefined materials, for GDML files that reference a Geant4 built-in
# material by name (<materialref ref="G4_Al"/>) with no inline <material>
# definition at all. Geant4's own GDML writer always bakes in full
# <material>/<element> blocks (the SRON mass model was built this way, so it
# never needs this table), but a GDML exported from a CAD/mesh tool can rely
# entirely on G4NistManager resolving "G4_*" names at load time - confirmed on
# CriostatoDM1_2_Richiesta11_06_2026-worldVOL.gdml, which has no <materials>
# section whatsoever despite 7 distinct materialrefs across its 52 volumes.
#
# Every entry is copied from G4NistMaterialBuilder.cc (Geant4 10.4.3,
# source/materials/src/), i.e. the actual AddMaterial()/AddElementBy*() calls
# that build these names - not textbook compositions, since Geant4's NIST
# tables occasionally differ from nominal (e.g. G4_BRASS includes 3% Pb).
# Extend this table, sourced the same way, if a new mass model references a
# NIST material not listed here - parse_gdml_volume_elements warns when that
# happens instead of silently returning zero elements.
# ----------------------------------------------------------------------------
_NIST_MATERIAL_ELEMENTS = {
    # single-element (AddMaterial(name, density, Z, ...))
    "G4_Al": [("Al", 13)],
    "G4_Si": [("Si", 14)],
    "G4_Ti": [("Ti", 22)],
    "G4_Fe": [("Fe", 26)],
    "G4_Ni": [("Ni", 28)],
    "G4_Cu": [("Cu", 29)],
    "G4_W":  [("W", 74)],
    "G4_Au": [("Au", 79)],
    "G4_Pb": [("Pb", 82)],
    # compounds (AddElementByAtomCount / AddElementByWeightFraction)
    "G4_BRASS": [("Cu", 29), ("Zn", 30), ("Pb", 82)],
    "G4_STAINLESS-STEEL": [("Fe", 26), ("Cr", 24), ("Ni", 28)],
    "G4_POLYPROPYLENE": [("C", 6), ("H", 1)],
    "G4_TEFLON": [("C", 6), ("F", 9)],
    "G4_MYLAR": [("C", 6), ("H", 1), ("O", 8)],
    "G4_KAPTON": [("C", 6), ("H", 1), ("N", 7), ("O", 8)],
    "G4_WATER": [("H", 1), ("O", 8)],
    "G4_AIR": [("C", 6), ("N", 7), ("O", 8), ("Ar", 18)],
    # vacuum: Geant4 models it internally as an extremely rarefied gas (with
    # a placeholder Z=1), but physically it has no elements - candidate lines
    # from "vacuum fluorescence" would be meaningless, so this is deliberately
    # empty rather than [("H", 1)].
    "G4_Galactic": [],
}


def parse_gdml_volume_elements(gdml_path):
    """Map each logical volume to the elements of its material.

    Returns (volume_to_elements, element_z) where volume_to_elements is
    {volume_name: {"material": ..., "elements": [symbol, ...]}} and element_z
    is {symbol: Z}. Needed to tell *detector* materials from the surrounding
    structure, which is what decides whose fluorescence can produce an escape
    peak.

    A material is resolved, in order: from the GDML's own inline <material>
    definition; else from the _NIST_MATERIAL_ELEMENTS fallback table above (a
    material referenced by name with no inline definition - see that table's
    docstring); else left with no elements, which is reported so it is never
    a silent gap.
    """
    with open(gdml_path) as f:
        text = f.read()

    # element ref name ("Cu_element") -> (symbol, Z)
    ref_to_element = {}
    element_z = {}
    for tag in re.finditer(r"<element\b([^>]*)>", text):
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag.group(1)))
        if "name" in attrs and "formula" in attrs and "Z" in attrs:
            ref_to_element[attrs["name"]] = attrs["formula"]
            element_z[attrs["formula"]] = int(attrs["Z"])

    # material -> [symbols], from <composite ref=...> / <fraction ref=...>
    material_to_elements = {}
    for block in re.finditer(r'<material\b[^>]*name="([^"]+)"[^>]*>(.*?)</material>',
                             text, flags=re.S):
        name, body = block.group(1), block.group(2)
        syms = []
        for ref in re.findall(r'<(?:composite|fraction)\b[^>]*ref="([^"]+)"', body):
            sym = ref_to_element.get(ref)
            if sym is None and ref in element_z:      # material referring to an element by symbol
                sym = ref
            if sym and sym not in syms:
                syms.append(sym)
        material_to_elements[name] = syms

    # volume -> material -> elements
    volume_to_elements = {}
    nist_fallback_used = set()
    unresolved_materials = set()
    for block in re.finditer(r'<volume\b[^>]*name="([^"]+)"[^>]*>(.*?)</volume>',
                             text, flags=re.S):
        name, body = block.group(1), block.group(2)
        m = re.search(r'<materialref\b[^>]*ref="([^"]+)"', body)
        if not m:
            continue
        material = m.group(1)

        if material_to_elements.get(material):
            syms = material_to_elements[material]
        elif material in _NIST_MATERIAL_ELEMENTS:
            syms_z = _NIST_MATERIAL_ELEMENTS[material]
            for sym, z in syms_z:
                element_z.setdefault(sym, z)
            syms = [sym for sym, _ in syms_z]
            nist_fallback_used.add(material)
        else:
            syms = []
            if material not in material_to_elements:
                unresolved_materials.add(material)

        volume_to_elements[name] = {"material": material, "elements": syms}

    if nist_fallback_used:
        print(f"parse_gdml_volume_elements: {len(nist_fallback_used)} material(s) had "
              f"no inline <material> definition in the GDML - resolved from the "
              f"built-in NIST composition table instead: {sorted(nist_fallback_used)}")
    if unresolved_materials:
        print(f"parse_gdml_volume_elements: WARNING - {len(unresolved_materials)} "
              f"material(s) could not be resolved (no inline <material> block, and "
              f"not in the NIST fallback table) - their volumes contribute NO "
              f"elements to the candidate-line table: {sorted(unresolved_materials)}. "
              f"If any of these is a genuine Geant4 NIST material, add it to "
              f"_NIST_MATERIAL_ELEMENTS in this file, sourced from "
              f"G4NistMaterialBuilder.cc (see that table's docstring).")

    return volume_to_elements, element_z


def parse_gdml_elements(gdml_path):
    """Return sorted [(symbol, Z), ...] for every element used by any volume
    in the GDML - whether declared inline as <element>/<material>, or (a
    materialref with no inline definition) resolved via the
    _NIST_MATERIAL_ELEMENTS fallback in parse_gdml_volume_elements.
    """
    volume_to_elements, element_z = parse_gdml_volume_elements(gdml_path)
    symbols = {sym for info in volume_to_elements.values() for sym in info["elements"]}
    return sorted(((sym, element_z[sym]) for sym in symbols if sym in element_z),
                  key=lambda kv: kv[1])


def build_escape_peak_lines(parent_lines, detector_elements, fluor_dir,
                            fluor_energy_dir=None, min_energy_kev=1.0,
                            max_lines=None):
    """Escape-peak candidates: parent line energy minus a detector-material
    fluorescence photon that leaves without depositing.

    These are *candidates only*. In this project the training data is the
    spectrum of particles CROSSING a virtual sphere around the cryostat, not
    energy deposited in the detector, so classical escape peaks need not be
    present at all - tools/line_centroid_audit.py --all-candidates is what
    decides which (if any) are really there before one becomes a mixture
    component.

    parent_lines : the strong lines an escape peak can be built from.
    detector_elements : [(symbol, Z), ...] of the detector / near-detector
        materials, from parse_gdml_volume_elements.
    """
    escapes = []
    for symbol, z in detector_elements:
        for rec in parse_fluor_lines(fluor_dir, z, energy_dir=fluor_energy_dir):
            e_f = rec["energy_kev"]
            for parent in parent_lines:
                e_esc = parent["energy_kev"] - e_f
                if e_esc < min_energy_kev:
                    continue
                escapes.append({
                    "label": f"{parent['label']} escape {symbol} {rec['suffix']}",
                    "energy_kev": round(e_esc, 5),
                    "energy_mev": round(e_esc / 1000.0, 8),
                    "origin": f"escape:{parent['label']}-{symbol}{rec['suffix']}",
                    "source": (f"{parent.get('source', 'parent')} minus "
                               f"{symbol} {rec['suffix']} fluorescence"),
                    "parent_label": parent["label"],
                    "parent_energy_kev": parent["energy_kev"],
                    "escape_element": symbol,
                    "escape_transition": rec["suffix"],
                    "escape_energy_kev": round(e_f, 5),
                    "confirmed": None,   # set by the centroid audit against real data
                })

    escapes.sort(key=lambda l: l["energy_kev"])
    return escapes[:max_lines] if max_lines else escapes


def _parse_fluor_file(path):
    """Parse one Geant4 fl-tr-pr-<Z>[.commented].dat into
    {group_header: [(transition_index, probability, energy_kev, comment), ...]}.

    File layout: a group starts with a repeated-id line, each data row is
    "<transition index> <probability> <energy in MeV>" (optionally followed by a
    "* alpha1 K Liii" comment in the .commented.dat variant), and "-1 -1 -1"
    closes the group.
    """
    with open(path) as f:
        raw = f.read()

    groups = {}
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
                groups[current_header] = current_rows
            current_header = None
            current_rows = []
            continue

        if current_header is None:
            current_header = int(nums[0])
            continue

        if len(nums) >= 3:
            trans_idx, prob, energy_mev = int(nums[0]), nums[1], nums[2]
            current_rows.append((trans_idx, prob, energy_mev * 1000.0, comment))

    if current_header is not None:
        groups[current_header] = current_rows

    return groups


def parse_fluor_lines(fluor_dir, z, energy_dir=None):
    """
    Extract K-alpha1, K-alpha2, K-beta (and, for high-Z elements, L-alpha,
    L-beta) transitions for element Z from Geant4's atomic-relaxation data.

    Transition *names* always come from `fluor_dir`'s `.commented.dat` files
    (only the Bearden directory ships the "* alpha1 K Liii" comments that
    identify which row is which line).

    Transition *energies* come from `energy_dir` when given, joined row-for-row
    on (group header, transition index). This matters: Geant4 emits the energies
    of whichever fluorescence directory the application selected, and the
    default is EADL (`fluor`), not Bearden. On the CryoSphere data the two
    differ by 17-43 eV - 4-11x the X-IFU 4 eV FWHM the v0.8 line widths are
    pinned to - so a table built from the wrong directory puts every line
    outside its own real peak. Measured CR peaks match EADL to <1 eV.

    Returns a list of dicts:
      {"suffix", "energy_kev", "energy_kev_labels_dir", "energy_kev_energy_dir",
       "delta_ev", "intensity"}
    where "energy_kev" is the value to use (from energy_dir if given).
    """
    label_path = os.path.join(fluor_dir, f"fl-tr-pr-{z}.commented.dat")
    if not os.path.exists(label_path):
        return []

    label_groups = _parse_fluor_file(label_path)

    energy_lookup = None
    if energy_dir is not None:
        for cand in (f"fl-tr-pr-{z}.commented.dat", f"fl-tr-pr-{z}.dat"):
            energy_path = os.path.join(energy_dir, cand)
            if os.path.exists(energy_path):
                energy_lookup = {
                    (hdr, row[0]): row[2]
                    for hdr, rows in _parse_fluor_file(energy_path).items()
                    for row in rows
                }
                break
        if energy_lookup is None:
            raise FileNotFoundError(
                f"No fl-tr-pr-{z}[.commented].dat in energy dir {energy_dir}"
            )

    def _emit(suffix, hdr, row):
        trans_idx, prob, energy_label, _comment = row
        energy_alt = None
        if energy_lookup is not None:
            energy_alt = energy_lookup.get((hdr, trans_idx))
            if energy_alt is None:
                raise KeyError(
                    f"Z={z}: transition (group {hdr}, index {trans_idx}) present in "
                    f"the label directory but absent from the energy directory - "
                    f"the two fluorescence datasets are not row-compatible."
                )
        chosen = energy_alt if energy_alt is not None else energy_label
        return {
            "suffix": suffix,
            "energy_kev": chosen,
            "energy_kev_labels_dir": energy_label,
            "energy_kev_energy_dir": energy_alt,
            "delta_ev": (None if energy_alt is None
                         else round((energy_alt - energy_label) * 1000.0, 3)),
            "intensity": prob,
        }

    lines = []

    # K-shell group (header id 1).
    k_group = label_groups.get(1, [])
    kalpha1 = [r for r in k_group if "alpha1" in r[3] and "alpha2" not in r[3]]
    kalpha2 = [r for r in k_group if "alpha2" in r[3]]
    kbeta = sorted((r for r in k_group if "beta" in r[3]), key=lambda r: -r[1])
    if kalpha1:
        lines.append(_emit("K-alpha1", 1, kalpha1[0]))
    if kalpha2:
        lines.append(_emit("K-alpha2", 1, kalpha2[0]))
    if kbeta:
        lines.append(_emit("K-beta", 1, kbeta[0]))

    # L-shell groups (header ids 3=LI, 5=LII, 6=LIII), high-Z only.
    if z >= L_LINE_MIN_Z:
        l_rows = [(hdr, r) for hdr in (3, 5, 6) for r in label_groups.get(hdr, [])]
        lalpha = sorted((hr for hr in l_rows if "alpha" in hr[1][3]),
                        key=lambda hr: -hr[1][1])
        lbeta = sorted((hr for hr in l_rows if "beta" in hr[1][3]),
                       key=lambda hr: -hr[1][1])
        if lalpha:
            lines.append(_emit("L-alpha", lalpha[0][0], lalpha[0][1]))
        if lbeta:
            lines.append(_emit("L-beta", lbeta[0][0], lbeta[0][1]))

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


DEFAULT_DETECTOR_VOLUME_PATTERNS = (
    "SensitiveDetector",   # X-IFU Au absorber
    "DetectorPlate",       # Si plate under it
    "CryoAC",              # Si anticoincidence
    "Thermistor",          # Ge thermistor
)


def select_detector_elements(gdml_path, patterns):
    """Elements of every volume whose name matches one of `patterns`."""
    volume_to_elements, element_z = parse_gdml_volume_elements(gdml_path)
    picked, matched_volumes = {}, []
    for vol, info in volume_to_elements.items():
        if not any(p.lower() in vol.lower() for p in patterns):
            continue
        matched_volumes.append((vol, info["material"]))
        for sym in info["elements"]:
            if sym in element_z:
                picked[sym] = element_z[sym]
    return sorted(picked.items(), key=lambda kv: kv[1]), matched_volumes


def build_candidate_lines(gdml_path, fluor_dir, photoevap_dir, fluor_energy_dir=None,
                          escape_peaks=False,
                          detector_volume_patterns=DEFAULT_DETECTOR_VOLUME_PATTERNS,
                          escape_min_kev=1.0):
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

    # -- Instrumental fluorescence lines, one entry per element/transition.
    # Labels come from fluor_dir's commented files; energies from
    # fluor_energy_dir when given (Geant4's default is EADL `fluor`, and the
    # Bearden values are 17-43 eV away - several line widths). Both are kept in
    # the record so a later run can tell which dataset the simulation used. --
    energy_dir_name = (os.path.basename(os.path.normpath(fluor_energy_dir))
                       if fluor_energy_dir else os.path.basename(os.path.normpath(fluor_dir)))
    for symbol, z in elements:
        for rec in parse_fluor_lines(fluor_dir, z, energy_dir=fluor_energy_dir):
            entry = {
                "label": f"{symbol} {rec['suffix']}",
                "energy_kev": round(rec["energy_kev"], 5),
                "energy_mev": round(rec["energy_kev"] / 1000.0, 8),
                "origin": "instrumental",
                "source": f"G4EMLOW7.3/{energy_dir_name}/fl-tr-pr-{z}.dat",
                "intensity": rec["intensity"],
            }
            if rec["energy_kev_energy_dir"] is not None:
                entry["energy_kev_bearden"] = round(rec["energy_kev_labels_dir"], 5)
                entry["bearden_minus_used_ev"] = -rec["delta_ev"]
            lines.append(entry)

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

    # -- Escape-peak candidates (opt-in): a parent line minus a fluorescence
    # photon of the detector material that escapes. Candidates only - the
    # centroid audit against the real spectrum decides which exist. --
    if escape_peaks:
        detector_elements, matched_volumes = select_detector_elements(
            gdml_path, detector_volume_patterns)
        parents = [l for l in lines
                   if l["origin"] == "instrumental" and l["energy_kev"] > escape_min_kev]
        escapes = build_escape_peak_lines(
            parents, detector_elements, fluor_dir,
            fluor_energy_dir=fluor_energy_dir, min_energy_kev=escape_min_kev)
        print(f"Detector volumes matched: "
              f"{sorted({m for _, m in matched_volumes})} "
              f"({len(matched_volumes)} volumes) -> elements "
              f"{[s for s, _ in detector_elements]}")
        print(f"Escape-peak candidates generated: {len(escapes)} "
              f"(from {len(parents)} parent lines)")
        lines.extend(escapes)

    return lines


def build_parser():
    p = argparse.ArgumentParser(
        description="Build a CANDIDATE_ENERGY_LINES data file from Geant4's own "
                     "fluorescence and decay databases, for a specific GDML mass model."
    )
    p.add_argument("--gdml", required=True, help="Path to the GDML mass model file.")
    p.add_argument("--fluor-dir", required=True,
                   help="Path to G4EMLOW's fluor_Bearden directory (supplies the "
                        "transition LABELS - only this directory ships the "
                        "'.commented.dat' files that name each transition).")
    p.add_argument("--fluor-energy-dir", default=None,
                   help="Path to the fluorescence directory whose ENERGIES the "
                        "simulation actually emitted, joined row-for-row to "
                        "--fluor-dir. Geant4's default is G4EMLOW's 'fluor' (EADL); "
                        "pass it here unless the application called "
                        "SetFluoDirectory(\"fluor_Bearden\"). Omit to use the "
                        "Bearden energies (the pre-v0.8.1 behaviour).")
    p.add_argument("--photoevap-dir", required=True,
                   help="Path to Geant4's PhotonEvaporation directory.")
    p.add_argument("--out-dir", default="CandidateLines",
                   help="Output directory for the JSON file.")
    p.add_argument("--escape-peaks", action="store_true",
                   help="Also emit escape-peak CANDIDATES (parent line minus a "
                        "detector-material fluorescence photon). Candidates only: "
                        "confirm them against the real spectrum with "
                        "tools/line_centroid_audit.py --all-candidates before "
                        "feeding any of them to the mixture head.")
    p.add_argument("--detector-volume", nargs="+",
                   default=list(DEFAULT_DETECTOR_VOLUME_PATTERNS),
                   help="Substrings identifying the detector / near-detector "
                        "logical volumes in the GDML, whose materials' "
                        "fluorescence can escape.")
    p.add_argument("--escape-min-kev", type=float, default=1.0,
                   help="Drop escape candidates below this energy.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    lines = build_candidate_lines(args.gdml, args.fluor_dir, args.photoevap_dir,
                                  fluor_energy_dir=args.fluor_energy_dir,
                                  escape_peaks=args.escape_peaks,
                                  detector_volume_patterns=args.detector_volume,
                                  escape_min_kev=args.escape_min_kev)

    mass_model_name = os.path.basename(args.gdml)
    mass_model_basename = os.path.splitext(mass_model_name)[0]

    payload = {
        "mass_model": mass_model_name,
        "mass_model_path": os.path.abspath(args.gdml),
        "sources": list(DECAY_CHAINS.keys()),
        "fluor_dataset_path": os.path.abspath(args.fluor_dir),
        "fluor_label_dataset_path": os.path.abspath(args.fluor_dir),
        "fluor_energy_dataset_path": (os.path.abspath(args.fluor_energy_dir)
                                      if args.fluor_energy_dir else
                                      os.path.abspath(args.fluor_dir)),
        "photoevap_dataset_path": os.path.abspath(args.photoevap_dir),
        "generated_by": "tools/build_candidate_lines_from_geant4.py",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_lines": len(lines),
        "lines": lines,
    }

    # Tag the filename with the energy dataset when it isn't the label dataset,
    # so a Bearden-energy table and an EADL-energy table for the same mass model
    # never overwrite each other (the earlier v0.8 checkpoints were trained
    # against the Bearden one).
    tag = ""
    if args.fluor_energy_dir and (os.path.abspath(args.fluor_energy_dir)
                                  != os.path.abspath(args.fluor_dir)):
        dir_name = os.path.basename(os.path.normpath(args.fluor_energy_dir))
        tag = "_EADL" if dir_name == "fluor" else f"_{dir_name}"
    if args.escape_peaks:
        tag += "_escape"

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f"CANDIDATE_ENERGY_LINES_{mass_model_basename}{tag}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Parsed {len(parse_gdml_elements(args.gdml))} elements from GDML.")
    print(f"Wrote {len(lines)} candidate lines to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
