"""
Grouped, primary-conditioned generation (HENABICS primary truth).

Instead of drawing independent crossings, draw primaries the way Geant4 did,
then for each primary its shower: k crossings, their types, and their
phase space from a model conditioned on [crossing type, primary type,
qt(log10 E_prim)]. Every row carries the primary's GroupKey, PDG and energy,
and the Geant4 injector fires one group as one event, so post-MAGI deposits
come out tagged with the primary that produced them.

Primaries are drawn from the same GPS macro the full simulation used,
replicating G4SPSEneDistribution::GenUserHistEnergies (Geant4 10.4.3) rather
than refitting the spectrum, so both arms of the closure test see the same
primary population and N_primaries_sampled is directly comparable to
N_thrown.
"""

import json
import os
import re

import numpy as np

from ..data.primary_truth import _coarse_bin, resample_primary_energy, sample_group_types, sample_multiplicity
from .export import (
    MAGI_BINARY_VERSION_PRIMTRUTH,
    NON_TRANSPORT_PARTICLES,
    PARTICLE_TO_PDG,
    PRIMARY_TRUTH_EXPORT_COLS,
    _append_primary_truth_binary_chunk,
    _ensure_parent_dir,
    _filter_non_transport_particles,
    _normalize_idx_to_type,
    _normalize_output_format,
    _rewrite_binary_header,
    _write_binary_header,
    generated_physics_to_detector_dataframe,
)
from .reconstruction import reconstruct_generated_features, reconstruct_generated_physics
from .sampling import generate_latent_outputs


def parse_gps_macro(path):
    """
    Read the sources of a GPS multi-source macro.

    Follows /control/execute into the per-source ESpectrum*.mac files.
    Supports what the DM1.2 macros use: /gps/source/intensity and
    /gps/source/add (relative intensities), /gps/particle, and a
    differential user energy histogram (/gps/ene/type User,
    /gps/hist/type energy, /gps/hist/point Ehi weight). Point energies are
    bare numbers, i.e. Geant4 internal units: MeV.

    Parameters
    ----------
    path : str
        Top-level GPS macro, e.g. gpsMu_iso.mac.

    Returns
    -------
    list[dict]
        One {"intensity", "particle", "points" (n x 2 array of Ehi, weight)}
        per source, in macro order.
    """
    sources = []

    def current():
        if not sources:
            sources.append({"intensity": 1.0, "particle": None, "points": [],
                            "ene_type": None, "hist_type": None})
        return sources[-1]

    def read(fp):
        with open(fp) as f:
            for raw in f:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                cmd, *args = re.split(r"\s+", line)
                if cmd == "/control/execute":
                    read(args[0])
                elif cmd == "/gps/source/intensity":
                    current()["intensity"] = float(args[0])
                elif cmd == "/gps/source/add":
                    sources.append({"intensity": float(args[0]), "particle": None,
                                    "points": [], "ene_type": None, "hist_type": None})
                elif cmd == "/gps/source/multiplevertex" and args and args[0].lower() in ("true", "1"):
                    raise ValueError("multiplevertex GPS sources are not supported")
                elif cmd == "/gps/particle":
                    current()["particle"] = args[0]
                elif cmd == "/gps/ene/type":
                    current()["ene_type"] = args[0]
                elif cmd == "/gps/hist/type":
                    current()["hist_type"] = args[0]
                elif cmd == "/gps/hist/point":
                    current()["points"].append((float(args[0]), float(args[1])))

    read(path)
    out = []
    for i, s in enumerate(sources):
        if s["ene_type"] != "User" or s["hist_type"] != "energy" or not s["points"]:
            raise ValueError(
                f"{path}: source {i} is not a differential user energy histogram "
                f"(ene/type={s['ene_type']}, hist/type={s['hist_type']}, "
                f"{len(s['points'])} points)"
            )
        out.append({
            "intensity": s["intensity"],
            "particle": s["particle"],
            "points": np.asarray(s["points"], dtype=np.float64),
        })
    return out


def parse_primary_hist(path):
    """
    Read the thrown-primary histogram of a full-simulation job as an E_prim source.

    `path` is a run_summary.json written by the Athena/SRON run (or by merge_summaries.py). Its
    "thrown" block holds, per PDG code, counts in [underflow, n log10E bins, overflow]; the bin edges
    come from "log10E_bins". The result has the same dicts as parse_gps_macro, with "log10_edges" and
    "weights" in place of "points": sample_gps_energies draws a bin by its count and then log10E
    uniformly inside it. Species ratios are the thrown ratios. Primaries in the under/overflow bins
    are an error, since their energy is unknown.
    """
    import json
    with open(path) as f:
        summary = json.load(f)
    b = summary["log10E_bins"]
    edges = np.linspace(b["min"], b["max"], int(b["n"]) + 1)
    pdg_to_particle = {v: k for k, v in PARTICLE_TO_PDG.items()}
    out = []
    for pdg, counts in summary["thrown"].items():
        counts = np.asarray(counts, dtype=np.float64)
        if counts[0] > 0 or counts[-1] > 0:
            raise ValueError(f"{path}: PDG {pdg} has {int(counts[0])} primaries below and "
                             f"{int(counts[-1])} above the histogram range")
        if int(pdg) not in pdg_to_particle:
            raise ValueError(f"{path}: PDG {pdg} has no particle name in PARTICLE_TO_PDG")
        out.append({
            "intensity": float(counts.sum()),
            "particle": pdg_to_particle[int(pdg)],
            "log10_edges": edges,
            "weights": counts[1:-1],
        })
    return out


def sample_gps_energies(sources, n, rng):
    """
    Draw primary energies [MeV] as GPS does.

    Source i is picked with probability intensity_i / sum(intensities).
    Within a source the cumulative of the point weights, normalized to 1, is
    inverted by linear interpolation (G4PhysicsOrderedFreeVector::GetEnergy):
    u <= w_0/sum returns the first point energy exactly, which puts a point
    mass of w_0/sum there - kept, since the full simulation has it too.

    Parameters
    ----------
    sources : list[dict]
        Result of parse_gps_macro.

    n : int
        Primaries to draw.

    rng : np.random.Generator

    Returns
    -------
    energy : np.ndarray (n,)
    source_idx : np.ndarray (n,)
    """
    intensity = np.array([s["intensity"] for s in sources], dtype=np.float64)
    source_idx = rng.choice(len(sources), size=n, p=intensity / intensity.sum())
    energy = np.empty(n, dtype=np.float64)
    for i, s in enumerate(sources):
        sel = source_idx == i
        if not sel.any():
            continue
        if "log10_edges" in s:                      # thrown histogram: bin by count, flat in log10E
            edges, w = s["log10_edges"], s["weights"]
            ibin = rng.choice(len(w), size=int(sel.sum()), p=w / w.sum())
            energy[sel] = 10.0 ** (edges[ibin] + rng.random(int(sel.sum())) * (edges[ibin + 1] - edges[ibin]))
            continue
        e, w = s["points"][:, 0], s["points"][:, 1]
        cdf = np.cumsum(w) / w.sum()
        energy[sel] = np.interp(rng.random(int(sel.sum())), cdf, e)
    return energy, source_idx


def _outgoing(df, center):
    """True where the direction points out of the sphere centred at `center`."""
    r = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64) - np.asarray(center, dtype=np.float64)
    v = df[["Vx", "Vy", "Vz"]].to_numpy(dtype=np.float64)
    return (r * v).sum(axis=1) > 0


def generate_primary_groups(
    *,
    model,
    filepath,
    n_primaries,
    gps_macro,
    multiplicity_table,
    idx_to_type,
    primary_type_to_idx,
    qt_primary_energy,
    center,
    radius,
    species_pdg=None,
    geometry_metadata=None,
    energy_head_mode="mixture",
    energy_transform=None,
    qt_energy=None,
    energy_metadata=None,
    output_format="binary",
    chunk_primaries=2_000_000,
    seed=42,
    key_offset=0,
    excluded_particle_names=None,
    float_format="%.8e",
    inward_only=True,
    max_resample_rounds=50,
    verbose=1,
):
    """
    Generate a grouped, primary-tagged Geant4 source file.

    For each of `n_primaries` primaries: draw the source, hence species and
    E_prim, from `gps_macro`, draw k and the crossing types from
    `multiplicity_table`, generate the k crossings with the model
    conditioned on that primary, and write them contiguously under one
    GroupKey. Primaries with k = 0 write nothing but are counted: the
    sidecar's N_primaries_sampled is the normalization denominator, the
    MAGI-arm counterpart of the full simulation's N_thrown.

    The macro may mix species, as the SRON CR macros do (mu-, gamma, e-
    sources in one GPS). GPS then picks one source per event with
    probability intensity / sum(intensities), and so does this function,
    so the species mix follows the macro's own flux ratios. The sidecar
    keeps N_primaries_sampled and thrown_hist per species ("per_species"),
    since each flux is still normalized on its own count. Sources with
    /gps/source/multiplevertex true (several primaries in one event) are
    rejected by parse_gps_macro: the GroupKey assumes one primary per event.

    Parameters
    ----------
    model : keras.Model
        A model trained with primary conditioning (cond_dim > n_types).

    filepath : str
        Output file. A JSON sidecar is written to `filepath + ".json"`.

    n_primaries : int
        Primaries to throw, k = 0 included, all species together.

    gps_macro : str
        The GPS macro of the full simulation, single- or multi-species.

    multiplicity_table : dict
        magi.data.primary_truth.build_multiplicity_table output, as saved in
        the checkpoint metadata.

    idx_to_type : dict[int, str]
        The model's crossing-type map; must match the table's type_names.

    primary_type_to_idx : dict[int, int]
        PDG -> primary one-hot index, as used in training.

    qt_primary_energy : sklearn QuantileTransformer
        Fitted on log10(E_prim / MeV) in training.

    center, radius :
        Sphere geometry [mm], as in training.

    species_pdg : int or None
        None (default) generates every species the macro fires. A PDG code
        (13 for mu-, 22 for gamma) keeps only that species' sources, with
        their relative intensities, e.g. one species out of a mixed macro.

    geometry_metadata, energy_head_mode, energy_transform, qt_energy, energy_metadata :
        Passed to reconstruct_generated_features, as in
        generate_detector_table_to_file.

    output_format : str
        "binary" (GNTBIN v2) or "text" (8 detector columns + GroupKey
        PrimaryTrackId PrimaryPDG PrimaryEnergy).

    chunk_primaries : int
        Primaries per chunk. Only the ones that cross reach the model, so
        chunks can be much larger than in ungrouped generation.

    seed : int
        Seeds the primary, multiplicity and type draws.

    key_offset : int
        GroupKey of the first primary; keys are key_offset + primary index,
        so they never repeat within a file.

    excluded_particle_names : set or None
        Removed before writing, neutrinos by default. Group rows stay
        contiguous; a group whose rows are all removed just disappears.

    inward_only : bool
        Resample, with the same conditioning, every crossing whose direction
        points out of the sphere. The training crossings are all ingoing, so
        the outgoing ones lie outside the support: they would leave without
        reaching the detector and bias the deposit rate low. Position and
        direction are generated in the global frame, so ingoing is a joint
        constraint the model only learns approximately. Resampling keeps
        each group's k and crossing types; the rejected fraction goes in
        the sidecar.

    max_resample_rounds : int
        Rounds of resampling before the few rows still outgoing are kept
        as they are (counted in the sidecar).

    Returns
    -------
    dict
        The sidecar contents.
    """
    output_format = _normalize_output_format(output_format)
    idx_to_type = _normalize_idx_to_type(idx_to_type)
    primary_type_to_idx = {int(k): int(v) for k, v in primary_type_to_idx.items()}
    if excluded_particle_names is None:
        excluded_particle_names = NON_TRANSPORT_PARTICLES

    n_types = int(model.n_types)
    n_primary_types = len(primary_type_to_idx)
    if int(model.cond_dim) != n_types + n_primary_types + 1:
        raise ValueError(
            f"model cond_dim={model.cond_dim} does not match n_types={n_types} + "
            f"{n_primary_types} primary types + 1 energy column"
        )
    table_names = list(multiplicity_table["type_names"])
    if table_names != [idx_to_type[i] for i in range(n_types)]:
        raise ValueError("multiplicity table type_names differ from the model's idx_to_type")

    # a run_summary.json gives the thrown histogram of a full run; anything else is a GPS macro
    sources = parse_primary_hist(gps_macro) if str(gps_macro).endswith(".json") else parse_gps_macro(gps_macro)
    unknown = sorted({s["particle"] for s in sources if s["particle"] not in PARTICLE_TO_PDG})
    if unknown:
        raise ValueError(f"{gps_macro} fires {unknown}, which have no PDG code here")
    if species_pdg is not None:
        sources = [s for s in sources if PARTICLE_TO_PDG[s["particle"]] == int(species_pdg)]
        if not sources:
            raise ValueError(f"{gps_macro} has no source of PDG {species_pdg}")
    source_pdg = np.array([PARTICLE_TO_PDG[s["particle"]] for s in sources], dtype=np.int64)
    species = sorted(set(source_pdg.tolist()))
    for pdg in species:
        if str(pdg) not in multiplicity_table["species"]:
            raise ValueError(f"no multiplicity data for PDG {pdg}")
        if pdg not in primary_type_to_idx:
            raise ValueError(f"the model was not trained on primary PDG {pdg}")

    rng = np.random.default_rng(seed)
    _ensure_parent_dir(filepath)
    if output_format == "text":
        open(filepath, "w").close()
    else:
        _write_binary_header(filepath, 0, version=MAGI_BINARY_VERSION_PRIMTRUTH)

    prim_onehot = np.eye(n_primary_types, dtype=np.float32)
    source_prim_idx = np.array([primary_type_to_idx[int(p)] for p in source_pdg])

    # Per species: sampled primaries per multiplicity-table E bin (the
    # MAGI-arm counterpart of the full run's thrown histogram, for per-bin
    # closure) and the k histogram.
    edges = np.asarray(multiplicity_table["log10E_edges"], dtype=np.float64)
    thrown_hist = {pdg: np.zeros(len(edges) - 1, dtype=np.int64) for pdg in species}
    k_hist = {pdg: {} for pdg in species}
    n_rows_generated = 0
    n_outgoing_first = 0
    n_outgoing_kept = 0
    n_rows_written = 0
    n_groups_written = 0
    done = 0
    while done < n_primaries:
        n_chunk = min(chunk_primaries, n_primaries - done)
        e_prim, source_idx = sample_gps_energies(sources, n_chunk, rng)
        log10_e = np.log10(e_prim)
        prim_pdg = source_pdg[source_idx]

        # Multiplicity per species, then back into primary order, so each
        # group's rows stay contiguous whatever the species mix.
        owner_parts, type_parts = [], []
        for pdg in species:
            sel = np.flatnonzero(prim_pdg == pdg)
            if not sel.size:
                continue
            k_s, types_s = sample_multiplicity(multiplicity_table, pdg, log10_e[sel], rng)
            for kk, c in zip(*np.unique(k_s, return_counts=True)):
                k_hist[pdg][int(kk)] = k_hist[pdg].get(int(kk), 0) + int(c)
            np.add.at(thrown_hist[pdg], _coarse_bin(log10_e[sel], edges), 1)
            # E_prim of the crossing primaries from the simulation's joint (E_prim, k)
            # inside the coarse bin (the thrown histogram keeps the spectrum draw).
            log10_e[sel] = resample_primary_energy(multiplicity_table, pdg, log10_e[sel], k_s, rng)
            fine_types = sample_group_types(multiplicity_table, pdg, log10_e[sel], k_s, rng)
            if fine_types is not None:
                types_s = fine_types
            owner_parts.append(np.repeat(sel, k_s))
            type_parts.append(types_s)
        order = np.argsort(np.concatenate(owner_parts), kind="stable")
        owner = np.concatenate(owner_parts)[order]
        types = np.concatenate(type_parts)[order]

        e_prim = 10.0 ** log10_e
        n_rows = len(owner)
        if n_rows:
            qt_e = qt_primary_energy.transform(log10_e[owner].reshape(-1, 1)).astype(np.float32)
            extra = np.concatenate([prim_onehot[source_prim_idx[source_idx[owner]]], qt_e], axis=1)

            def rows(sel):
                gen_raw = generate_latent_outputs(
                    model=model, n_samples=int(sel.sum()), type_probs=None,
                    n_types=n_types, idx_to_type=idx_to_type, rng=rng,
                    gen_type_idx=types[sel], extra_cond=extra[sel],
                )
                gen_feat = reconstruct_generated_features(
                    gen_raw, geometry_metadata=geometry_metadata,
                    energy_head_mode=energy_head_mode, energy_transform=energy_transform,
                    qt_energy=qt_energy, energy_metadata=energy_metadata, rng=rng,
                )
                gen_phys = reconstruct_generated_physics(gen_feat, center=center, radius=radius)
                return generated_physics_to_detector_dataframe(gen_phys, idx_to_type=idx_to_type)

            df = rows(np.ones(n_rows, dtype=bool))
            if inward_only:
                out = _outgoing(df, center)
                n_outgoing_first += int(out.sum())
                for _ in range(max_resample_rounds):
                    if not out.any():
                        break
                    idx = np.flatnonzero(out)
                    df.iloc[idx] = rows(out).to_numpy()
                    out[idx] = _outgoing(df.iloc[idx], center)
                n_outgoing_kept += int(out.sum())
            df["GroupKey"] = np.int64(key_offset + done) + owner.astype(np.int64)
            df["PrimaryTrackId"] = np.int32(1)
            df["PrimaryPDG"] = prim_pdg[owner].astype(np.int32)
            df["PrimaryEnergy"] = e_prim[owner]

            df = _filter_non_transport_particles(df, excluded_particle_names)
            n_rows_generated += n_rows
            n_groups_written += int(df["GroupKey"].nunique())
            if output_format == "text":
                df.to_csv(filepath, mode="a", sep="\t", header=False, index=False,
                          float_format=float_format)
                n_rows_written += len(df)
            else:
                n_rows_written += _append_primary_truth_binary_chunk(df, filepath)

        done += n_chunk
        if verbose:
            print(f"[MAGI] primaries {done}/{n_primaries}: rows generated="
                  f"{n_rows_generated} written={n_rows_written}")

    if output_format == "binary":
        _rewrite_binary_header(filepath, n_rows_written, version=MAGI_BINARY_VERSION_PRIMTRUTH)

    per_species = {
        str(pdg): {
            "N_primaries_sampled": int(thrown_hist[pdg].sum()),
            "thrown_hist": thrown_hist[pdg].tolist(),
            "k_hist": {str(k): v for k, v in sorted(k_hist[pdg].items())},
        }
        for pdg in species
    }
    k_total = {}
    for pdg in species:
        for kk, c in k_hist[pdg].items():
            k_total[kk] = k_total.get(kk, 0) + c
    sidecar = {
        # One PDG when the file holds a single species, else None: each
        # species' normalization then comes from "per_species".
        "species_pdg": species[0] if len(species) == 1 else None,
        "species_pdgs": species,
        "N_primaries_sampled": int(n_primaries),
        "per_species": per_species,
        "n_groups_written": n_groups_written,
        "n_rows_generated": n_rows_generated,
        "n_rows_written": n_rows_written,
        "k_hist": {str(k): v for k, v in sorted(k_total.items())},
        "inward_only": bool(inward_only),
        "n_outgoing_first_draw": n_outgoing_first,
        "n_outgoing_kept": n_outgoing_kept,
        "log10E_edges": edges.tolist(),
        "thrown_hist": np.sum([thrown_hist[pdg] for pdg in species], axis=0).tolist(),
        "key_offset": int(key_offset),
        "gps_macro": os.path.abspath(gps_macro),
        "seed": int(seed),
        "output_format": output_format,
        "binary_version": MAGI_BINARY_VERSION_PRIMTRUTH if output_format == "binary" else None,
        "columns": ["ParticleName", "Energy", "X", "Y", "Z", "Vx", "Vy", "Vz",
                    *PRIMARY_TRUTH_EXPORT_COLS],
    }
    with open(filepath + ".json", "w") as f:
        json.dump(sidecar, f, indent=2)
    if verbose:
        print(f"[MAGI] wrote {filepath} and {filepath}.json")
    return sidecar
