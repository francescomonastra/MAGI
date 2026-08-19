# Plan to the 24 August collaborator meeting

Written 16 Aug 2026. Revised 21:30 the same evening: training moved to Colab, and the
CR run's real ETA measured (see below — it is 10 h later than first planned).
Seven working days: Mon 17 → Sun 23. Meeting Mon 24.

## Priorities, in the order stated

1. **⁴⁰K** — the strongest speed-up + statistics-gain story for a general audience. Must be nailed.
2. **Memorisation test** — done in time. Load-bearing after the KDSource benchmark.
3. **Talk** — revision plus real timed rehearsals, not a last-minute rebuild.
4. **Paper** — complete.
5. **Repository** — cleaned so only what matters is public.

## Schedule reality check — measured 16 Aug 21:25

The CR full run is at **14 % after 2 h 44 m**, and Geant4's own estimate is
**16 h 43 m remaining — 19.5 h total, landing ≈ 14:10 Monday**, not the 03:40 I
originally planned against. Everything on Monday shifts about ten hours later.

Root cause for future planning: **this is an M1 with 4 performance + 4 efficiency cores,
and the script launches 6 jobs.** Two of them land on efficiency cores, which are several
times slower for this workload. Throughput is set by 4 P-cores, not 6 equal ones. Size
future campaigns on 4, or accept the tail.

## Three lanes, and one freeze

| Lane | Resource | Character |
|---|---|---|
| **Geant4** | 4 P-cores (8 logical) | Strictly serial. One campaign at a time, monopolises the machine. |
| **Training** | **Colab GPU** | Fully parallel to the other two. No local cost. |
| **Desk** | — | Writing, figures, repo. Fills the gaps. |

Moving training off the local GPU removes it from the critical path entirely — which
matters more now that the Geant4 lane is 10 h behind.

**One numbers freeze: Thursday 20 August, evening.** After that nothing new enters the state
reference, the paper, or the deck. Results landing later go to the journal version.

---

## Mon 17 — CR lands midday, ⁴⁰K decided, memorisation started

**Morning — desk only, the cores are still busy**
- Implement the memorisation test (needs no compute, uses existing checkpoints).
- Prepare the Colab notebook and the cleaning step so the handoff at 14:10 is minutes, not hours.
- Repo cleanup pass 1: inventory what ships and what does not.

**≈14:10 — CR full lands**
- Validate: per-primary yields, aimed-flux table, mu⁻ median against 815.7 MeV / 58.55 %.
  Confirms the *only* difference from the old set is the ingoing cut. (~30 min)
- Clean 13-column → 9-column, upload, **launch the CR retrain on Colab**. 4 types, not 6.

**≈15:00 — ⁴⁰K floor A/B** (~30–60 min)
Same binary, same source, same N, two floor configs: reduced (1500×1500×100 @ −1250)
vs original (4000×4000×800 @ −1600). Two independent discriminators against the training set:
χ = 1.756e-05, and crossing median ≈ 235 keV.

**GATE — three outcomes, all planned for**
1. **Original matches** (expected). Training set is the thick slab. Speed-up becomes
   measurable. The 44-days comparison needs a post-meeting campaign — see below.
2. **Reduced matches.** Training set shares the 44-days reference's geometry, that
   comparison becomes directly available — but then the 8× χ gap has another cause that
   must be found before trusting it.
3. **Neither matches.** The slab is not the variable. Stop and find the real one; better
   discovered in 30 minutes than after another 8-hour run.

**≈16:00 — launch ⁴⁰K ARM 1** under the winning floor. Budget 8 h under the reduced slab and
**longer under the original**, since more material means more transport per decay. Expect it
to land late Monday night or early Tuesday.
ARM 2 is **not** re-run: c_crossing = 1.0019e-07 core-h sits inside the CryoSphere, is
independent of the floor, and already agreed to 10 % across two measurements.

**Fallback if the A/B is inconclusive:** the speed-up stays unquotable and ⁴⁰K is told as an
amplification story — **287× on an exhausted sample**, already quotable, anchored by
R = 1.118 ± 0.373, KS p = 0.60. The general-public message survives either way.

### Colab hygiene — the one place this can go wrong

Training in a different environment is exactly when the **train↔generate metadata drift**
bites, and the repo already carries a guard against it for a reason.

- Export the fitted **quantile transformers alongside the checkpoint**, and reload both
  together for generation. A model trained on Colab with transformers refitted locally will
  produce silently wrong physics, not an error.
- Pin package versions to the local generation environment; record them in the run notes.
- Checkpoint to Drive every N epochs — Colab sessions disconnect on idle and cap around 12 h.
- Record the seed. The 3-seed detector study means seed provenance is part of the record.
- Before anything downstream: regenerate a small sample **locally** from the downloaded
  checkpoint and confirm species fractions and marginals match what Colab reported. Five
  minutes, and it catches every version of this failure.

---

## Tue 18 — ⁴⁰K result, memorisation run, CR closes

**Geant4 lane**
- ⁴⁰K ARM 1 finishes (overnight or early morning) → speed-up computed.
- Then CR detector-level retest against the prediction on record, **restated 17 Aug 09:36
  from the real training set** (the earlier ×1.078 / ×0.770 came from the void single-step
  run and is discarded): **aimed flux b<20 ×1.268, aimed muon flux b<20 ×1.092**, MIP band
  moving with the muon factor. Derived: the 1–7 keV MIP ratio should go 0.750 → **≈0.82**,
  i.e. the deficit narrows from 25 % to ~18 % but does not close.

**Desk / local**
- **Memorisation test** on DM1.2 and ⁴⁰K (existing checkpoints — never waited on CR).

**The test, and why it is the week's most valuable hour**
For each generated sample, nearest-neighbour distance to the training set in the 5-D
quantile-transformed space, per species. Compare against held-out-real → training.
Memorisation shows up as generated-to-train distances systematically shorter than
real-to-train. Report the ratio, the KS between the two distance distributions, and the
near-duplicate fraction below ε.

**Run the identical test on KDSource.** A smoothed bootstrap places every sample on a
training point by construction, so it should fail conspicuously. That turns the memorisation
test from a defensive check into MAGI's clearest differentiator — precisely the gap the
KDSource benchmark opened.

---

## Wed 19 — results consolidated

- CR detector-level retest results vs prediction.
- ⁴⁰K comparison plots (BuildingModel — *not* the 44-days archive, see below).
- Memorisation test on the retrained CR model, if the queue allows. Optional: DM1.2 + ⁴⁰K
  + the KDSource contrast already make the argument.
- Memorisation figure.

### The 44-days comparison — why it is off the table before the 24th

The 44-days reference spectrum was produced with the **reduced** slab. The training set's
χ and spectral hardness both point to the **original** slab. They sit on opposite sides of
the 57× (= 4000×4000×800 ÷ 1500×1500×100), and no scalar bridges them: the scaling factor
handles activity, volume and exposure, but 800 mm of concrete self-absorption yields a
physically different escaping spectrum than 100 mm.

Both remedies are too expensive for this week:
- retrain on a reduced-slab training set — the regeneration already deferred as expensive;
- generate a fresh reference under the original slab — at χ = 1.756e-05, Monday's 36 M
  decays give ~630 crossings. Fine for a cost measurement, nowhere near a reference spectrum.

**Decide at the meeting** whether a reduced-slab ⁴⁰K training set is what the September
campaign buys. It would also deliver the low-statistics ladder on the one case that already
has a 44-day reference to score against.

---

## Thu 20 — NUMBERS FREEZE

- Single consolidating pass on `docs/MAGI_state_reference.tex`: every new result, new
  provenance tags, updated quotable / not-quotable lists, updated open items.
- Paper: fill both `[PENDING]`s (SRON speed-up **3.33×**; three-seed CR **0.767 ± 0.122**).
- Paper: **fix the wrong correlation claim** and rewrite related work around the memorisation
  result rather than around amplification.

**By tonight the numbers are final.**

---

## Fri 21 — paper to 4 pages

- Cut 7 → 4 body pages. Order: architecture detail → transform derivations → variant lineage →
  prose a table already carries → any figure panel not doing independent work.
  Protect: the general-tool statement, the detector-level result, the cost pair, the limitations.
- Compile, re-measure with `lengthcheck.py`, build the Overleaf zip.
- Repo cleanup pass 2: prune, LICENSE, CITATION.cff, README.

---

## Sat 22 — release + talk revision

- Repo: final prune, tag, Zenodo DOI. Verify a clean-environment install.
- Talk: revise with final numbers. Sync SPEAKER_NOTES to the deck's general tagline
  ("Simulate the outside once. Redesign the inside forever.").
- Rebuild the deck; visually check all six PNGs — the layout is fragile and has overflowed before.

---

## Sun 23 — rehearsal and assembly

- Timed rehearsals against the 295 s budget. Cut from slide 5, never slide 4.
- Assemble the meeting pack: state reference PDF, 4-page paper, deck, figure set,
  and a one-page "what changed since last time".
- Buffer for anything that slipped.

---

## Mon 24 — collaborator meeting

---

## The opportunity Colab creates

With training off the local machine the GPU lane is idle from Monday night. The one thing
that fits it, needs **no new Geant4 at all**, and speaks directly to the meeting's audience is
the **low-statistics ladder**: subsample an existing training set at 10³ / 10⁴ / 10⁵ / 10⁶
crossings, train a model at each rung, plot fidelity against training-set size.

- It demonstrates the claim the whole tool rests on. Right now that claim is asserted, not shown.
- It pairs with the memorisation test rather than competing: a model trained on 10³ crossings
  that still generalises is the argument; the same model caught memorising is the refutation.
  Same figure, both answers.
- Subsampling is free, and the GPU time is now free.

**Bounded version, if taken:** DM1.2 only, three rungs, one seed each, one figure. Launch
Tuesday, read Wednesday, freeze Thursday with everything else. If it is not clean by Wednesday
evening it becomes the seed of the September campaign instead.

---

## If it slips, drop in this order

1. The low-statistics ladder (optional from the start).
2. Memorisation test on the **CR** model (DM1.2 + ⁴⁰K + KDSource contrast suffice).
3. Zenodo DOI (tag alone is enough to show collaborators; DOI can follow before the conference).
4. Paper down to exactly 4 pages — a 5-page draft is showable on the 24th; the cut must be
   done before the conference, not before the meeting.

**Never dropped:** the ⁴⁰K decision on Mon 17, the memorisation test on DM1.2 + KDSource,
and the Thursday numbers freeze.

---

## Machine notes (16 Aug 21:25)

- Apple M1, **4 P + 4 E cores**. Six-job campaigns oversubscribe the P-cores.
- Volume `/Volumes/X10Pro`: 700 GiB free, healthy, no thermal warnings.
- A transient loss of filesystem access to the volume occurred around 19:30–21:20 and has
  cleared. `athena` holds only `run.log` open — **the output file is created at end of run**,
  so a permission loss at the wrong moment would have cost the whole campaign. Worth checking
  access is live before any future long run.
- Background contention observed: `mds_stores` 21 % (Spotlight indexing the same volume the
  run writes to), WindowServer 12 %, AirPlayXPCHelper 9 %, Chrome ~10 %, plus a video encoder.
  Roughly half a core reclaimable. `sudo mdutil -i off /Volumes/X10Pro` needs your password.
- Swap 3.5 GiB of 5 GiB used.
