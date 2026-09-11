# Roadmap — improvements and visualization ideas

Everything here is feasible **offline, no GPU, from data we already have or can
pull from KLIFS**, consistent with the branch philosophy.

## A. Pipeline improvements

1. **Close the coverage gap (highest impact).** Descriptor fingerprints cover
   ~12-15 of 318 kinases; compute them for all 318 (pockets are cacheable, same
   code). Unblocks every visualization below.
2. **Use the untapped KLIFS columns.** Cached records carry `resolution`,
   `quality_score`, and sub-pocket flags (`bp_II_out`, `bp_II_in`, `bp_I_A/B`,
   `gate`, `Grich_*`). Use them to (a) weight/filter by structure quality and
   (b) replace the single hard 16 A "allosteric-accessible" threshold with
   evidence-based back-pocket flags.
3. **Combined conformational label.** We compute both DFG and alphaC states —
   fuse them: DFG-in + aC-in = active; DFG-in + aC-out = Src-like inactive;
   DFG-out = inactive / Type-II-open. Richer, clinically meaningful, ~free.
4. **Trust signals.** Add a "borderline" flag near confidence 0.5, and surface
   DFG<->aC agreement as a reliability cue.
5. **Auto-detect provenance.** AlphaFold models store pLDDT in the B-factor
   column — detect it and auto-tag `predicted` instead of trusting a manual flag.

## B. Visualizations (ranked by usefulness)

1. **Kinome tree colored by conformational accessibility.** The Manning human
   kinome dendrogram, each kinase colored by Type-II-opportunity score / DFG-out
   accessibility. The canonical kinase view; answers "where in the kinome are the
   untapped Type II opportunities?" Topology is public; our per-kinase data drops on.
2. **Opportunity bubble chart.** x = fraction reaching DFG-out, y = number of
   known Type II drugs, bubble size = structure count, color = opportunity score.
   Top-left quadrant = "can go DFG-out but undrugged Type II" = a shortlist of
   openings. Makes the signature metric actionable.
3. **Two-descriptor conformational map.** DFG-distance (x) vs Lys-Glu salt-bridge
   distance (y), colored by combined state — resolves the four real conformational
   quadrants in one view. Uses the new alphaC descriptor.
4. **AlphaFold vs experimental overlay.** For kinases with both, plot AlphaFold's
   point against the experimental cloud — visualizes AlphaFold's DFG-in / active
   bias across many kinases. A genuinely publishable caution.
5. **Per-kinase conformational ridgeline.** Density of DFG-distance per kinase —
   rigid single-peak vs bistable two-peak kinases; a direct read of conformational
   plasticity / druggability by conformational selection.
6. **Honesty bar chart.** Per-fold LOKO AUROC per kinase, with the bad MET fold
   (0.26) visible — visualizes that we didn't hide the ugly fold.
7. **In-browser 3D viewer (usability).** Embed Mol*/NGL in the atlas so clicking a
   kinase spins its structure with DFG/aC highlighted (no PyMOL). Trade-off: needs
   a bundled JS lib, so a separate "rich" page rather than the single offline file.

## Recommended next 3
1. Coverage to all 318 kinases (A1) — unblocks everything.
2. Opportunity bubble chart (B2) — highest utility per effort.
3. Kinome tree (B1) — highest impact for a paper / demo.

Together these turn the atlas from "a table with a scatter" into a drug-discovery
opportunity map of the human kinome. All new visuals should follow the existing
colorblind-safe palette (Okabe-Ito: DFG-in #0072B2, DFG-out #E69F00).
