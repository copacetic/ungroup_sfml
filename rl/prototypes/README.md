# Prototype and measurement scripts behind docs/SKILL_CEILING.md

Scratch code from the September 2026 skill-ceiling analysis, kept so the numbers in the document can be
re-derived. Each directory matches a section of the document. The scripts were written against scratch
copies of the core; the rule prototypes (ecology, social, spatial, evolution, smartbot) patch or replace
`rl/native/ungroup.cpp` in a copy and never touch the canonical core. Paths inside the scripts point at the
scratch directory they ran in; adjust before running.

| directory | section | what it holds |
| --- | --- | --- |
| economy | 2 | closed-form derivation, solo and group oracles, the solo bot attribution |
| payoff-landscape | 3.1 to 3.3 | 28-composition sweep (CSV), replicator dynamics, invasions |
| smartbot | 3.4 | the best-response bot iterations (C++ snippets) and paired ladders |
| lobbies | 4 | 12, 20 and 32 player recordings and the crowd statistics |
| skill-ceiling | 5 | steering, spill, leave-share and endgame measurements |
| ecology | 6.2 | Bloom with seeding: patch, ladders, depletion-recovery analysis |
| social | 6.1, 6.3 | Crown, Bond and Brand: full core copy, ladders, fairness and contagion statistics |
| spatial | 6.4 | Tidewater and the clover shrink: diff, crowd chirality, lobe statistics |
| evolution | 6.5 | persistent ledger and the imitation league experiments |

Figures: `replicator_pooled.png` (strategy simplex and time series), `fig1_structure_20.png` (20-player
crowd structure), `fig_stock_heatmaps.png` (mine stock, legacy versus Bloom), `fig_brand_groups.png`.
