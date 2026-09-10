# Analyze demo - 5 experimental + 2 AlphaFold

Reproduce: `python scripts/demo_analyze_batch.py`. Every row is validated
against `hyaline/kinase/analysis_schema.json`.

| Input | Kinase | Provenance | DFG | Conf. | Inhibitor class | Pocket | Schema |
|---|---|---|---|---|---|---|---|
| 2hyy | ABL1 | experimental | DFG-out | 0.674 | Type II | 85/85 | valid |
| 1m17 | EGFR | experimental | DFG-in | 0.879 | Type I | 85/85 | valid |
| 1uwh | BRAF | experimental | DFG-out | 0.694 | Type II | 85/85 | valid |
| 1t46 | KIT | experimental | DFG-out | 0.598 | Type II | 85/85 | valid |
| 2src | SRC | experimental | DFG-out | 0.772 | Type II | 85/85 | valid |
| AF-P00519 | ABL1 | predicted | DFG-in | 0.74 | Type I | 85/85 | valid |
| AF-P00533 | EGFR | predicted | DFG-in | 0.526 | Type I | 85/85 | valid |
