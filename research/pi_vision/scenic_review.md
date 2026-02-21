# SCENIC+ Literature Review

**Source:** Nature Methods 2023, Bravo González-Blas et al.

---

## What SCENIC+ Provides

| Output | Description | Use in Our Model |
|--------|-------------|------------------|
| **eRegulons** | TF + enhancers + target genes | Network topology |
| **TF Activity Scores** | Per-cell TF activity (AUCell) | Activation labels |
| **Chromatin Topics** | Co-accessible region sets | Context features |
| **Cell Type Annotations** | Cell type embeddings | Context encoder input |

---

## Pipeline Steps

1. **cisTopic**: Topic modeling on scATAC → region accessibility patterns
2. **pycistarget**: Motif enrichment in accessible regions
3. **GRN inference**: Link TFs to targets via enhancer correlation
4. **eRegulon identification**: Bundle TF + enhancers + targets

---

## Key Insight for Hyaline

> SCENIC+ TF activity scores DIRECTLY measure context-dependent activation

For TF $i$ in cell $c$:
- High TF activity + accessible chromatin → **LOW spike threshold** (easy to fire)
- Low TF activity + closed chromatin → **HIGH spike threshold** (hard to fire)

This maps perfectly to our spiking architecture:

```python
threshold_offset = context_encoder(scenic_features)
# threshold_offset < 0 → permissive context → easier firing
# threshold_offset > 0 → repressive context → harder firing
```

---

## Validation Cases from SCENIC+

| TF | Cell Type | Expected Activity | Evidence |
|----|-----------|-------------------|----------|
| SOX10 | Melanocyte | VERY_HIGH | Master regulator |
| MITF | Melanocyte | HIGH | SOX10 target |
| SOX10 | Hepatocyte | SILENT | Wrong lineage |
| HNF4A | Hepatocyte | VERY_HIGH | Master regulator |
| GATA4 | Cardiomyocyte | VERY_HIGH | Master regulator |

---

## Data Sources

- **Melanoma SCENIC+**: GEO GSE115978 (Jerby-Arnon 2018)
- **Pre-computed eRegulons**: SCENIC+ database
- **TF Structures**: AlphaFold DB
