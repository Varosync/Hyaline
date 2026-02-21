# Biological Validation Cases

## Test Suite for Context-Dependent TF Activation

---

## Category 1: Melanocyte TFs

| TF | Context | Expected | Target Genes | Evidence |
|----|---------|----------|--------------|----------|
| **SOX10** | Melanocyte | VERY_HIGH | DCT, TYRP1, MITF | Master regulator |
| **SOX10** | Hepatocyte | SILENT | - | Wrong lineage |
| **MITF** | Melanocyte | HIGH | TYR, DCT, PMEL | SOX10 target |
| **MITF** | Fibroblast | LOW | - | Wrong lineage |

---

## Category 2: Hepatocyte TFs

| TF | Context | Expected | Target Genes | Evidence |
|----|---------|----------|--------------|----------|
| **HNF4A** | Hepatocyte | VERY_HIGH | ALB, AFP, APOB | Master regulator |
| **HNF4A** | Melanocyte | SILENT | - | Wrong lineage |
| **CEBPA** | Hepatocyte | HIGH | G6PC, PCK1 | Metabolic TF |

---

## Category 3: Cardiomyocyte TFs

| TF | Context | Expected | Target Genes | Evidence |
|----|---------|----------|--------------|----------|
| **GATA4** | Cardiomyocyte | VERY_HIGH | MYH6, TNNT2, NPPA | Core cardiac TF |
| **GATA4** | Hepatocyte | LOW | - | developmental role |
| **NKX2-5** | Cardiomyocyte | VERY_HIGH | GATA4, TBX5 | Cross-activation |

---

## Category 4: Ubiquitous TFs

| TF | Context | Expected | Notes |
|----|---------|----------|-------|
| **SP1** | Any | MEDIUM | Housekeeping |
| **YY1** | Any | MEDIUM | Housekeeping |
| **CTCF** | Any | MEDIUM | Chromatin organizer |

---

## Validation Protocol

1. Train model on 80% cell types
2. Test on held-out 20%
3. Specifically check: "Does SOX10 activate in melanocytes but not hepatocytes?"
4. If yes WITHOUT explicit supervision → mechanistic understanding

---

## Expected Results

| Metric | Target |
|--------|--------|
| Intra-cell-type AUC | >0.85 |
| Cross-cell-type AUC | >0.70 |
| SOX10 melanocyte vs hepatocyte | >0.5 difference |
