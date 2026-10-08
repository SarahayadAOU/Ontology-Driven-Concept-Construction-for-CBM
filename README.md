# Ontology-Driven Concept Construction for Concept Bottleneck Models

Reproducibility package for the Gallstone case study comparing the original
**manual top-down CBM concept scheme** with two **ontology-derived bottom-up**
concept schemes.

## Research questions

**RQ1.** Does ontology-driven bottom-up construction provide a more reproducible,
traceable, and externally verifiable method for constructing the concept layer than
manual top-down concept definition?

**RQ2.** When inserted into the same CBM architecture, do ontology-derived concepts
preserve or improve discrimination, concept completeness, explanation stability,
leakage-controlled performance, calibration, conformal coverage, and selective
prediction relative to manually defined concepts?

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── download_data.py
├── run_experiments.py
├── data/
│   └── README.md
├── ontology/
│   ├── README.md
│   └── download_ontologies.py
├── config/
│   ├── concept_groups.csv
│   ├── concept_groups.json
│   ├── manual_groups.json
│   ├── ontology_default_groups.json
│   └── ontology_bridged_groups.json
├── src/
│   ├── cbm.py
│   └── metrics.py
├── docs/
│   ├── Gallstone_Data_Dictionary.xlsx
│   └── gallstone_CMO_crosswalk.xlsx
└── results/
    ├── gallstone_CBM_concept_scheme_comparison.xlsx
    └── README.md
```

## 1. Obtain the dataset

The public UCI Gallstone dataset is licensed CC BY 4.0.

Dataset page:
https://archive.ics.uci.edu/dataset/1150/gallstone-1

The source data contain 319 patient records and 39 columns (one target plus 38
predictors). UCI codes `0 = gallstones present` and `1 = absent`; the analysis
recodes this internally to `1 = gallstones present`.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
```

This creates `data/gallstone_uci.csv`.

## 2. Ontologies

The case study uses CMO as the primary ontology and MONDO, OBA/PATO, and VT as
complementary resources. Canonical links and licenses are listed in
`ontology/README.md`.

For the archival paper release, freeze the exact ontology snapshots and checksums:

```bash
python ontology/download_ontologies.py
```

## 3. Concept schemes

The repository contains three frozen concept configurations:

1. `manual_groups.json` — original hand-built nine-concept scheme.
2. `ontology_default_groups.json` — ten-concept bottom-up ontology-derived scheme.
3. `ontology_bridged_groups.json` — ontology scheme with the two explicitly documented
   bridge choices (Hyperlipidemia -> Lipid panel; Diabetes Mellitus -> Glycemic).

`config/concept_groups.csv` gives variable-level provenance, evidence type, and the
manual-vs-ontology placement. It also makes clear where domain backfill rather than
formal ontology ancestry was used.

## 4. Run the complete experiment

```bash
python run_experiments.py
```

Defaults:

- 10 repeated stratified CV repeats
- 10 folds per repeat
- identical folds for every scheme/encoder comparison
- supervised concept encoder: L1 logistic log-odds, `liblinear`, `C=1.0`
- unsupervised concept encoder: first PCA component, label-free orientation
- logistic regression concept head
- 100 bootstrap resamples for the current stability replication
- supervised leakage audit: proper fold-wise encoders vs deliberately naive full-data encoders

For a longer stability run matching the original paper's 200 bootstrap target:

```bash
python run_experiments.py --bootstraps 200
```

## 5. Main outputs

The script writes:

- `cv_summary.csv` — AUC and accuracy mean/SD.
- `classification_metrics.csv` — AUC, accuracy, sensitivity, specificity, precision,
  NPV, F1, F2, balanced accuracy, MCC, and confusion-matrix counts.
- `delong_comparisons.csv` — paired manual-vs-ontology AUC tests.
- `leakage_audit.csv` — proper AUC, naive full-data-encoder AUC, and AUC inflation.
- `predictive_completeness.csv` — AUC-based concept completeness and interpretability tax.
- `stability.csv` — concept Rank SD and top-k Jaccard.
- `concept_impurity.csv` — mean/max off-diagonal concept-score R².
- `reliability.csv` — raw/Platt/isotonic ECE, conformal coverage, singleton rate.
- `selective_abstention.csv` — accuracy at 100%, 90%, 80%, 70%, 60%, and 50% coverage.
- `oof_predictions.csv` — paired out-of-fold probabilities for auditing/re-analysis.

## 6. Leakage-controlled improvement

The key comparison uses concept encoders fitted **only inside each training fold**.
The deliberately leaky full-data-encoder experiment is reported only as an audit.
Thus, higher ontology-CBM AUC under the proper pipeline is interpreted as a
**leakage-controlled improvement**, not as a gain produced by held-out labels entering
the concept representation.

The current comparison workbook reports supervised AUCs of approximately 0.844
(manual), 0.856 (ontology default), and 0.861 (ontology bridged), while the deliberate
full-data encoder produces inflated values around 0.874, 0.896, and 0.898 respectively.
Reviewers should treat only the fold-wise estimates as valid predictive performance.

## 7. Reproducibility note

The code in this repository is the reconstructed implementation used for the ontology
comparison. The comparison workbook documents all disclosed deviations from the earlier
CBM paper, including the 100-bootstrap stability run and the nested reliability protocol.
Do not silently replace these settings when reproducing the reported comparison.

## 8. Data and ontology licenses

- UCI Gallstone dataset: CC BY 4.0. Cite the original dataset authors and UCI.
- CMO: CC0 1.0.
- MONDO: CC BY 4.0.
- OBA: CC0 1.0.
- PATO: CC BY 3.0.
- VT: CC BY 4.0.
- Repository code: MIT (see `LICENSE`).

## Dataset citation

Esen, I., Arslan, H., Aktürk, S., Gülşen, M., Kültekin, N., & Özdemir, O.
Gallstone. UCI Machine Learning Repository. Associated article DOI:
10.1097/MD.0000000000037258.
