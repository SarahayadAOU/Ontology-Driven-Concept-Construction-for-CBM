# Results

`gallstone_CBM_concept_scheme_comparison.xlsx` is the comparison workbook from
the current experiment. Running `python run_experiments.py` writes machine-readable
CSV outputs into this directory, including AUC/accuracy, threshold metrics (F1, F2,
sensitivity, specificity, precision, NPV, balanced accuracy, MCC), DeLong tests,
stability, leakage audit, concept impurity, predictive completeness, calibration,
conformal coverage, and selective abstention.

Important: the workbook documents the replication choices and disclosed deviations
from the original paper. Review that `Method` sheet before interpreting exact values.
