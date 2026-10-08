# Gallstone dataset

The experiment uses the public **UCI Gallstone dataset (ID 1150)**.

- Dataset page: https://archive.ics.uci.edu/dataset/1150/gallstone-1
- UCI download archive: https://cdn.uci-ics-mlr-prod.aws.uci.edu/1150/gallstone-1.zip
- UCI license: **CC BY 4.0**
- Expected data: **319 patient rows, 39 columns = 1 target + 38 inputs**.
- Raw UCI target coding: `0 = gallstones present`, `1 = absent`.
- The experiment code recodes the target internally so `1 = gallstones present`.

Run from the repository root:

```bash
python download_data.py
```

This creates `data/gallstone_uci.csv`. If you redistribute the CSV in GitHub,
retain the UCI attribution and CC BY 4.0 notice.

The UCI website currently displays a metadata count of 320 instances, but the
downloaded worksheet used for this study contains 319 data rows plus one header
row. This discrepancy is documented in `docs/Gallstone_Data_Dictionary.xlsx`.
