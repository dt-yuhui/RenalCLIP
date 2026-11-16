# Source Data update — 16 November 2026

This directory provides an author-corrected Source Data workbook for [A disease-centric vision-language foundation model for precision oncology in kidney cancer](https://doi.org/10.1038/s41467-026-74175-w), *Nature Communications* **17**, 7313 (2026).

**[Download the updated Source Data workbook](source_data_corrected_2026-11-16.xlsx?raw=true)**

This update addresses four issues in the Source Data: operating-point metrics exported with a different threshold rule, an incorrect confidence interval, a missing event-status column, and insufficient numerical precision of exported risk scores. We apologize for these inconsistencies and thank the researchers who brought the Figure 3b discrepancy to our attention.

## Changes

| Worksheet | Issue | Correction |
| --- | --- | --- |
| `Figure3b` | Sensitivity (`recall`), specificity and F1 were exported at a fixed probability threshold of 0.5, while Figure 3b uses model-, task- and cohort-specific thresholds maximizing Youden's J. | Recomputed these metrics from the supplied individual labels and logits using the stated Youden procedure. Of 90 threshold-dependent entries, 85 change and five remain numerically identical. The 60 ROC AUC and PR AUC entries do not change. |
| `Figure3h` | The 95% CI for the Nomogram's malignancy AUC in small renal masses in the combined external cohort was entered as 0.577–0.675. | Corrected cells `F7:G7` to **0.565–0.712**, consistent with Supplementary Table 21 and the saved 1,000 bootstrap results. The AUC in `E7` remains 0.636. |
| `Figure3f-g&Supp-Fig5` | The recurrence-free survival event indicator was omitted. | Added `rfs_event` in column N for all 271 existing records (61 internal and 210 TCIA). `1` indicates an RFS event and `0` indicates censoring. These records were matched uniquely to the original analysis data using follow-up duration, five risk scores and five predicted labels. |
| `Figure3f-g&Supp-Fig5`, `Fig4b-h&Supp-Fig6a-j&Supp-Fig7`, `Figure4i&Supp-Fig6k-o` | Risk scores were stored rounded to three decimal places, introducing ties that can change concordance estimates. | Restored 3,730 risk-score entries to their original full precision. Cell formatting continues to display three decimal places; calculations use the underlying full-precision values. |

## Verification

- All 150 numerical entries in the published Figure 3b were reproduced to its displayed precision from the supplied individual logits using Youden thresholds.
- The Nomogram bootstrap percentile bounds are 0.5647965558 and 0.7121421960, yielding the corrected three-decimal CI above.
- The 15 TCIA Harrell C-index estimates checked across five models and RFS, DSS and OS were reproduced to the reported three-decimal precision using the restored risk scores.
- All 27 worksheet names and all records were retained. Cell values and formulas outside the specified corrections were checked against the original author-supplied workbook.

The Figure 3b calculation uses the positive-class softmax probability, selects the first maximum of TPR minus FPR returned by `sklearn.metrics.roc_curve`, and assigns a positive prediction when the probability is greater than or equal to that threshold. Threshold selection is performed separately for each model, task and cohort.

Supporting files:

- [Figure 3b before-and-after values for the 85 changed cells](figure3b_changed_values.csv)
- [Thresholds for all 30 model/task/cohort combinations](youden_thresholds.csv)
- [SHA-256 checksum of the updated workbook](SHA256SUMS.txt)

## Publication status

This is an author-supplied repository update, not a journal-issued correction or a replacement of the journal-hosted publication record. We are preparing the corresponding materials for the Nature Communications editorial office. This update covers the four Source Data items described above; the verification statements are limited to the checks specified here. Any subsequent revised figures or Supplementary Information will be separately identified and versioned.
