# Figure 5 and Supplementary Table 25 update - 16 November 2026

This directory contains author-corrected figure and supplementary files for [RenalCLIP](https://doi.org/10.1038/s41467-026-74175-w).

## Downloads

- Figure 5: [PDF](Figure5_corrected.pdf?raw=true), [editable SVG](Figure5_corrected.svg?raw=true), [PNG preview](Figure5_corrected.png?raw=true).
- [Complete corrected Supplementary Information (70 pages)](SI_corrected.pdf?raw=true).
- [Corrected Supplementary Table 25 only](Supplementary_Table_25_corrected.pdf?raw=true).
- [Full-precision thresholds, eligible sample sizes and group counts](Supplementary_Table_25_full_precision.csv).
- [File checksums](SHA256SUMS.txt).

## Figure 5a

The image-to-text recall panel had an inconsistency between its plotted geometry and displayed y-axis scale during figure assembly. The corrected panel was regenerated from the original values and bootstrap intervals with a y-axis range of 0.0 to 0.3, and reinserted into the complete figure. The largest upper 95% CI is 0.2985074615 (RenalCLIP, Zhangye, Recall@5), so no interval is truncated. This correction changes the graphical presentation, not the underlying recall estimates or confidence intervals. The complete Figure 5 files were exported by the authors from the updated Illustrator artwork.

## Supplementary Table 25

The former caption stated that the same median risk-score thresholds applied identically to RFS, DSS and OS. The actual Kaplan-Meier analysis calculates a separate median within the eligible patients for each cohort and endpoint. The corrected table therefore lists RFS, DSS and OS separately and reports the eligible sample sizes. The risk groups follow the implemented rule: high risk for a score above the median, and low risk otherwise.

All 90 cutoffs were recalculated from the original full-precision risk scores for the six cohorts, three endpoints and five models. This also corrects the Xiamen/Merlin/RFS entry from 1.590 to -0.886516. Values in the PDF are displayed to six decimal places; the CSV preserves full precision and includes the low- and high-risk group sizes.

Only page 41 was replaced in the complete Supplementary Information PDF. Its other 69 pages retain identical extracted text and rendered page content to the original author-supplied SI PDF.

## Related Source Data update and publication status

The [updated Source Data workbook and its change log](../2026-11-16-source-data/README.md) are provided separately.

These are author-supplied repository updates, not a journal-issued correction. **We will contact the publisher.** The journal-hosted publication record has not been replaced by this repository update.
