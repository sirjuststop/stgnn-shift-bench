# Camera-ready changes, 2026-11-03

- Added the authors, affiliations, email addresses, funding acknowledgments, and competing-interests statement; enabled the NeurIPS 2026 E&D final format.
- Updated temporal comparisons throughout the paper and Figure 1 to distinguish normalized/plain MAE from masked MAE in mph. The two conventions can reverse the weekday/weekend gap.
- Added leave-one-out correlations and an exact permutation test for the original four-model routing comparison.
- Integrated the additional routing, corruption, severity, and few-shot aggregate tables reported in the authors' rebuttal.
- Corrected the mask-aware standard deviation, paired false-positive/false-negative penalties, and several seed-count and rounding descriptions.
- Clarified the role of imputation and the scope of the recommendation artifact.

Existing experiment protocols and saved results were retained. This revision did not run new training or inference. Reaggregated results and author-reported aggregates have separate provenance records in `generated/`.

The source package includes all inputs needed to compile the paper and to regenerate its revised figure and tables. The main text is 9 pages; acknowledgments, references, appendix, and checklist bring the PDF to 29 pages.
