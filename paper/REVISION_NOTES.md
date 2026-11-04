# Camera-ready v4 review revision, 2026-11-04

- Qualified the abstract and conclusion as associations under silent zero-filling; stated that LOCF and mask-aware routing require identifying failed sensors.
- Added highway loop-detector, three-seed, zero-shot, within-split temporal, and walk-forward limitations, plus the reason for choosing GWNet for the intervention.
- Corrected the GWNet capacity-gap recovery, PEMS04 transfer comparisons, and structured-dropout random baselines. Explained the different hub definitions used by two experiments.
- Identified the original routing script's fixed historical delta references. Reusing the main-table primary-mask deltas gives four-model Pearson 0.879649 instead of 0.978827. Concentration standard deviations now use the population convention.
- Listed all six retained concentration means, with unavailable GTS/DGCRN normalized deltas explicitly marked. The six-model coefficients remain historical aggregates. Replaced unsupported historical p-values in the table with approximate two-sided Pearson t tests from rounded correlations (df=4): 0.001, 0.086, and 0.180.
- Disclosed missing supplementary raw outputs and configurations in Appendix B.2 and table captions. Five-seed and peak/off-peak remarks are labeled as record-only observations. The canonical-bank mph range has no retained target-mask convention and is not treated as a matched masked-mph comparison.
- Qualified the normalized-metric recommendation table, looseness of the local exposure bound, and causal language; repaired cross-references and added/updated ASTGCN, DSTAGNN, and DGCRN citations.
- Changed checklist items 4--6 to No with explanations because full supplementary reproducibility and training details cannot be claimed. This is a disclosure correction, not a new experimental result.

No new training or inference was run. The original experiment results and v3
delivery are preserved. The v4 PDF has 29 pages, with the main text on pages 1--9.

# Earlier camera-ready changes, 2026-11-03

- Light language edit: replaced abstract labels and conversational metaphors with direct descriptions, and shortened repeated summaries. Numbers, equations, citations, tables, figures, and the approved contribution bullets are unchanged.
- Follow-up style revision: simplified the author affiliations to university, city, and email; rewrote each contribution as one sentence beginning with "We", without bold lead-ins or numerical results.
- Added the authors, affiliations, email addresses, funding acknowledgments, and competing-interests statement; enabled the NeurIPS 2026 E&D final format.
- Updated temporal comparisons throughout the paper and Figure 1 to distinguish normalized/plain MAE from masked MAE in mph. The two conventions can reverse the weekday/weekend gap.
- Added leave-one-out correlations and an exact permutation test for the original four-model routing comparison.
- Integrated the additional routing, corruption, severity, and few-shot aggregate tables reported in the authors' rebuttal.
- Corrected the mask-aware standard deviation, paired false-positive/false-negative penalties, and several seed-count and rounding descriptions.
- Clarified the role of imputation and the scope of the recommendation artifact.

Existing experiment protocols and saved results were retained. This revision did not run new training or inference. Reaggregated results and author-reported aggregates have separate provenance records in `generated/`.

The source package includes all inputs needed to compile the paper and to regenerate its revised figure and tables. The main text is 9 pages; acknowledgments, references, appendix, and checklist bring the PDF to 29 pages.
