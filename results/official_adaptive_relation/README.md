# results/official_adaptive_relation/

The V7 integration wave (Apr 13-16): added three **official adaptive baselines** — **AGCRN**, **MegaCRN**, **STAEformer** — and merged them with the canonical 4-backbone bank. This wave produced the paper's current headline numbers.

## CANONICAL (use these)

| File | Role |
|------|------|
| **`UPDATED_MERGED_COMPARISON_TABLE_2026-04-16.csv`** | **Source of truth** — 8 backbones × 3 shifts, merged master table |
| **`UPDATED_MERGED_COMPARISON_TABLE_2026-04-16.json`** | Same content, JSON format |
| `OFFICIAL_ADAPTIVE_FULL_EVAL_2026-04-16.json` | 185 KB complete raw eval snapshot |
| `FULL_OVERNIGHT_EVAL_SUMMARY_2026-04-16.{csv,json,md}` | High-level summary of the overnight eval run |
| `official_adaptive_shift_eval.json` | Per-shift breakdown for the three official adaptive models |
| `official_adaptive_unified_registry.json` | Merged registry: canonical bank + official-fix additions |
| `official_adaptive_training_audit.{csv,json}` | Per-backbone training metadata (epochs, early-stop, seeds) |
| `SENSOR_SEVERITY_CURVES_2026-04-16.json` | Sensor-dropout severity curves |
| `TEMPORAL_RERUN_SUMMARY_2026-04-16.json` | Temporal-shift rerun results |
| `CITY_RERUN_SUMMARY_2026-04-16.json` | City-transfer rerun results |

## SUPERSEDED (kept for audit trail, do NOT use in paper)

| File | Why superseded |
|------|----------------|
| `merged_comparison_table.csv` (Apr 15) | Pre-update version; replaced by `UPDATED_MERGED_COMPARISON_TABLE_2026-04-16.csv` |
| `merged_comparison_table.json` (Apr 15) | Pre-update version; replaced by `UPDATED_MERGED_COMPARISON_TABLE_2026-04-16.json` |

The Apr 15 versions were produced before the Apr 16 claim-freeze and anomaly audit. They are retained so the audit trail is complete, but the paper **must not** reference them.

## Audit / Memo Trail (chronological)

Codex memos documenting the V7 wave (Apr 13–16):

**Apr 13 — integration start:**
- `AGCRN_INIT_DIAGNOSIS.md`, `INTEGRATION_REPORT.md`, `STATUS_MEMO.md`, `CLAIM_CALIBRATION_MEMO.md`

**Apr 14 — training + handoff:**
- `CODEX_STATUS_MEMO_2026-04-14.md`, `HANDOFF_2026-04-14.md`
- `CODEX_MASKED_METRIC_CALIBRATION_2026-04-14.json`, `CODEX_PREPROC_PROBE_2026-04-14.json`

**Apr 15 — context + protocol lock:**
- `CODEX_CONTEXT_MEMO_2026-04-15.md`, `CODEX_PROTOCOL_MEMO_2026-04-15.md`, `CODEX_LIMITATION_MEMO_2026-04-15.md`, `CODEX_RESULT_MEMO_2026-04-15.md`, `CODEX_QUICK_STATUS_2026-04-15.md`, `CODEX_PAPER_COMPARISON_2026-04-15.md`, `CODEX_PAPER_EDIT_MEMO_2026-04-15.md`, `CODEX_CLAIM_CALIBRATION_2026-04-15.md`, `CODEX_COMMAND_LOG_2026-04-15.md`, `COMMANDS_RUN.md`

**Apr 16 — anomaly audit + claim freeze:**
- `ANOMALY_AUDIT_2026-04-16.md` (15 KB detailed anomaly log)
- `CLAIM_FREEZE_2026-04-16.md` — **headline claims frozen**: 4.3× GWNet/DCRNN dropout gap, 8-9× MegaCRN/STAEformer temporal fragility, 75.2% DCRNN cross-city win rate, ρ=+0.979 routing-sparsity correlation
- `CLAIM_FREEZE_REFRESH_2026-04-16.md` — refresh run confirmed stability
- `PROTOCOL_VERIFICATION_2026-04-16.md`, `METRIC_RECONCILIATION_2026-04-16.md`, `TARGETED_RERUN_LOG_2026-04-16.md`
- `V7_REVISION_MEMO_2026-04-16.md` — paper-level revision notes for V7
- `COMMAND_LOG_2026-04-16.md`, `PAPER_EDIT_SUGGESTIONS.md`
- `FULL_OVERNIGHT_EVAL_SUMMARY_2026-04-16.md`

## Supporting Files

- `logs/` — training logs for the three official adaptive models
- `official_fix_registry_megacrn.json`, `official_fix_registry_staeformer.json` — per-backbone official-fix registries
- `official_adaptive_summary.json` — high-level summary JSON

## Downstream References

- Paper: all headline claims in `paper/main.tex` sections §4 (taxonomy) and §8 (appendix) trace to the canonical table listed above.
- Audit: `review-stage/AUTO_REVIEW.md` (Apr 18) verified the numbers still reproduce; `PAPER_CLAIM_AUDIT.md` (Apr 18) flagged 5 aggregation-layer issues (4 deferred to May 6 cleanup window, 1 fixed Apr 18 01:26).
