# Paper-edge research packet — task board

As of 2026-10-04 (Asia/Saigon). Owner: LUNA-EDGE. Source-server operations remain user-operated; this paper/readiness packet does not supersede the E2L1-030 execution gates.

| Queue | Status | Evidence / dependency | Next action |
| --- | --- | --- | --- |
| Source ONNX CPU reference | `operator_pending`; 0/1,636 | Full foreground command and POSIX mock prerequisite are in [`E2L1-030_SOURCE_CPU_RUNBOOK.md`](../E2L1-030_SOURCE_CPU_RUNBOOK.md). No full-dev source output has been returned. | User runs that exact one-pass command on SERVER-01; return sanitized counters, provider/version receipt and artifact hashes for audit. |
| E2 target full-dev comparison | `operational_hold`; 0/1,636 | E2 was reported offline; no E2 probe/retry was made under E2L1-031. Attempt 1 timed out; attempt 2 built/loaded and used its full three-call smoke allowance, with raw target mismatch. | After source/package/POSIX evidence and restored E2 access, perform only the existing bounded read-only prerequisite check, then follow E2L1-030's conditional GO if every gate passes. |
| Saved-output diagnosis / evidence ledger | `complete_from_existing_evidence` | Accepted E2L1-022/023/024 manifests and CPU replays; raw mismatch remains a FAIL. | Preserve as limitation; do not rerun, change boxes/tolerances or infer AP. |
| Manuscript edge section | `prepared_review_required` | [`edge_section.md`](edge_section.md) and [`evidence_ledger.md`](evidence_ledger.md). | Astra review; replace pending study statements only after complete paired analysis. |
| Paired AP layouts and CPU recipe | `prepared_source_and_target_pending` | [`paired_ap_tables.md`](../../results/paper_edge_v1/paired_ap_tables.md) and [`cpu_reproduction_recipe.md`](../../results/paper_edge_v1/cpu_reproduction_recipe.md). | Populate only from hash-audited 1,636-row source and target evidence. |
| E1/E4/E5 availability and readiness | `read_only_inventory_complete` | Three current inventories, 27/27 commands each; sanitized allowlist and hashes in [`edge_contingency_20261004`](../../results/edge_readiness_v1/edge_contingency_20261004/README.md). | Keep E5 as a prospective option only; do not substitute it for E2 without one scope decision. |
| Private source/target package | `waiting_for_source_artifacts` | No 1,636-image source output or associated per-image input bindings returned. Existing private three-fixture smoke evidence is not the full-dev package. | After operator return, validate the exact private allowlist and hashes; keep model/image/tensor/engine bytes out of public Git. |

Terminal labels for this checkpoint: E2L1-030 `operational_hold`; E2L1-031 `edge_research_packet_review_required`. No full-dev accuracy, deployment, latency or energy result is claimed.
