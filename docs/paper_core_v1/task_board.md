# Paper core v1 — task board

Updated: 2026-10-04. Owner: LUNA-SERVER. Astra is the integrated reviewer;
the user is the only SERVER-01 operator. No Luna SSH or server execution.

| ID | State | Workstream | Owner | Dependency / evidence | Next action |
|---|---|---|---|---|---|
| O1 | `operator_pending` | Deliver fresh SERVER-01 preflight and confirm checked-out revision, frozen inputs, runtime, process paths, disk and v2-root absence | User | A2L-054 conditional GO; A2L-055 Section A read-only command issued | User returns unedited output; desktop PID/path confirms identity only and does not authorize competing compute |
| O2 | `operator_pending` | Create fresh CPU plan, validate its bindings, then run the one approved v2 attempt foreground | User | O1 clean; output root absent; exact reviewed executable bytes; accepted input hashes | Run CPU plan first; only after its successful output, run one 84-builder/78-capture attempt; stop on any failure |
| E1 | `done` | Build-repeat, paired ablation, latency, bridge and smoke source inventory | LUNA-SERVER | Existing committed result artifacts | Keep exact evaluator, sampling unit and limitations attached to each row |
| E2 | `done` | Reproducible table extraction and source hashing | LUNA-SERVER | `scripts/export_paper_core_evidence.py`; `results/paper_core_v1/tables/` | Regenerate after scoped commit; preserve source artifact bytes |
| F1 | `done_local_review_required` | Render Figure 1 from committed CSVs; register hashes/exports and complete manual draft critique | LUNA-SERVER | `outputs/figures/`; local CPU measurement environment; canonical Icarus package/gate unavailable | Release label remains `rendered_draft_review_required`; canonical independent critique remains outstanding |
| W1 | `done_local_review_required` | Claim-audited core manuscript, evidence ledger and descriptive tables | LUNA-SERVER | `docs/paper_core_v1/manuscript_audit_20261004.md`; cross-model evidence pending | Integrate only after server v2 audit; keep current pending language |
| W2 | `done_with_open_admin_items` | Focused related work and primary-source/reference audit | LUNA-SERVER | arXiv/CVF/institutional-repository and DOI-registry checks; publisher page/ranking limits documented | Coauthors/institution must resolve ranking system/year/category, APC and archive-specific dataset terms |
| A1 | `ready` | Audit server artifact inventory, provenance, lifecycle and all 84/78 design cells | LUNA-SERVER | O2 artifact commit | Pull without reset/clean; compare canonical Git blobs and run audit before analysis |
| A2 | `ready` | Locked pooled AP, paired intervals and two-axis build/selection variability | LUNA-SERVER | A1 passes; accepted analyzer/helpers | Execute on committed output only; any incomplete cell remains incomplete, with no replacement run |
| W3 | `ready` | Integrate cross-model results and prepare one complete Astra packet | LUNA-SERVER | A2 plus pending Edge-owned section if available | Recommend supported/mixed/incomplete claims and list remaining submission issues |
| W4 | `ready` | Coauthor/institutional journal choice, ranking system/year/category, APC and authorship confirmation | User + coauthors | Draft and publisher records | Decide administratively; no acceptance-probability or Q2 guarantee |

## Milestones

1. **Operator-ready + discovery draft:** local claim audit and figure QA are
   complete, but Figure 1 remains `rendered_draft_review_required`; executable
   A2L-054 revision is accepted; a fresh user snapshot is pending and no v2
   result is present.
2. **Matrix audited:** not started; requires actual v2 artifact.
3. **Confirmation analyzed:** not started; requires complete valid audit.
4. **Manuscript integrated:** draft underway; cross-model and Edge findings are
   not yet incorporated.
5. **Submission review:** not ready until evidence, bibliography and
   institutional journal requirements are resolved.

Current terminal distinction: documentation/table extraction is local work;
neither signifies GPU execution nor a scientific confirmation verdict.
