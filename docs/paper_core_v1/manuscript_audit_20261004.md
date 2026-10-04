# Manuscript claim, table and reference audit — 2026-10-04

Status: local evidence audit complete for the existing paper-core draft;
cross-model confirmation remains `operator_pending`. This audit does not
convert a plan, partial run, CPU application bridge or feasibility smoke into
an accuracy result. It is not submission approval.

## Evidence and reproducibility checks

- `results/paper_core_v1/tables/evidence_source_manifest.json` records eight
  existing study JSON sources and six generated CSVs. All eight source-file
  SHA-256 values and all six CSV SHA-256 values were rechecked against the
  manifest and matched. Hashes refer to worktree bytes; the manifest separately
  records canonical Git blob IDs at its generation revision.
- CSV row counts: 40 precision-head points, 120 paired-contrast rows, 12
  build-variability rows, 5 latency rows, 2 source/export bridge rows, and 1
  TensorRT feasibility accounting row. Units, evaluator names, source paths,
  and source hashes are retained in the row data.
- Figure inputs are the committed paired-contrast and build-variability CSVs.
  Their manifest hashes match the current source files. The figure was
  regenerated using the already installed CPU interpreter; no inference,
  export, engine build, calibration, or GPU action was performed.
- Focused figure tests: **4/4 PASS**. They check the locked row selection,
  primary contrast/source values, bounded three-format export, byte-stable
  repeat rendering, and review status. This is a local test gate, not the
  canonical Icarus critique gate.

## Manuscript claims crosswalk

| Manuscript claim | Source/evaluator and sample unit | Audit result and qualification |
|---|---|---|
| YOLO11n repeated-build full AP50–95 range 1.93 pp; XS AP50 range 7.82 pp | `server_uniform_build_repeat_v1/repeat_summary.json`; COCO/XML evaluator; three builds, one checkpoint and one Uniform selection | Matches the source table: 1.931478 pp and 7.819790 pp. These are observed ranges, not confidence intervals or calibration-selection variance. |
| Table 1 build means, sample SDs and ranges | Same build-repeat source; `build_variability.csv`; `ddof=1` | Displayed values match when rounded to three decimals. The table labels build SD and observed range and does not imply independent calibration samples. |
| YOLO11n all-size head-ablation points | `precision_head_paired_analysis_v1/point_estimates.json`; COCO/XML; same 1,636 dev images / 2,706 XML instances | Baseline INT8 66.542979%, bbox-FP32 71.969322%, classification-FP32 70.170042%, both-FP32 75.049692%, FP16 75.654611%; manuscript rounding is correct. Ultralytics values remain separately labeled. |
| Primary both-FP32 minus baseline-INT8 effect | `contrast_ci.json`; paired image bootstrap, 1,000 PCG64 resamples, seed 20260916 | Full AP50–95 +8.506713 pp, 95% percentile interval [+7.710367, +8.999865], 1,000 valid / 0 undefined draws. All six registered size strata have positive intervals; they remain conditional on the saved captures. |
| Both-FP32 versus FP16; branch/size contrasts | Same paired-contrast source; same paired resampling unit | Both-FP32 minus FP16 is −0.604919 pp [−0.936105, −0.367044]. This is not a non-inferiority test. Branch contrasts are shown as diagnostics, not independent causal effects; all registered contrasts remain in the 120-row CSV. |
| Figure 1 comparison | Same primary contrast plus separate three-build repeat table | Figure panel (a) plots the paired interval; panel (b) plots max–min spans with absolute AP endpoints. The studies, uncertainty types, and sample units are explicitly distinct. No mechanism or cross-model effect is inferred. |
| YOLO11n latency means and 39-session count | `server_yolo11n_precision_head_latency_v1/latency_summary.json`; synchronous batch-one end-to-end predict wall time | There are 39 sessions and 39,000 measured calls (200 warmups + 1,000 measured calls per session). Arm means match `latency_summary.csv`; pooled calls are not independent builds. Shared-GPU snapshots do not prove isolation; no energy or SLA claim is made. |
| Native/ONNX CPU application bridge | `precision_head_source_export_dev_bridge_v1/` model reports; COCO/XML evaluator | Each model has 1,636 native CPU forwards and 1,636 ONNX CPU calls. AP50 delta is 0.0 pp for both; AP50–95 deltas are +0.0000091 pp (YOLOv8n) and −0.0000106 pp (YOLO26n). These application metrics do not override strict raw numeric/localization FAILs. |
| Strict numeric/localization history | Numeric v1/v2 and localization v2 manifests | v1 stopped at a head/schema contract issue before a numerical verdict. Numeric v2 completed the fixed CPU comparisons and retained strict FAIL for both models; localization follow-up preserved that verdict. YOLOv8n had a small number of tolerance failures; YOLO26n fixed-row boxes/classes diverged despite score-channel agreement. This is not matched-detection localization proof or equivalence. |
| FP16 feasibility smoke | `precision_head_trt_feasibility_v1/smoke_manifest.json` | Exactly two builds total (one/model), 16 TRT application enqueues and 16 ORT CPU references; native forwards, warmups, calibration, dev capture and retries are zero. This proves only the bounded path executed, not accuracy, equivalence, benchmark performance or GPU isolation. |
| Failed confirmation v1 and prospective v2 | `server_precision_head_confirmation_v1/execution_manifest.json`, first child state/log; v2 output root absent in this checkout | v1 attempted one auxiliary calibration build and failed at calibration batch 1 (`18709.jpg != 00006.jpg`); no engine was published and no scored capture completed. Preserve this partial failure unchanged. A2L-054 conditionally authorizes only one fresh v2 attempt; no v2 artifact/result is present here. |

All reported percentage-point effects use differences of percentage-valued AP
points, not relative percent changes. Absolute points, paired contrasts,
build variability, session timing, application bridge and runtime feasibility
remain separate channels. No positive or unfavorable contrast was removed to
make the draft read more favorably.

## Caption, table and reference audit

- Table 1 is a direct rounded view of the six all/XS/S COCO/XML rows in the
  three-build source and retains the `n=3` / one-selection limitation.
- Figure 1 caption states the result, CI method/seed, image and XML-instance
  counts, and the distinct `n=3` descriptive range panel. The body now refers
  to it as “Fig. 1”.
- Citation numbering [1]–[6] is consistent with the focused PTQ, reliability,
  error-analysis and deployment sources. Reg-PTQ pages and DOI are consistent
  with the CVF record and DOI-registry metadata. TIDE proceedings pages and DOI
  are now included from DOI-registry metadata; the Springer chapter page
  remained behind a cookie redirect, so that access limitation is disclosed.
- CCTSDB2021 is now cited to the University of Reading repository record and
  the DOI shown there (`10.22967/HCIS.2022.12.023`), with the official release
  repository linked. The exact archive/version and archive-specific license
  or attribution requirements still need confirmation against the XML archive
  used.
- The COCO paper is cited for the evaluation convention. The manuscript
  explicitly identifies the project’s registered CCTSDB area-bin evaluator;
  it does not call these thresholds native COCO bins.
- Primary arXiv/CVF/University-repository landing records were available for
  the sources described in `related_work.md`. Venue/DOI status for the PTQ
  reliability preprint and InlierQ remains unverified; AgriJetsonBench is
  treated as a preprint. MDPI journal endpoints returned HTTP 403 and the
  Springer scope endpoint returned a cookie challenge; current journal
  ranking system/year/category, APC and scope facts remain administrative
  verification items, not manuscript claims.

## Open items / decision

1. The fresh SERVER-01 snapshot is pending. The user must complete the current
   Section A preflight; after it passes, Luna will bind the CPU plan and one
   foreground scored command to the returned revision, inputs, GPU/process
   state and resources. No SSH is performed by Luna.
2. Only after the user pushes v2 evidence will Luna verify all 84 builder
   invocations / 78 captures, run the locked analyzer, and integrate supported
   results. No plan status is a scientific result.
3. The paper remains a mixed/incomplete cross-model draft and is not
   submission-ready. Coauthors/institution must confirm dataset terms,
   authorship, journal ranking basis/year/category, APC and venue choice.
4. Figure 1 remains `rendered_draft_review_required`: the canonical Icarus
   `paperfig` package and `scripts/critique.py` gate are absent. The local
   shared-style fallback, bounded data/export tests, color and grayscale
   visual checks, and manual four-axis review are documented separately; they
   are not represented as the canonical gate or publication-ready approval.
