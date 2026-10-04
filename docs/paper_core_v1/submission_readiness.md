# Submission readiness — core paper v1

Status: **not submission-ready; substantive draft + cross-model evidence
pending.** Snapshot: 2026-10-04. This checklist distinguishes research
readiness from a journal's ranking and does not estimate acceptance probability.

## Candidate journals (provisional)

| Candidate | Scope fit | Current ranking evidence | Readiness/risk |
|---|---|---|---|
| **Journal of Imaging** | Plausible for a carefully framed imaging/object-detection empirical paper. The work must present an interpretable research finding rather than a result catalogue. | A prior repository source audit cites the publisher's 2026-06-25 announcement reporting 2025 JIF 3.8 and Q2, rank 18/39 in *Imaging Science and Photographic Technology*: [announcement](https://www.mdpi.com/journal/jimaging/announcements/17195); [scope](https://www.mdpi.com/journal/jimaging/about). This pass received HTTP 403 from both MDPI endpoints; these figures are **not revalidated here**. Confirm ranking system/year/category, current APC and open-access terms with the publisher or institutional library before relying on them. | The manuscript is not ready: v2 confirmation is pending, cross-model transfer is unknown, and the novelty must be distinguished from detection PTQ/robustness prior art. |
| **Journal of Real-Time Image Processing** | Potentially relevant only if the final paper substantively explains end-to-end runtime/implementation and an accuracy–latency trade-off. Official publisher scope URL: [Springer Nature aims and scope](https://link.springer.com/journal/11554/aims-and-scope). | **Quartile not verified.** The endpoint returned HTTP 200 with a `cookies_not_supported` error redirect; page content was not validated. No Q2 claim should be attached until the institution confirms ranking system, year and category. | Current timing evidence is a shared single RTX 8000 YOLO11n diagnostic, not a cross-model real-time deployment evaluation. On current evidence, fit is conditional and weaker than the title may suggest. |

These are two scope candidates, not a guaranteed-Q2 list. The user/coauthors
must specify the accepted ranking source (e.g. JCR/JIF versus SJR/CiteScore),
year and category. One journal can have different quartiles by category and
ranking system. Confirm institutional eligibility, APC/funding, article type,
word/page limits, data/code policy and editorial scope before selecting.

## Evidence and manuscript gates

- [ ] Operator completes fresh v2 preflight and one conditional-GO attempt;
      do not dispatch if runtime, inputs, output root, GPU identity or current
      workload confirmations fail.
- [ ] Audit every one of the 84 scheduled builder invocations and 78 captures;
      verify all source/checkpoint/ONNX/calibration/dev/XML/hash bindings,
      canonical artifact inventory, exact counters, failure/cleanup evidence,
      timing-cache input/output, telemetry and lifecycle status.
- [ ] Run the locked analyzer on the accepted artifact only. Publish all
      primary/control/FP16 contrasts, COCO/XML full/XS/S metrics, valid/undefined
      bootstrap counts, within-selection build SD and between-selection-mean
      SD. Do not treat nine engines as nine independent calibration selections.
- [ ] Preserve strict numeric/localization FAIL history and distinguish
      application-level AP from raw numerical equivalence.
- [ ] Integrate LUNA-EDGE material only from its reviewed report/data, or leave
      the Edge subsection explicitly pending. Do not edit Edge-owned runtime or
      claim offline status without confirmation.
- [ ] Finalize bibliography from primary sources, including DOI, publication
      status, page/article numbers and exact version of preprints.
- [ ] Confirm the CCTSDB2021 archive’s exact release/version and required
      attribution/license text against the official dataset paper and release
      README; the paper citation is present, but archive-specific terms were
      not independently checked in this manuscript audit.
- [ ] Ask coauthors to approve authorship/order, contributions, corresponding
      author, dataset/checkpoint sharing rights, reproducibility repository,
      and the claim strength.
- [ ] Institutional/coauthor decision on ranking system, year/category,
      journal scope, APC and publisher policy.
- [ ] Rebuild every final figure/table from versioned source rows and archive
      code, environment, CSV inputs, captions and checksums.
- [ ] The present Figure 1 is `rendered_draft_review_required`: canonical
      Icarus style/critique tooling is unavailable; complete independent
      figure review when canonical tooling is available before submission.
- [ ] Perform an independent scientific and language review; remove all
      prospective/pending language only after the corresponding artifact is
      audited.

## Claim decision before v2

Current draft status: **mixed / incomplete for cross-model generalization**.
YOLO11n evidence supports a conditional precision-head AP association and shows
nontrivial build variation in one repeated-build study; it does not yet show
that the head effect is architecture-independent, causally isolated, or a
deployment recommendation. The feasibility smoke proves only that a bounded
FP16 TRT path executed. The source/ONNX AP bridge coexists with strict numeric
FAILs. No journal submission decision should be made from the pilot alone.

## Release checklist

Before submission, verify no private TensorRT engine, calibration/timing cache,
raw tensor or restricted checkpoint is accidentally included in public
artifacts. Preserve data provenance, code revision and exact evaluator
version. Document shared-GPU telemetry as sampled evidence and avoid claims of
isolation, production reliability, energy efficiency or cross-device speedup
unless separately supported.
