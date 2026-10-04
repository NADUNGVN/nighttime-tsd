# Paper-core reproducible tables

These files are derived only from already committed study summaries. They do
not run a model, import CUDA/TensorRT/ONNX Runtime, rebuild an engine, or compute
a new bootstrap. Every CSV row carries a source path and raw-worktree SHA-256;
the manifest adds each committed source's canonical Git blob object ID, so
line-ending conversion cannot be mistaken for a canonical artifact hash.

Regenerate from the repository root with:

```powershell
python scripts/export_paper_core_evidence.py
```

Outputs are in `results/paper_core_v1/tables/`:

- `precision_head_points.csv`: captured YOLO11n absolute points, separated by
  evaluator and size bin.
- `precision_head_contrasts.csv`: all accepted paired contrast intervals,
  copied from the stored 1,000-draw result (not recomputed here).
- `build_variability.csv`: mean, ddof=1 sample SD and observed range for the
  three-build Uniform study.
- `latency_summary.csv`: session-pooled end-to-end timing summaries.
- `source_export_bridge.csv`: application-level CPU native/ONNX AP deltas.
- `trt_feasibility.csv`: scope/call accounting only, not performance.
- `evidence_source_manifest.json`: generator/source/output hashes and scope.

Use the CSVs as inputs for plots/tables, not as a substitute for the linked
JSON source artifacts. AP and latency units/endpoints are not commensurate.
Do not interpret the 39,000 timed image calls as 39,000 independent engine
builds. No confirmation-v2 values are included until the operator artifact has
been audited. Figure 1 is reproducibly rendered from the paired-contrast and
build-variability CSVs with:

```powershell
& 'D:\Research\paper\local\measurement_audit_env\Scripts\python.exe' scripts/fig1_precision_head_effect_and_build_spread.py
```

The generator writes PDF, editable SVG, 600-dpi PNG and a hash-bearing
manifest under `outputs/figures/`. It uses the committed repository-local
style/export fallback because canonical `paperfig`/`critique.py` are absent.
The status is `rendered_draft_review_required`, not a canonical Icarus-gate
pass or publication-ready figure. See the figure contract and manual four-axis
QA record in `docs/paper_core_v1/`.
