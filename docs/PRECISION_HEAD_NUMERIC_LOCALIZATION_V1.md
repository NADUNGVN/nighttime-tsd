# Precision-head numeric localization v1

Status: local implementation/test package for Astra review. This document does
not authorize a server run. The accepted `precision_head_confirmation_numeric_v1`
and `precision_head_confirmation_numeric_v2` verdicts remain immutable; the
strict v2 verdict remains `FAIL`.

## Scope

The diagnostic uses exactly two already observed cases:

- YOLOv8n/U42 image `00009`, the historical two-element box failure;
- YOLO26n/U42 image `00006`, the first canonical fixture image, not a selected
  best/worst case.

It uses the existing preprocessing and input bindings from numeric_v2. The
bounded execution contract is exactly two ordinary native model forwards and
three ONNX session runs: one native and one original ONNX run for each model,
plus one private derived-graph ONNX run for YOLO26n. A child failure preserves
its stage/count state and never causes an automatic extra call.

No new image, checkpoint, reference mode, fusion, threshold, score cutoff,
optimization sweep, AP/test evaluation, TensorRT, export, rebuild, GPU or
matrix is in scope. The strict numeric comparison is reported again as
supplemental evidence but is never replaced or relaxed.

## V8 localization

The accepted YOLOv8n primary is the raw fixed-anchor output `[1,7,8400]`; it
has no Top-K stage in the accepted graph. The diagnostic reports, for indices
`[0,1,8004]` and `[0,1,8014]`, native/reference scalar, ONNX/observed scalar,
absolute error, `1e-5 + 1e-4*abs(reference)`, and error/allowance ratio. It
also reports all current box/score mismatches and the three class scores at
each offender anchor. The native returned raw boxes/scores and the ONNX raw
channel spans are summarized separately. A mismatch remains a strict `FAIL`
even when its absolute magnitude is small.

## V26 localization

The accepted YOLO26n graph is inspected before any derived graph is written.
The private view only adds existing tensors as graph outputs; it does not
modify nodes, weights or attributes and never overwrites the accepted ONNX.
The exposed tensors are:

- pre-TopK merged/transpose/split boxes and class-score tensors;
- first `TopK` values/indices;
- selected-class flattening and second `TopK` values/indices;
- `Div`/`Mod` class arithmetic, `Gather` original-anchor indices and gathered
  boxes/class scores;
- the unchanged original `output0` alongside those additions.

The runner records the actual node attributes for `Split`, both `TopK` nodes,
`GatherElements`, `Div`, `Mod` and related nodes, including `axis`,
`largest`, `sorted` and `fmod` where present. It records shapes, dtypes,
finite/zero counts, magnitudes and hashes for private tensors, while publishing
only bounded summaries.

Before the YOLO26 session, every exposed tensor is required to have the
accepted inferred metadata: merged `[1,7,8400]`, transposed `[1,8400,7]`,
boxes `[1,8400,4]`, class probabilities `[1,8400,3]`, first/second Top-K
values and indices `[1,300]`, gathered class matrix `[1,300,3]`, rank/class
indices `[1,300]`, original anchors `[1,300,1]` and selected boxes
`[1,300,4]`, with float32/int64 dtypes as appropriate. Missing or different
metadata is `unresolved` before any diagnostic session.

For the ONNX side, original anchor indices are direct outputs of the derived
view. For the native side, selected indices are explicitly labelled as
reconstructed from the returned `one2one` preselection tensors using the
installed PyTorch `topk(largest=True, sorted=True)` implementation. They are
not claimed to be directly observed native indices. Each side reconstructs
its own final boxes/classes/scores from its own tensors and indices.

For the ONNX side, the final output is reconstructed from its exposed
preselection boxes/class probabilities and its direct `TopK`/`Div`/`Mod`/
`Gather` indices, then compared with unchanged original `output0`. The direct
gathered class matrix, selected boxes and stage-2 values are checked against
those reconstructions; a copied final output cannot stand in for Top-K or
class-index evidence.

Selection alignment uses exact `(original_anchor, class_id)` pairs only. It
separates:

- same-anchor numerical error before selection;
- same selected set with a different order (permutation);
- different selected-set membership, with overlap and bounded examples;
- unresolved association when shape/index semantics are unavailable.

There is no nearest-neighbor matching, rematching, sorting imposed on the
reference, NMS, clipping or post-hoc score cutoff. Raw logits are not compared
to sigmoid probabilities; only tensors with an explicitly mapped score role
are compared as preselection scores. If the private view changes original
`output0` versus the unmodified session, instrumentation is marked sensitive
and its internals are not used as proof of the original execution.

## Artifact contract

Publishable files are `localization_plan.json`,
`localization_manifest.json`, `report.md`, both model reports or failures,
their bounded JSON partials and two logs. The derived ONNX and full tensors
remain under `models/yolo26n/private/` on the server and are not pushed.
Child `partial_files` are model-owned; the parent owns the complete
publishable inventory. The plan records the canonical Git-blob hashes of the
linked numeric_v2 files, distinguishing them from CRLF checkout hashes:

| File | Canonical Git SHA256 |
| --- | --- |
| `numeric_manifest.json` | `0d28fc3261ad4fa42baa8459c449c50e98777f9c3a43123f288109159523787d` |
| `numeric_plan.json` | `05216d73d69b0a9f5d621aee2fbd3a444d5e6cdc9a11050a4c6a709970c06831` |
| `report.md` | `5bcb70399bb634e6e449ea54ad36850405dae6e19d574afb259e88f74abb56eb` |

## Local test boundary

Tests use deterministic CPU arrays and faithful runtime/graph doubles for
permutation-only selection, tie-driven membership changes, same-anchor box
errors, score differences within tolerance, index/class mapping errors,
instrumentation drift, near-zero diagnostics and child inventory/failure
state. They do not claim a real frozen-model forward or ONNX session until a
separate reviewed server run is authorized.
