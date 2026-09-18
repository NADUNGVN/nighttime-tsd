# Precision-head TensorRT FP16 feasibility smoke v1

Status: **local preparation complete; actual server smoke is NO-GO until Astra
reviews this implementation/protocol**.

This package closes one untested boundary after the accepted CPU source/export
bridge: whether each accepted ONNX graph can parse, build and execute with the
intended server TensorRT runtime while preserving the accepted output
contract. It is a bounded feasibility smoke, not an accuracy result, timing
benchmark, scored confirmation, or precision matrix.

## Locked scope

- Frozen YOLOv8n and YOLO26n checkpoints and the accepted ONNX binaries are
  read-only inputs. Their existing identity and SHA256 bindings are checked;
  no ONNX export, retraining, calibration, or precision-head override is
  allowed.
- The two models run sequentially in isolated child processes on one server
  GPU. A fresh private temporary build workspace and an empty in-memory timing
  cache are used for each model. The serialized engine is never written to the
  publishable output root.
- The fixture is exactly the first eight distinct U42 train images in the
  accepted manifest order:

  `00006, 00009, 00028, 00036, 00054, 00061, 00098, 00104`.

- Each model has one independent FP16-enabled TensorRT build, eight TensorRT
  application enqueues and eight CPU ONNX Runtime reference calls. Across the
  smoke this is exactly 2 builds, 16 TensorRT enqueues and 16 ORT calls. There
  are zero native forwards, warmups, retries, calibration batches, dev/test
  captures and matrix runs.
- Input is batch 1, `640x640`, one float32 `images` tensor with shape
  `[1,3,640,640]`. The same traced tensor bytes are supplied to TensorRT and
  the CPU ORT reference.

## Model output and application contracts

| Model | Parser/engine output | Application route | Comparison boundary |
|---|---|---|---|
| YOLOv8n | `output0`, float32, `[1,7,8400]` | Installed Ultralytics class-aware NMS and `scale_boxes` route | Raw output and ordered application diagnostics; descriptive only |
| YOLO26n | `output0`, float32, `[1,300,6]` | Installed `end2end=True` filtering route | Fixed row index; no rematching, second NMS or sorting |

The YOLO26n output is a Top-K/fixed-row representation. Agreement of row
values is not asserted to mean identical detection membership. The earlier
strict source numeric/localization `FAIL` verdicts remain unchanged.

## Builder and runtime contract

The child configures TensorRT with:

- workspace limit `4 GiB`;
- builder optimization level `3`;
- average timing iterations `1`;
- FP16 enabled, INT8 cleared, TF32 cleared;
- detailed profiling verbosity;
- an empty timing cache created in memory and not serialized or reused;
- no layer precision override and no calibration object.

The runner records parser return/errors, parser IO, builder flag observations,
serialized-engine hash/size, engine IO, and the available engine inspector JSON
when exposed by the runtime. “FP16 enabled” is only a builder setting; the
report must not claim that every layer executed in FP16. The engine bytes are
private and are not an artifact for cross-device reuse.

The intended server environment is the already reviewed environment:

`torch 2.5.1+cu121`, `ultralytics 8.4.102`, `tensorrt 10.16.1.11`,
`numpy 2.4.4`, `pycocotools 2.0.10`, CUDA `12.1`, and one selected GPU.

The parent records UUID, model, driver, P-state, temperature, power, clocks and
memory from `nvidia-smi`. The child checks the same GPU UUID/name at start,
after build and after execution. This is a shared lab server: the protocol
does not claim isolation. Current desktop processes may be allowed only by
explicit operator confirmation of the exact current PID/path against the
narrow shared allowlist. An exact operator-confirmed background workload may
also be recorded. Unknown or unverifiable workloads block dispatch; a new
workload observed during/after the run is preserved as a review violation.
No process is killed or paused, and no permissions, clocks or power limits are
changed.

## Producer chain and hard validity

The child performs this fixed chain for each model:

1. verify the frozen checkpoint and accepted ONNX bytes before and after;
2. decode the eight source images with the pinned producer and record
   preprocessing/input hashes;
3. create a CPU ORT session explicitly using `CPUExecutionProvider` and call
   it once per fixture;
4. parse/build/deserialize one private TensorRT engine with the locked flags;
5. enqueue the same input once per fixture, synchronize completion, validate
   shape/dtype/finite output and record output hashes;
6. apply only the locked model-specific application route for bounded
   diagnostics.

Hard validity requires identity/hash/input/output semantics, successful parse,
successful build and deserialization, successful dispatch/completion and
finite outputs. A failure writes model-scoped `failure.json` and preserves
partial JSON/JSONL/log files; there is no silent resume and no retry.

Raw TensorRT/ORT arrays, checkpoints, ONNX binaries and engine binaries are
never publishable. JSONL records contain hashes, shapes, dtypes, finite flags,
ordered detections and the descriptive raw-output comparison. No AP, official
test or timing statistic is computed by this smoke. Each child materializes the
exact input bytes and the one ORT reference bytes in its private temporary
directory, records their byte/hash evidence, and deletes that directory after
the model finishes.

## Numerical interpretation

The raw TensorRT-versus-ORT comparison policy is fixed before execution:
shape, float32 dtype and finite values are hard checks; max/mean absolute
differences and exact-equality indicators are descriptive. No equivalence
tolerance is applied because no justified application-equivalence margin was
pre-registered. A completed smoke therefore does not prove numerical
equivalence, INT8 validity, calibration validity, or deployment superiority.

## Candidate server invocation (pending review)

The following is a candidate only. Replace the two desktop placeholders with
the exact current rows from the server snapshot; do not reuse historical PIDs
without checking them. Add exact background confirmations only if applicable.
Each command is foreground and must be run as one line.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master
```

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/run_precision_head_trt_feasibility.py --readiness-root results/measurement_audit_v1/server_precision_head_confirmation_readiness_v2 --graph-audit-root results/measurement_audit_v1/precision_head_confirmation_graph_audit_v4 --onnx-root results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2 --out-dir results/measurement_audit_v1/precision_head_trt_feasibility_v1 --model all --device 0 --confirm-desktop-process DESKTOP_PID_1=/snap/snapd-desktop-integration/CURRENT_REVISION/usr/bin/snapd-desktop-integration --confirm-desktop-process DESKTOP_PID_2=/snap/snapd-desktop-integration/CURRENT_REVISION/usr/bin/snapd-desktop-integration
```

The placeholder command is not authorized for execution yet. The operator
must first report the exact output of the server preflight and then receive
the reviewed command with the current PID/path values.

## Publishable artifact contract

After a reviewed server run, push only these files under
`results/measurement_audit_v1/precision_head_trt_feasibility_v1/`:

- `smoke_plan.json`, `smoke_manifest.json`, `report.md`;
- `logs/yolov8n.log` and `logs/yolo26n.log`;
- for each model, `model_report.json` or `failure.json`,
  `smoke_records.jsonl` and `preprocess_trace.jsonl`.

Luna will audit the canonical Git blobs, inventory, model order, exact call
counters, identity/hash bindings, parser/builder/engine IO, ORT provider,
finite flags, telemetry and external-workload status. The audit will preserve
historical FAIL verdicts and will not open a scored matrix automatically.

## Local boundary

Local verification covers only pure contract checks and external runtime
doubles for builder/parser/engine IO, provider rejection, GPU identity,
two-child orchestration, timeout and partial preservation. Local tests do not
import TensorRT, import CUDA, load a frozen model, execute real ORT graphs,
build an engine, use a GPU or install packages. Passing these tests is not
TensorRT end-to-end verification.
