# Jetson adapter and protocol smoke v1

**Status:** proposed local contract; not executed on a device. This package is
for E2 (Jetson Xavier NX) first, with E3 and E5 as later compatibility
candidates. It does not authorize model transfer, engine build, engine export,
inference, benchmark, or configuration changes.

## Frozen reference and target boundary

The engineering reference is YOLO11n FP16 using
`results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt`, SHA-256
`3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8`. The
server RTX engine hash in the deployment config is provenance only; it is not
copied to or used as a Jetson artifact. The proposed target path is a
target-native TensorRT FP16 build from the frozen checkpoint, after a separate
review and target-side disk/dependency check.

The local adapter imports no CUDA, TensorRT, NumPy, image decoder, or model
runtime. Runtime, execution context, stream, and buffer operations are injected
behind small interfaces and tested with CPU mocks. The buffer manager owns
allocation pointers, owner/lifetime identifiers, host staging, and copy-stream
identity; the adapter never creates a device address.

| Candidate | Observed runtime inventory | Adapter API candidate | Compatibility decision |
|---|---|---|---|
| E2 Xavier NX | TensorRT 8.5.2.2 | binding-list `execute_async_v2` | target smoke validation required |
| E3 AGX Xavier | TensorRT 8.5.2.2 | binding-list `execute_async_v2` | target smoke validation required |
| E5 Orin Nano Super | TensorRT 10.3.0 | named addresses + `execute_async_v3` | target smoke validation required |

These are version-matched Python adapter candidates, not claims that a target
engine has already exposed those calls. TensorRT 10 documents the named tensor
address and `enqueueV3` execution model; NVIDIA's migration guidance describes
the corresponding 8.x-to-10.x API pattern. The archived 8.5 release notes and
current Python API documentation are retained as the version/API references:
[TensorRT 8.5.3 release notes](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-853/pdf/TensorRT-Release-Notes.pdf),
[TensorRT 10.3 Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1030/pdf/TensorRT-Developer-Guide.pdf),
[8.x to 10.x migration patterns](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-8x-to-10x-c-api-patterns.html),
and [TensorRT Python API documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html).

## Contract

- Input is exactly batch 1, 3 channels, 640x640, contiguous NCHW. Preprocessing
  is OpenCV-style decoded BGR -> RGB -> aspect-preserving letterbox -> float
  scaling to `[0, 1]`; decoder work is outside any future inference timer.
- Input and every output binding must declare name, role, static positive shape,
  dtype (`float16` or `float32`), location, and exact byte count before enqueue.
  The staging dtype is explicit and must equal the engine input dtype; the
  frozen model's FP16 precision is not permission to silently cast buffers.
- The adapter records synchronization boundaries before H2D copy, before
  enqueue, after enqueue, and after D2H output copy. H2D, binding-list v2,
  named-address v3, and D2H all use the exact buffer-owner allocation map and
  one validated copy stream. Host output payloads are inaccessible until the
  final stream completion marks them ready. A host enqueue return or host-side
  timer alone is not completed-device latency.
- E2/E3 use the legacy binding-list candidate; E5 assigns every named tensor
  address before the v3 enqueue candidate. Any missing API, false enqueue,
  hash mismatch, runtime mismatch, or binding mismatch is a structured stop.
- Native output names/shapes/order are captured before postprocessing. For this
  YOLO11n no-NMS engineering contract the single output is `output0` with shape
  `[1,7,8400]`; a generic `[1,84,8400]` positive shape is rejected. Source
  tensor -> binding mapping is explicit and comparison occurs before decoding.
  Exactly one later layer owns NMS; no double NMS. The smoke checks raw output
  contract and numerical tolerances, not AP, FPS, energy, or official-test
  performance.

## Proposed correctness smoke

The current CPU fixture is explicitly synthetic mock data, not CCTSDB training
evidence. It contains three 4x4 PPM byte payloads with IDs
`mock-synthetic-00` through `mock-synthetic-02`; individual content hashes,
derived input hashes, and an order-bound sequence hash are emitted by
`fixture_manifest()`. Before a real smoke, the first three U42 CCTSDB train
images in canonical order and their accepted source content hashes must be
materialized. If that material is unavailable, the real smoke remains blocked;
synthetic bytes must not be relabelled as train images.

Before any target output is observed, the run manifest must freeze:

1. checkpoint SHA-256 and target-built engine SHA-256;
2. target identity, runtime versions, execution API and engine bindings;
3. preprocessing constants and fixture image order/content hashes;
4. raw output tensor names, shapes, dtypes, locations and output ordering; and
5. the following comparison tolerances, recorded before comparison:

   | Check | Proposed acceptance | Reason |
   |---|---|---|
   | every raw value finite | 100% of values finite | catches failed/uninitialized buffers |
   | output names/order/shape | exact equality | binding/reshape errors are not numeric noise |
   | FP16 raw output | `abs <= 5e-3` **or** `rel <= 1e-2` | allows bounded half-precision/runtime round-off while rejecting material drift |
   | FP32 raw output | `abs <= 1e-4` **or** `rel <= 1e-3` | tighter bound when the exposed output is float32 |

   These are engineering-smoke thresholds, not accuracy claims; a future run
   may tighten them from observed reference precision but must not widen them
   after seeing target outputs. No post-hoc tolerance widening.

The reference output and target output must be compared before any decoding or
NMS. If postprocessing is checked, it must use the declared single NMS owner
and preserve the native output contract. Official-test images and AP are out
of scope.

## Proposed operator command and resources

The executable local command is:

    python scripts/edge_readiness/jetson_adapter_smoke.py --target E2 --out-dir results/edge_readiness_v1/e2l1-007/mock-dry-run --dry-run

It runs only the adapter through the accepted CPU/mock harness for both timing
boundaries and emits the proposed manifest/layout; it performs no model load or
forward. A future real target entry point is intentionally not implemented in
this gate. Its required dependencies are target-local JetPack/TensorRT,
target-native engine bytes, accepted checkpoint/fixture material, and the
reviewed output directory. The allowed resource set is the existing
JetPack/TensorRT runtime on the named target, the accepted checkpoint bytes,
the materialized CCTSDB train-only fixture, and a new target-scoped output
path. No server engine transfer is allowed.

### Future real recipe and preflight (not executable in this gate)

The source-reference locations are the frozen checkpoint
`results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt`, deployment/preprocessing
policy `configs/deployment/yolo11n_fp16_trt10.json`, and accepted server FP16
reference evidence under
`results/calibration_method_v1/rtx8000/yolo11n/eval/yolo11n_fp16_reference/`.
The callable source-reference forward is not bundled in this CPU-only package;
the future run must identify and hash it before comparison.

After authorization, the target-native recipe is: verify checkpoint bytes and
source-reference hashes; materialize the first three U42 train images and
content hashes; use the installed target TensorRT major version and the exact
640/NCHW/preprocessing policy; build a new FP16 engine locally on the target
from that checkpoint; inspect and freeze its bindings; run the adapter with the
same derived tensor bytes; compare raw `output0` before decode/NMS; then write
only the scoped manifest/artifacts. The server RTX engine is never an input to
this recipe. E2/E3 require the observed 8.5.2.2-compatible path and E5 the
observed 10.3.0-compatible path, subject to target API validation.

Read-only preflight must check target identity/runtime and Python/TensorRT
availability without installing, disk capacity and output-directory absence,
checkpoint/config/fixture existence plus hashes, and that the scoped output
path is writable without touching any existing artifact. The exact allowed
writes are only:

    results/edge_readiness_v1/e2l1-007/<run_id>/manifest.json
    results/edge_readiness_v1/e2l1-007/<run_id>/input_fixture_manifest.json
    results/edge_readiness_v1/e2l1-007/<run_id>/adapter_events.json
    results/edge_readiness_v1/e2l1-007/<run_id>/output_contract.json
    results/edge_readiness_v1/e2l1-007/<run_id>/failure.json  # failure only

Forbidden mutations and work in this phase are package installation, sudo,
power/clock/fan changes, reboot, export/build, model load/forward, official
test use, and AP/FPS/energy scoring. The adapter package also does not repeat
the accepted device inventory.

## Future artifact layout and stop conditions

An authorized run should write only under:

    results/edge_readiness_v1/e2l1-007/E2/<run_id>/
      manifest.json
      input_fixture_manifest.json
      adapter_events.json
      output_contract.json
      failure.json (only on a failed mock orchestration)

The local dry-run additionally creates `inference_only/` and
`decoded_image_to_detections/` subdirectories, each containing a mock session
and adapter event log.

The CPU/mock run additionally writes one session and adapter event file beneath
each declared timing boundary. It retains structured failure data and output
hashes. A future real run must retain raw runtime/source error text, per-source
timestamps and clock identity/conversion evidence. Missing or unverified final
synchronization, native output contract, checkpoint/engine hash, runtime
compatibility, or binding shape/dtype is a correctness stop condition. Missing
power boundary or clock alignment makes energy unavailable, but does not stop
correctness or correctly synchronized latency. Remote collector lifecycle and
per-source clock/error persistence remain an explicit unimplemented
prerequisite for real telemetry sessions; no inference data is fabricated here.

E2L1-006 ends at local code and mock validation. A later device-side action
requires Astra review of this package and explicit authorization; E3/E5 remain
compatibility candidates rather than automatic scored arms.
