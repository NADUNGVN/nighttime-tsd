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
behind small interfaces and tested with CPU mocks.

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
  enqueue, after enqueue, and after D2H output copy. A future device adapter
  must wait for the final stream/event completion before returning outputs;
  a host enqueue return or host-side timer alone is not completed-device
  latency.
- E2/E3 use the legacy binding-list candidate; E5 assigns every named tensor
  address before the v3 enqueue candidate. Any missing API, false enqueue,
  hash mismatch, runtime mismatch, or binding mismatch is a structured stop.
- Native output names/shapes/order are captured before postprocessing. Exactly
  one layer owns NMS; no double NMS. The smoke checks raw output contract and
  numerical tolerances, not AP, FPS, energy, or official-test performance.

## Proposed correctness smoke

The same deterministic train-only engineering fixture is supplied in the same
order and as the same bytes to the source-reference and target adapter. The
current CPU fixture contains three 4x4 PPM byte payloads with IDs
`train-fixture-00` through `train-fixture-02`; their individual SHA-256 values
and an order-bound sequence hash are emitted by `fixture_manifest()`.

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

After explicit authorization, the proposed command shape is:

    python scripts/edge_readiness/jetson_adapter_smoke.py --target E2 --engine <target-local-engine> --fixture train-only-engineering

This command is intentionally not present as an executable device workflow in
E2L1-006; it is a review placeholder. The allowed resource set is the existing
JetPack/TensorRT runtime on the named target, the accepted checkpoint bytes,
the deterministic train-only fixture, and a new target-scoped output path.
No server engine transfer is allowed.

Forbidden mutations and work in this phase are package installation, sudo,
power/clock/fan changes, reboot, export/build, model load/forward, official
test use, and AP/FPS/energy scoring. The adapter package also does not repeat
the accepted device inventory.

## Future artifact layout and stop conditions

An authorized run should write only under:

    results/edge_readiness_v1/e2l1-006/E2/<run_id>/
      manifest.json
      input_fixture_manifest.json
      adapter_events.json
      output_contract.json

It must retain raw runtime/source error text, per-source timestamps and clock
identity/conversion evidence, and output hashes. Missing or unverified clock
alignment, power boundary, final synchronization, native output contract,
checkpoint/engine hash, runtime compatibility, or binding shape/dtype is a
stop condition. Telemetry feasibility remains separate from scored energy.

E2L1-006 ends at local code and mock validation. A later device-side action
requires Astra review of this package and explicit authorization; E3/E5 remain
compatibility candidates rather than automatic scored arms.
