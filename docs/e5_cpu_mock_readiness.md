# E5 TensorRT CPU/mock compatibility packet

Status: `e5_cpu_mock_readiness_review_required`. This packet is contract and
mock evidence only. No E5 SSH, library load, runtime import, install, source or
engine transfer, export, build, model load, inference, warmup, or benchmark was
performed. It does not replace E2 or validate the Orin Nano Super as a model
target.

## Observed device facts

The sole accepted basis is the sanitized 2026-10-03 E5 inventory at
[`inventory_sanitized_v2.json`](../results/edge_readiness_v1/edge_contingency_20261004/E5/inventory_sanitized_v2.json),
SHA-256 `e56a570f38e97ee8ce313b2dad5155da9d3a441943b88ed7057df550002a45c1`:

- NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super; L4T
  R36.5.2 / Ubuntu 22.04.5, Python 3.10.12, six Cortex-A78AE CPUs.
- TensorRT package metadata 10.3.0.30 built for CUDA 12.5; `trtexec` reports
  v100300. `nvcc` reports CUDA 12.6.68 and the NVIDIA driver reports 540.5.0.
  These are inventory observations, not proof of the Python module's loaded
  libraries or the runtime ABI used by an engine.
- NumPy 1.21.5; OpenCV import was missing. About 5.3 GiB available RAM and
  792 GiB root space were observed at inventory time. MAXN_SUPER and thermal /
  process readings were observed, not changed or controlled.
- The inventory did not establish any concrete `libcudart` path, successful
  CUDA library load, TensorRT Python binding path, idle state, or calibrated
  energy boundary.

## CPU/mock result and compatibility matrix

Implementation is isolated in `scripts/edge_readiness/e5_tensorrt_compat.py`;
tests are isolated in `tests/test_e5_tensorrt_compat.py`. Neither imports the
E2 adapter, TensorRT/CUDA Python bindings, or OpenCV, nor calls `ctypes.CDLL`
at module import time. All runtime, context, filesystem, library-loader and
memory-owner interfaces are injected.

| Contract | Local evidence | Remaining device uncertainty |
| --- | --- | --- |
| TensorRT 10.3 named-I/O profile; exact `images` INPUT and `output0` OUTPUT; FP32/device location; `[1,3,640,640]` and `[1,7,8400]`; reject extra/dynamic/wrong tensors | Mock introspection and fail-closed mismatch tests pass | Actual engine metadata and TensorRT 10.3 Python ABI not queried |
| Address-before-dispatch ordering; same-stream H2D, `execute_async_v3`, D2H, completion synchronization; independent fresh host output per call | Delayed-copy doubles prove output is read only after sync and differs across calls | CUDA stream semantics and actual TensorRT enqueue not exercised |
| Allocation lifetime, partial allocation, enqueue/sync/free failures, drain-before-free | CPU doubles cover cleanup; if drain fails, pointers are retained and output is withheld | Real CUDA error behavior and process-level recovery remain unverified; this synchronous contract has no wall-clock watchdog |
| Source ONNX, per-image input tensor, engine, and engine-source ONNX SHA-256 binding | Tests hash actual byte payloads and reject changed/mismatched data before allocation | No E5 engine or source/input transfer was audited |
| CUDA 12.x library candidate ordering and required symbol presence | Injected filesystem/library doubles select only CUDA 12 candidates; no library is loaded by these tests | Actual candidate paths and ABI remain unverified; do not infer support from `nvcc` or package metadata |
| OpenCV-required preprocessing | Missing/incomplete OpenCV fails closed; there is no Pillow or other fallback | E5 inventory reports OpenCV missing; a future scoped task must resolve that prerequisite |

The E5-specific suite reports **15/15 PASS**. This is CPU/mock interface
coverage only and is not E5-validated, a CUDA-runtime test, or model correctness
evidence.

## Real-device prerequisites (not authorized by this packet)

Before any future E5 operation, obtain a separate written scope decision and
then satisfy all of the following without borrowing the E2 engine or silently
changing E2's call/build budget:

1. Reconfirm exact device identity, free memory/storage, thermal state,
   competing workloads and power mode through one separately bounded,
   approved read-only preflight. Do not change power/clock/fan settings.
2. Resolve the missing OpenCV dependency through a separately authorized
   environment decision. Record its version and prove the exact preprocessing
   transform against the frozen source path; fail closed on any mismatch.
3. Read the actual TensorRT Python/runtime versions and resolved library paths;
   establish compatibility across TensorRT 10.3.0.30, CUDA runtime 12.x,
   CUDA driver, and the device-built engine. Record real `libcudart` path,
   ABI/symbol evidence and runtime/driver versions. Inventory metadata alone is
   insufficient.
4. Audit the private canonical source ONNX, per-fixture input bytes and hashes,
   and an E5-native engine whose manifest binds its source ONNX hash. Never
   copy/reuse the E2 engine. Keep all source/model/image/tensor/engine bytes
   private.
5. Verify named I/O, device locations, concrete shapes/dtypes, context address
   acceptance, same-stream copy/dispatch ordering, synchronized output bytes,
   error cleanup and output hash before interpreting any numerical comparison.
6. Agree the fixed fixture order, stop conditions, logging and resource
   envelope in advance. Preserve raw evidence and prohibit retries, warmup,
   benchmark claims, or unapproved extra calls.

## One bounded prospective E5 proposal — not authorization

If a separate scope decision is later granted, propose a target-runtime smoke
only: build one E5-native TensorRT 10.3 engine from the approved frozen source
ONNX (proposed FP16, 1-GiB workspace, 3,600-second hard build ceiling as an
operational stop, not a runtime estimate), then run exactly one ordered pass
over fixtures `00006`, `00009`, `00028`, zero warmup, and a proposed hard
120-second per-inference watchdog. Each inference must run in a killable child
process; the synchronous CPU/mock function alone cannot enforce a timeout. No
retry is allowed. Stop on any dependency, identity, resource, hash,
engine-contract, timeout, cleanup, or correctness failure. Compare raw
source-ONNX and E5 outputs, then report the same-origin post-NMS diagnostic;
do not calculate AP, latency/FPS, energy, or claim deployment readiness.
Preserve binaries and raw tensors privately and publish only sanitized hashes,
counters and diagnostics. This bounded smoke is not the 1,636-image E2
full-dev study and cannot substitute for it.

Until that separate decision and prerequisites pass: E5 build/inference budget
is **zero**; current status remains `e5_cpu_mock_readiness_review_required`.
