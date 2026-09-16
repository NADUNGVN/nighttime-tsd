# Edge readiness v1 — E2L1-002 protocol and setup plan

Status: local CPU/mock harness implemented; device inference, export, compiler
execution, calibration, installation and benchmark remain **not authorized**.

The accepted E2L1-001 inventory is an initial live snapshot, not deployment or
performance validation. This document defines the next interface and setup
options without selecting a final model, arm or scored device scope.

## Local harness contract

`scripts/edge_readiness/measurement_harness.py` is backend-neutral and imports no
TensorRT, HailoRT, CUDA, image decoder or model runtime. Only `--mock` is exposed
by its CLI. The backend adapter owns synchronization and exposes:

- `synchronize("before_timer")` and `synchronize("after_inference")`;
- `infer(preprocessed)`;
- a declared synchronization policy and backend name.

The caller binds a stable target ID, hardware identity, runtime versions, model
ID and direct source-weight SHA-256. A mismatch is a structured error; missing,
permission-denied and not-measured evidence is represented as a status object,
never as a fabricated numeric value.

Two timing boundaries are explicit:

| Boundary | Included | Excluded |
|---|---|---|
| `inference_only` | decoded image preprocessing is outside timer; synchronized backend inference | postprocessing, disk decode, model load/allocation |
| `decoded_image_to_detections` | preprocessing, synchronized inference and postprocessing from an already-decoded image | disk decode, model load/allocation |

Both boundaries require batch 1 and observed preprocessed shape `(1, 3, 640,
640)`. The protocol starting design is 200 warmup calls, 1,000 measured calls,
three sequential sessions. Warmup is excluded from raw samples. Every measured
sample is retained per session; summary fields are mean, median, linear p50/p95/p99,
min, max and serial FPS (`1000 / mean_ms`). Pipelined throughput is a separate
metric and remains unavailable until a producer/consumer measurement exists.

## Power, memory and thermal contract

Power samples must contain strictly increasing finite monotonic nanoseconds,
nonnegative finite watts, unit `W`, one declared boundary such as `whole_device`
or `module_input`, and verified clock identity/alignment source. Duplicate or
out-of-order timestamps, mixed boundaries/units/clocks, unverified alignment and
invalid values fail closed; sorting is not implicit. Energy uses trapezoidal
integration over the covered interval and linearly interpolates power at clipped
interval endpoints without extrapolation. A result is `measured` only when
samples bracket both requested endpoints; overlapping but incomplete coverage is
`partial` with coverage metadata, and no-overlap is `unavailable`. TDP/TOPS is
never used as measured power. Memory is a backend-specific, unit-labelled
observation (for example `MiB`, scope `runtime_allocator`), not silently promoted
to whole-device peak.

The integration record includes requested/covered start and end, duration,
coverage fraction, sample gaps/max gap, clock identity, alignment source and
method. No maximum-gap acceptance threshold is invented in advance.

Persist absolute monotonic session start/end and duration. Energy covers that
outer session interval, including Python/provider overhead, while latency samples
retain their narrower declared timing boundary. Persist before/after observations
for power mode, thermal sources, throttle state, runtime versions and hardware
identity. These snapshots do not establish a controlled idle or steady-state
condition. External meter model, sampling rate, measurement boundary, clock
alignment, cooling and power supply remain operator-supplied.

## Setup action matrix — proposed only

| Device | Observed prerequisite | Missing/unresolved | Proposed next action | Disk/permission risk |
|---|---|---|---|---|
| E1 Pi5 + Hailo-8 | HailoRT 4.23.0; Hailo-8 identify succeeds; ~4.9G free | No external power meter; cooling/supply boundary; PCIe x1 status not rechecked | Keep CPU FP32 reference and Hailo-native path as separate conditions; use an approved x86 compiler host only after export/operator review | Keep local headroom; no install on board; Hailo compile not authorized |
| E2 Xavier NX | L4T R35.4.1; CUDA 11.4.19; cuDNN 8.6.0.166; TensorRT 8.5.2.2; `tegrastats`; ~94G free | `nvidia-smi` absent; `jetson_clocks --show` needs root; power meter/cooling unknown | Prioritize GPU-native TensorRT FP16 review, then backend-native INT8 review; do not assume DLA full-model support | No apt/pip/sudo; root-only probes remain unresolved |
| E3 AGX Xavier | L4T R35.6.4; CUDA 11.4.19; cuDNN 8.6.0.166; TensorRT 8.5.2.2; `tegrastats`; ~11G free | Exact RAM SKU; `nvidia-smi` absent; `jetson_clocks --show` needs root; power meter/cooling unknown | Prioritize GPU-native TensorRT FP16 review, then INT8; confirm memory and thermal setup before any artifact transfer | eMMC headroom is tight; no install or engine build authorized |
| E5 Orin Nano Super | L4T R36.5.2; CUDA 12.6.68; TensorRT 10.3.0; driver 540.5.0; ~792G free | External power meter/cooling boundary; not legacy Nano; no DLA scope | Consider GPU-native TensorRT review only after model/backend confirmation; do not substitute it for legacy Nano | No firmware/power-mode changes; current MAXN_SUPER is observation only |

The matrix deliberately prioritizes E2/E3 GPU-native planning and E1 CPU/Hailo
feasibility. It does not add E4, Qualcomm, DLA, or an all-15-model matrix.

## Proposed run options after review

1. CPU FP32 reference: local or approved edge CPU condition, same frozen source
   weights, no relabelling as INT8.
2. NVIDIA GPU: device-native TensorRT FP16 first; INT8 only with an approved
   calibration manifest and backend-native conversion.
3. Hailo: compile on an approved x86 host and run on E1 only after graph/export
   review; an existing server engine is not portable to Hailo or Jetson.

No option is a deployment result until source-weight hash, calibration IDs/order,
runtime versions, preprocessing/postprocessing, input shape, synchronization,
session counts, telemetry boundary and thermal/throttle evidence are bound.

## Operator questions before expansion

- What are E3's exact RAM SKU, cooling solution and power supply?
- What external meter and measurement boundary will be used for each board?
- Should E1 CPU and Hailo be reported as two conditions on one physical board?
- Which representative model/arms and which E2/E3/E5 scored scope should review
  authorize? E5 must not silently stand in for the missing legacy Nano.
- Is the existing x86 Hailo compiler host the approved offline compiler host for
  this paper, with output storage and transfer path specified?

The mock artifact under `results/edge_readiness_v1/e2l1-002/mock/` is a contract
exercise only and must not be cited as edge latency, accuracy, energy or runtime
evidence. Existing mock/inventory artifacts are not rewritten by E2L1-003.
