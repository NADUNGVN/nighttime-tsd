# Alternative-device readiness — read-only snapshot

Collected once per currently reachable approved alias on 2026-10-03 18:19:39 UTC. These are current observations, not controlled idle measurements. No model was loaded, compiled, inferred, benchmarked or downloaded; no package, runtime, power mode, clock or fan was changed. Sanitized records and hashes: [`edge_contingency_20261004`](../../results/edge_readiness_v1/edge_contingency_20261004/README.md).

| Device | Observed software/hardware | Resource and telemetry snapshot | Assessment for this paper |
| --- | --- | --- | --- |
| E1 | Raspberry Pi 5 Model B Rev 1.1, Debian 13/aarch64, Hailo-8; Python 3.13.5, NumPy 2.2.4, OpenCV 4.10.0, HailoRT CLI 4.23.0 / `hailo-all` 5.1.1. | 7.3 GiB available RAM; 2.1 GiB free on 29-GiB root (93% used). `vcgencmd` present; no tegrastats, sensors or powercap/IIO source reported. | The prior Hailo toolchain/readiness is relevant only as a distinct backend. Tight local storage needs an operator-owned capacity decision. No Hailo compile/run is authorized here and the E2 TensorRT engine is not portable. |
| E4 | Thundercomm RUBIK Pi 3, Ubuntu 24.04.4/aarch64, QAIRT libraries/DSP binaries/tools 2.46.0; Python 3.12.3, NumPy 1.26.4, OpenCV 4.6.0. | 6.3 GiB available RAM; 26 GiB free on 112-GiB root (77% used). `sensors` and powercap/IIO paths are present; no validated meter boundary or sampled energy. | Separate Qualcomm/QAIRT architecture; not a TensorRT substitute. Installed package versions do not establish CLI path, model/operator coverage or CCTSDB compatibility. Keep to readiness only. |
| E5 | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super, Ubuntu 22.04.5/L4T R36.5.2, CUDA 12.6.68, driver 540.5.0, TensorRT 10.3.0.30 (`trtexec` reports v100300); Python 3.10.12, TensorRT Python 10.3.0, NumPy 1.21.5; OpenCV import is missing. | 5.3 GiB available RAM; 792 GiB free on 915-GiB root. Current mode reports `MAXN_SUPER` (observed, unchanged); tegrastats and IIO path exist. CPU/GPU thermal reads were about 51.2°C; other zones included unavailable values. | Closest alternative TensorRT route, but not ready as a drop-in CCTSDB target. The code has an E5 TensorRT 10.3/named-I/O/v3 profile, while the concrete CUDA owner only searches CUDA 11.4 libraries. OpenCV is also missing. Existing adapter tests do not demonstrate a real E5 engine/runtime. Requires CPU/mock completion and a separately reviewed scope decision before any E5 model operation. |

`ps` snapshots are single observations and include the collection/session workload; they are not evidence of controlled idle, contention over a study, or process safety. Device temperature, telemetry-source presence and selected power mode are likewise not benchmark conditions.

## E5 contract assessment and required local work before a separate study

The existing low-level adapter explicitly declares E5 TensorRT `10.3.0` and `named_tensor_execute_async_v3`. `TensorRTProvider` contains a `num_io_tensors` inspection path, and `JetsonRuntimeAdapter` binds tensor addresses before `execute_async_v3`; the model output validator expects `output0 [1,7,8400]`. This is useful interface coverage, not hardware validation. The full-dev runner remains E2-specific, and the adapter's current engineering-reference checkpoint is not automatically the frozen CCTSDB source artifact.

Before proposing a bounded E5 experiment, local CPU/mock work should cover:

- E5 version/profile validation against the observed TensorRT 10.3 API, including named I/O mode, shape, dtype and device location; reject fallback or missing metadata.
- TensorRT 10.3 engine/context doubles through tensor-address setup, `execute_async_v3`, synchronized H2D/D2H, fresh output bytes, exception/timeout cleanup and the `[1,7,8400]` output contract.
- A CUDA 12.6-compatible library-discovery profile. The current `cuda_runtime_owner.py` candidates are CUDA 11.4-specific; the device's actual `libcudart` path and ABI were not queried or loaded.
- Fail-closed preprocessing tests for missing OpenCV; exact source/target image-byte and transform bindings; canonical postprocess/AP path; and target-specific engine/checkpoint hashes.
- Existing full-chain CPU tests in the pinned dependency environment. The current workstation skips the pinned Ultralytics 8.4.102 NMS/ORT tests; do not turn a skip into a pass by installing dependencies during this readiness task.

## Capacity, timing and scope proposal (not authorization)

Current E5 inventory leaves 792 GiB disk and 5.3 GiB RAM available, and the existing E2 packet's planning contract uses 1 GiB private output and a 4-GiB available-memory floor. These are point-in-time readings and a draft budget, not proof that a future E5 TensorRT build will fit. Do not reuse the E2 engine; any future E5 engine must be built on E5 from the approved frozen source artifact and separately hash-bound.

If Astra/user later chooses a separate E5 correctness study, a concrete proposal is: one E5-specific TensorRT 10.3/CUDA 12.x adapter repair and CPU/mock review; then, only after a single explicit scope decision and all prerequisites, at most one target engine build and one pass over the 1,636 canonical dev images (one enqueue/image), zero warmup, extra smoke, retry/resume or benchmark. A 3,600-s build ceiling may be proposed as an operational stop, not an expected duration; the E2 build window is not an E5 estimate. Reuse source predictions only if their ONNX/image/input bindings verify exactly. Report paired AP separately and preserve the source export discrepancy. This would be a distinct study, not a silent replacement for the E2 lane.

E1/Hailo and E4/QAIRT each require their own toolchain and backend-native conversion/runtime correctness review. Neither is included in the current call budget.
