# Luna 1 to Astra — Edge deployment lane

## L1A-001 — E2L1-001 GO SSH read-only inventory and local harness preparation

**Status:** GO inventory/preparation complete; **NO-GO** for installation, export,
compilation, inference, calibration, or scored benchmarking.

**Collection:** 2026-09-16 09:24:47 UTC (16:24:47 Asia/Saigon), schema
e2l1-edge-inventory-v1, run ID 20260916T092447Z. The collector source hash
recorded in every inventory is
b9abe9272cae4c015d6094248e559f854f417840b9ac1abe877745b6ed87b212.

### Scope and transport evidence

Only the four mapped targets E1/E2/E3/E5 were contacted. E4 was excluded as
required; no network scan was performed for the unmapped Legacy Nano or a
separate Pi5 CPU board. SSH used the existing local aliases and key configuration:

    ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes <alias> sh -s

| Device | SSH alias | Remote identity | Transport | Queries |
|---|---|---|---:|---:|
| E1 | pi5 | raspberrypi / Raspberry Pi 5 Model B Rev 1.1 | exit 0 | 24/24 recorded |
| E2 | nx | arar-desktop / NVIDIA Jetson Xavier NX Developer Kit | exit 0 | 24/24 recorded |
| E3 | agx | arar / Jetson-AGX | exit 0 | 24/24 recorded |
| E5 | nano | ubuntu / NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super | exit 0 | 24/24 recorded |

The E1 source document's historical Tailscale address differs from the current
local pi5 alias. The alias was used as the authorized access binding; no address
is copied into this paper artifact.

### Live inventory snapshot

| Device | CPU / RAM | OS / kernel | Storage snapshot | Accelerator and observed runtime |
|---|---|---|---|---|
| E1 | 4× Cortex-A76, aarch64; 7.9 GiB | Debian 13.5; 6.18.34+rpt-rpi-2712 | MMC root 29G, 23G used, 4.9G free (83%) | Hailo-8 PCIe; HailoRT CLI/Python 4.23.0; firmware identify succeeds |
| E2 | 6 CPU(s), aarch64; 6.7 GiB | Ubuntu 20.04.6; 5.10.120-tegra; L4T R35.4.1 | eMMC root 117G, 18G used, 94G free (17%) | TensorRT 8.5.2.2, CUDA 11.4.19, cuDNN 8.6.0.166; trtexec --help header v8502 |
| E3 | 8 CPU(s), aarch64; 14 GiB | Ubuntu 20.04.6; 5.10.216-tegra; L4T R35.6.4 | eMMC root 28G, 16G used, 11G free (59%) | TensorRT 8.5.2.2, CUDA 11.4.19, cuDNN 8.6.0.166; trtexec --help header v8502 |
| E5 | 6× Cortex-A78AE, aarch64; 7.4 GiB | Ubuntu 22.04.5; 5.15.199-tegra; L4T R36.5.2 | NVMe root 915G, 77G used, 792G free (9%) | Orin GPU; driver 540.5.0; CUDA 12.6.68; TensorRT 10.3.0; trtexec --help header v100300 |

E3's exact RAM SKU remains unverified. E5's live L4T package is R36.5.2, correcting
the older R36.5.0 observation in the shared reference. NVIDIA maps R35.6.4 to
JetPack 5.1.6 and R36.5.2 to the JetPack 6.2.3 generation; these are documentation
cross-checks, not a claim that a portable engine exists across devices:
[NVIDIA Jetson Linux R35.6.4](https://developer.nvidia.com/embedded/jetson-linux-r3564),
[NVIDIA JetPack archive](https://developer.nvidia.com/embedded/jetson-linux-archive),
[NVIDIA JetPack SDK 6.2.3](https://developer.nvidia.com/embedded/jetpack-sdk-623).

### Command evidence and incomplete checks

The JSON files retain raw output, command labels, and return codes:

- [E1 inventory](../results/edge_readiness_v1/E1/20260916T092447Z/inventory.json)
- [E2 inventory](../results/edge_readiness_v1/E2/20260916T092447Z/inventory.json)
- [E3 inventory](../results/edge_readiness_v1/E3/20260916T092447Z/inventory.json)
- [E5 inventory](../results/edge_readiness_v1/E5/20260916T092447Z/inventory.json)

Key outcomes: E1 hailo_identify exit 0; E2/E3 nvcc, trtexec, L4T package query
exit 0; E5 nvidia-smi, nvcc, trtexec, L4T package query exit 0. Missing-tool
states were recorded as exit 127, not repaired. jetson_clocks --show returned
exit 1 on E2/E3/E5 with an explicit non-root error; no sudo was used.
nvidia-smi is absent on E2/E3, so their driver identity remains unavailable from
this non-root inventory.

Power/thermal evidence is readiness-only: E1 reported 49.1°C CPU thermal; E2
CPU/GPU were about 28.5–29.0°C with PMIC 50°C; E3 CPU/GPU were about 28.5–30.0°C
with PMIC 50°C; E5 CPU/GPU were about 48.9–49.0°C. E2 power mode was
MODE_20W_6CORE, E3 MAXN, and E5 MAXN_SUPER; these are snapshots, not benchmark
conditions. tegrastats was present on E2/E3/E5 and vcgencmd on E1, but no
telemetry sample was taken. No external power meter, cooling description,
power-supply description, or whole-device measurement boundary is observable
through SSH; these remain operator-supplied prerequisites. No power/clock/fan
state was changed.

### Compatibility matrix

The machine-readable matrix is [compatibility_matrix.json](../results/edge_readiness_v1/20260916T092447Z/compatibility_matrix.json).
Observed runtime availability is deliberately separated from model/export support:

| Target | Observed capability | CCTSDB YOLO export / YOLO26 native end-to-end | Current decision |
|---|---|---|---|
| E1 | Hailo-8 + HailoRT 4.23.0 | Unverified; no ONNX/HEF/compiler path run | Candidate after Hailo-specific review |
| E2 | Xavier GPU stack: TensorRT 8.5.2.2 / CUDA 11.4 | Unverified; DLA full-model suitability not assumed | Candidate for later GPU-only review |
| E3 | AGX Xavier GPU stack: TensorRT 8.5.2.2 / CUDA 11.4 | Unverified; exact RAM SKU and DLA scope remain risks | Candidate for later GPU-only review |
| E5 | Orin GPU stack: TensorRT 10.3 / CUDA 12.6 | Unverified; not legacy Jetson Nano; no DLA path assumed | Candidate after model/backend review |

NVIDIA's support documentation confirms that hardware precision/runtime support
and serialized-engine portability are separate questions. TensorRT also documents
Xavier precision/DLA capabilities by architecture; the exact installed 8.5.2
parser/operator behavior for this YOLO graph still requires a later authorized
conversion check: [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/support-matrix/),
[current TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix/).

Risks to review before any conversion include native YOLO26 end-to-end output,
NMS/output tensor contracts, dynamic shapes, operator coverage, precision
constraints, and backend-native quantization. No compatibility conclusion is
inferred from a board name alone.

### Proposed measurement protocol (not authorized)

The schema-versioned proposal is [measurement_protocol.json](../results/edge_readiness_v1/20260916T092447Z/measurement_protocol.json).
Before authorization, bind the frozen source-weight SHA-256, calibration image
IDs/order/manifest hash where logically applicable, preprocessing/postprocessing
code/config hash, backend-native quantization mode, runtime versions, input shape,
batch, synchronization, latency boundary, warmup/measured counts, p50/p95/p99,
peak memory, thermal steady state, throttle state, and the external power-meter
boundary. Counts remain unset pending review. CPU FP32 is a legitimate reference
condition and must not be relabeled all-INT8.

### Local preparation and action list

Added:

- scripts/edge_readiness/collect_inventory.py: fixed, bounded SSH collector;
  new-output-only JSON writer; no installation/build/inference/benchmark flags.
- tests/test_edge_readiness.py: five CPU fixture tests covering target
  validation, marker parsing, missing/permission/timeout states, and a
  read-only command guard.

Local test command and result:

    python -m unittest tests/test_edge_readiness.py -v
    Ran 5 tests ... OK

Next action requires Astra/reviewer approval: choose representative model/arms
and backend conditions, then separately authorize any dependency installation,
export/conversion/compiler smoke test, inference, or benchmark. Do not copy a
server RTX engine to an edge device.

**Git:** branch `luna1/e2l1-001-edge-readiness`; final commit is `HEAD` at
handoff. Confirm with `git rev-parse HEAD` (the hash is reported in the handoff
message; embedding HEAD's hash in this file would be self-referential).
