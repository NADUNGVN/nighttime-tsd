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

## L1A-002 — local measurement harness and setup planning

**Status:** GO local CPU/mock harness and protocol; **NO-GO** device inference,
export, compilation, calibration, installation, or scored latency/power study.
No device was re-SSH'd for this task; the accepted E2L1-001 snapshots remain
canonical.

### Implementation

- `scripts/edge_readiness/measurement_harness.py` defines the backend-neutral
  adapter contract. It requires explicit synchronization hooks, model SHA-256,
  stable target/device binding, batch 1 and `(1,3,640,640)` shape.
- Timing boundaries are explicit: `inference_only` excludes postprocess, while
  `decoded_image_to_detections` includes preprocess/inference/postprocess from an
  already-decoded image and excludes disk decode/model load/allocation.
- The starting protocol is 200 warmup calls, 1,000 measured calls and 3 serial
  sessions. Raw samples are retained per session; p50/p95/p99 use linear
  interpolation; serial FPS is kept separate from unavailable pipelined
  throughput.
- `integrate_power_energy` accepts only time-aligned nonnegative watts with one
  declared boundary and uses trapezoidal joule integration. Missing telemetry,
  memory, thermal, throttle or permission evidence remains a structured
  `unavailable`/`permission_denied` state.
- `scripts/edge_readiness/collect_inventory.py` was hardened for future use
  without rerunning devices: SSH aliases reject option-like values; complete
  collection rejects duplicate, unexpected and missing command markers; timeout
  partial records are preserved; transport status is separate from inventory
  completeness; `trtexec` future probing uses direct `--help` without `head`.

### Setup and protocol artifacts

- [EDGE_READINESS_V1.md](EDGE_READINESS_V1.md): concrete boundaries, metric
  semantics, proposed E1/E2/E3/E5 action matrix, options and operator questions.
- [EDGE_READINESS_V1.json](EDGE_READINESS_V1.json): machine-readable proposed
  setup matrix. It excludes E4, the unmapped legacy Nano, a separate Pi5 board,
  DLA and the all-15-model matrix.
- [mock manifest](../results/edge_readiness_v1/e2l1-002/mock/manifest.json) and
  three session JSONs: local contract exercise only, not edge evidence.

### Tests

```text
python -m unittest tests/test_edge_readiness.py tests/test_edge_measurement_harness.py -v
Ran 15 tests ... OK
python -m py_compile scripts/edge_readiness/collect_inventory.py scripts/edge_readiness/measurement_harness.py
```

The full repository suite was not used as the gate for this isolated lane; its
existing tests require unrelated optional/runtime artifacts. The 15 relevant
edge tests and both module syntax checks pass. No server/device command was run.

### Review options and outstanding operator questions

Proposed order is CPU FP32 reference, E2/E3 GPU-native TensorRT FP16 then
backend-native INT8 after review, and E1 Hailo-native conversion on an approved
x86 compiler host. These are planning options only; no final model/arm/backend
selection is made. E5 remains Orin Nano Super and cannot substitute silently for
legacy Nano.

Before expansion, Astra/operator must provide or approve E3 exact RAM SKU,
per-board cooling/supply and external-meter boundary, whether E1 CPU/Hailo are
two conditions on one physical board, representative model/arm/device scope,
and the approved offline Hailo compiler host/output transfer path.

**Git:** worktree `D:/Research/paper-luna-edge-002`, branch
`luna1/e2l1-002-edge-harness`; final commit is `HEAD` at handoff. Do not open or
merge a PR automatically.

## L1A-003 — corrected energy integration and session binding

**Status:** GO bounded local fixes accepted for implementation; **NO-GO** for real
telemetry, device inference, conversion, installation or benchmark. No SSH or
device command was run, and the accepted E2L1-001 inventories and E2L1-002 mock
artifacts were not rewritten.

### Corrections

- `integrate_power_energy` now rejects non-finite/negative power, invalid
  timestamps, duplicates and out-of-order samples, mixed units/boundaries, and
  missing or unverified clock identity/alignment. It uses an explicit validated
  input-order policy; no implicit sorting occurs.
- A full `measured` result requires samples bracketing both requested endpoints.
  An overlapping interval with missing left/right coverage is `partial` and
  includes requested/covered duration and fraction; no overlap is
  `unavailable`. Partial results cannot claim full-session average power or
  energy-per-image.
- Clipped integrations interpolate power at the clipped endpoints. The ramp
  case `(0,0W),(10s,10W)` over `[0,2s]` now returns 2J, not 10J. The
  `[0,10s]` request with samples only at `[2s,8s]` returns `partial` with 6s
  coverage and 60J covered-overlap energy, not `measured` full-session energy.
- Integration persists coverage, duration, gaps/max gap, clock identity,
  alignment source and method. It does not invent a max-gap threshold.
- `run_session` persists absolute monotonic session start/end/duration and
  integrates energy over the outer session interval, including provider/Python
  overhead; latency timing retains its narrower explicit boundary. Pool IDs,
  pool hash and order are explicit bindings rather than a hidden `%256`
  assumption. Pipelined throughput cannot be claimed by config.
- `write_json_no_overwrite` now uses exclusive `open("x")` semantics to close
  the exists-then-write race. Boundary values must be the explicit `Boundary`
  enum, not arbitrary strings.

### Tests and protocol update

Added coverage for full constant/ramp integration, clipped ramp interpolation,
partial left/right/interior coverage, no overlap, duplicate/out-of-order samples,
mixed boundary/clock, non-finite and negative values, unverified alignment,
explicit pool binding, outer session timing and exclusive output creation.

```text
python -m unittest tests/test_edge_readiness.py tests/test_edge_measurement_harness.py -v
Ran 18 tests ... OK
python -m py_compile scripts/edge_readiness/collect_inventory.py scripts/edge_readiness/measurement_harness.py
```

The corrected public protocol is in [EDGE_READINESS_V1.md](EDGE_READINESS_V1.md);
the setup matrix remains proposed-only. No real telemetry sample was taken and
no maximum-gap threshold or power boundary was selected for future runs.

**Git:** branch `luna1/e2l1-003-energy-fix` in worktree
`D:/Research/paper-luna-edge-002`; commit to be recorded after scoped commit.

## L1A-004 — bind pool consumption, measured energy window and session clock

**Status:** GO bounded local CPU/mock contract completion; **NO-GO** for device
inference, telemetry collection, conversion, installation or benchmark. No SSH or
device command was run. Earlier inventory and mock artifacts were not rewritten.

### E3 — declared pool is used by every provider call

- `run_session` now applies the declared `pool_order` in both warmup and measured
  loops. The explicit policy is `restart_each_phase`, so each phase begins at the
  first declared pool index and wraps over that permutation; the old hidden
  `index % 256` path is gone.
- Pool IDs must be unique, and `pool_order` must contain integer, non-boolean
  indices forming a complete permutation. Each phase persists actual consumed
  indices, IDs, count and a deterministic sequence hash. The provider receives
  only declared pool indices, so non-256, reversed-order and wraparound cases are
  testable and out-of-range providers fail immediately.
- The persisted ID hash remains mock provenance only; it is not a substitute for
  future image-content provenance.

### E4 — energy is measured-loop scoped

- The result now persists `session_window` (warmup + measured) and a distinct
  post-warmup `measured_window` whose outer boundaries include provider/Python
  overhead for measured calls.
- Primary `power_energy` integrates only the measured window and records
  `includes_warmup=false`, measured image count and scope. Optional
  `include_warmup_energy=true` produces separately named
  `warmup_inclusive_energy_interval` and `warmup_inclusive_power_energy` fields;
  they cannot be silently mixed with the primary result.
- A deterministic injected-clock test uses a large warmup interval and verifies
  50J measured-window energy versus 150J warmup-inclusive energy for the same
  constant 10W fixture.

### E5 — explicit session clock/alignment and synchronous postprocess

- `integrate_power_energy` requires an explicit expected clock identity and
  alignment source. Missing/unverified binding, contradictory metadata and mixed
  source clocks fail closed; equal row strings are not treated as proof of
  alignment to the session boundaries.
- A differing source clock requires a validated conversion mapping with matching
  source/target identities, finite positive scale, integer offset, `validated=true`
  and non-empty evidence. The session record binds host `time.perf_counter_ns`,
  its alignment source and the limitation that this is harness/session binding,
  not independent device-clock verification.
- The CPU/mock decoded-to-detections path rejects awaitable postprocessing with
  `ASYNC_POSTPROCESS_UNSUPPORTED`; no future adapter is assumed synchronous.

### Verification

```text
python -m unittest tests/test_edge_readiness.py tests/test_edge_measurement_harness.py -v
Ran 23 tests ... OK
python -m unittest discover -s tests -p 'test_edge*.py' -v
Ran 23 tests ... OK
python -m py_compile scripts/edge_readiness/collect_inventory.py scripts/edge_readiness/measurement_harness.py
```

No real runtime/telemetry/energy measurement was performed. The local tests use
only the mock backend and deterministic fixtures; no benchmark claim is made.

**Git:** branch `luna1/e2l1-004-session-pool-clock` in worktree
`D:/Research/paper-luna-edge-002`; commit and push recorded after the scoped
handoff commit. Do not open or merge a PR automatically.

## L1A-005 — bounded raw telemetry feasibility sample

**Status:** GO raw read-only telemetry feasibility completed for the four
authorized targets E1/E2/E3/E5; **NO-GO** for inference, model/runtime build,
benchmark, measured inference energy/FPS/AP, installation or device-state change.
The active GitHub account was `NADUNGVN`. No private endpoint or credential is
included in these artifacts.

### Capture scope and method

The new
[`collect_telemetry.py`](../scripts/edge_readiness/collect_telemetry.py) sends a
fixed foreground shell script through SSH with `BatchMode=yes`,
`ConnectTimeout=5` and `StrictHostKeyChecking=yes`. It performs five snapshots
per canonical run, records device `time.monotonic_ns()` before each snapshot and
records local receive monotonic timestamps separately. `tegrastats` is a child
bounded to 2 seconds per snapshot; the full canonical sample windows were below
30 seconds/device. Only `free -b`, readable thermal sysfs files,
`tegrastats`, `nvidia-smi` query, `vcgencmd` reads and readable power-file probes
were used. No load, model, inference, conversion, benchmark, package command,
sudo, signal, mode/clock/fan change or process-stop action was sent.

The parser preserves each raw sample block and source label. It reports memory,
thermal, throttle and power observations without merging rails. Power values are
not measured inference energy: no workload ran, source boundaries are not
whole-device by default, and SSH receipt time is not treated as sensor time.

### Canonical runs

| Device | SSH alias | Run ID | SSH | Local duration | Samples / device timestamp gaps | Observed channels |
|---|---|---|---|---:|---|---|
| E1 | `pi5` | `20260916T163312816579Z` | 0 | 5.6 s | 5 / 1.030–1.037 s | `free -b` memory; thermal sysfs + `vcgencmd`; `vcgencmd get_throttled` |
| E2 | `nx` | `20260916T163228269358Z` | 0 | 15.5 s | 5 / 3.165–3.173 s | `free -b` + `tegrastats` memory; thermal sysfs + `tegrastats` |
| E3 | `agx` | `20260916T163249320243Z` | 0 | 15.9 s | 5 / 3.150–3.160 s | `free -b` + `tegrastats` memory; thermal sysfs + `tegrastats` |
| E5 | `nano` | `20260916T163323031572Z` | 0 | 16.3 s | 5 / 3.084–3.086 s | `free -b` + `tegrastats` memory; `tegrastats` thermal and labelled rails |

Feasibility observations from the raw samples:

- E1: memory is available in bytes; `vcgencmd` reports approximately
  49.4°C and `throttled=0x0`. No software/exposed power channel was available.
- E2: `tegrastats` reports RAM approximately 1891–1892/6854 MB and thermal
  labels around 30–30.5°C plus PMIC 50°C. `nvidia-smi`, vcgencmd and readable
  power files were unavailable.
- E3: `tegrastats` reports RAM approximately 2571–2574/14887 MB and thermal
  labels around 30.5–35.25°C plus PMIC 50°C. `nvidia-smi`, vcgencmd and
  readable power files were unavailable.
- E5: `tegrastats` reports RAM approximately 2191–2196/7607 MB and thermal
  labels approximately 49.5–51.7°C. It exposes separate `VDD_IN`,
  `VDD_CPU_GPU_CV` and `VDD_SOC` values (roughly 4.600–4.672 W,
  0.559–0.638 W and 1.397–1.437 W respectively); these remain labelled
  source rails with `unknown_rail_boundary`, are not summed and are not called
  whole-device power. `nvidia-smi` is present but returned `N/A` for queried
  telemetry fields. Several E5 thermal sysfs reads returned `No data available`
  and remain in captured stderr.

### Artifact locations and hashes

Each canonical directory contains `telemetry.stdout`, `telemetry.stderr`,
`telemetry.json`, `parsed_channels.json` and manifests. Hashes below are SHA-256
of the stored bytes; the per-run `manifest.json` contains all raw/summary file
hashes and `parsed_manifest.json` contains the parsed artifact hash.

| Device / run | Artifact directory | stdout SHA-256 | telemetry.json SHA-256 | parsed_channels.json SHA-256 |
|---|---|---|---|---|
| E1 / `20260916T163312816579Z` | [E1 run](../results/edge_readiness_v1/e2l1-005/E1/20260916T163312816579Z/) | `105926a7b51048e97077207940f05caa7acdb9f5c87be38d2e576ae39b2d844a` | `57f876b2168980cbfdd2a81e057a59ef75ae24323dbfadc0b679cb87e5025423` | `9a71e1c5dc10a4fbbc9a7250af2a77869fe665ec1569513f03f9a5cfaebaf7b9` |
| E2 / `20260916T163228269358Z` | [E2 run](../results/edge_readiness_v1/e2l1-005/E2/20260916T163228269358Z/) | `dc9c9b419d72a40989f9d411393cac96d0a49dd99e0e681dfe6c80a35b4290e7` | `527117584767b18ac3fc07224ac49ceb2703d58a0ff9fd79575e3456031af3c3` | `d2a7d1a90446d2c4830bb85fd9feee78df0ad405025094cd5a5ebaf013571a77` |
| E3 / `20260916T163249320243Z` | [E3 run](../results/edge_readiness_v1/e2l1-005/E3/20260916T163249320243Z/) | `c523fcb6d2e7dbc0f02f9f62eddf7e5589ea7227fadf92cffa27b13791a3977e` | `7f539a67f1f94ea6084ef31297c1a24634de79b29ae48d627bb38026d191c819` | `abf1028d77187374c9f8d517f2af63e60a31284c0633291a3f36bb14f2123d66` |
| E5 / `20260916T163323031572Z` | [E5 run](../results/edge_readiness_v1/e2l1-005/E5/20260916T163323031572Z/) | `61ae2c6ad2917457b8ee2ce6c9d2fff71369c482b61a76dfbf139f593268367a` | `926af1f41b3c138a74ce3edb26080965e35868d2e410fb76fb28a3c1137e8fd9` | `93e636ed9f9452b9f7b725ce6a0ec825002fc308a3cf871307d4fd9830d69b52` |

The first E2 attempt, retained as diagnostic evidence rather than canonical
output, is [E2 retry-diagnostic](../results/edge_readiness_v1/e2l1-005/E2/20260916T163147328463Z/): it captured 5 samples in ~15.6 seconds but returned shell code 1 because the initial loop's final `&&` condition was false. The collector was corrected before the canonical E2 run. Thus E2 has two bounded attempts totaling approximately 31.1 seconds; this small cumulative deviation is recorded rather than hidden. No device state was changed in either attempt.

### Local verification and limitations

```text
python -m unittest discover -s tests -p 'test_edge*.py' -v
Ran 26 tests ... OK
python -m py_compile scripts/edge_readiness/collect_inventory.py scripts/edge_readiness/collect_telemetry.py scripts/edge_readiness/measurement_harness.py
```

The parser tests include the captured-style `tegrastats` `RAM`/temperature/
rail labels and `vcgencmd` temperature/throttle fields. These samples establish
source/parser availability only. They do not establish controlled idle,
steady-state, clock alignment accuracy, external-meter boundary, cooling or
power-supply conditions; they must not feed `power_energy` or any inference
energy/FPS/AP statistic. A future real adapter still needs device workload
boundaries, synchronized source timestamps, actual content provenance and
reviewed power measurement scope.

**Git:** branch `luna1/e2l1-005-telemetry-feasibility` in worktree
`D:/Research/paper-luna-edge-002`; commit/push follows after final local checks.
No automatic PR or merge.

## L1A-006 — local Jetson adapter and protocol smoke package

**Status:** local implementation and CPU/mock protocol validation complete;
**NO-GO** for device execution, model transfer, TensorRT export/build,
inference, benchmark, installation, or device configuration changes.

Implemented on branch `luna1/e2l1-006-jetson-adapter-smoke`:

- `scripts/edge_readiness/jetson_adapter.py` defines a narrow injected-runtime
  contract with structured boundary errors, batch-1/640 I/O validation,
  explicit float16/float32 byte sizing, host/device buffer records, and
  synchronization checkpoints around copies and enqueue.
- E2 Xavier NX is the proposed first smoke target. E3 AGX Xavier is the
  subsequent TensorRT 8.5.2.2 compatibility candidate. E5 Orin Nano Super is
  the TensorRT 10.3.0 named-address/v3 candidate. These API paths are proposed
  from the observed inventory and still require target validation.
- `docs/JETSON_ADAPTER_SMOKE_V1.md` freezes the YOLO11n engineering reference
  checkpoint SHA-256
  `3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8`, keeps
  the server RTX engine as provenance only, specifies preprocessing,
  postprocessing/no-double-NMS, pre-declared output checks/tolerances, the
  train-only deterministic fixture, allowed resources, stop conditions, and
  future artifact layout. Version-specific API references are the official
  [TensorRT 8.5.3 release notes](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-853/pdf/TensorRT-Release-Notes.pdf),
  [TensorRT 10.3 Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1030/pdf/TensorRT-Developer-Guide.pdf),
  [8.x to 10.x migration patterns](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-8x-to-10x-c-api-patterns.html),
  and [TensorRT Python API documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html).
- `tests/test_jetson_adapter.py` covers proposed-plan boundaries, target/runtime
  and checkpoint rejection, E2 binding-list execution, E5 named-address/v3
  execution, byte/input validation, enqueue failure, synchronization ordering,
  and deterministic fixture order/content binding.

### Verification

The complete local edge suite was run after this package was added:

    python -m unittest discover -s tests -p 'test_edge*.py' -v
    Ran 34 tests ... OK

The adapter and existing edge modules also pass `py_compile`, and
`git diff --check` passes. No GPU runtime was imported; no SSH connection,
device command, engine build, model load, forward pass, or benchmark was run
for L1A-006. The accepted L1A-005 telemetry artifacts remain immutable,
including the disclosed E2 cumulative 31.109-second two-attempt deviation.

### Remaining prerequisites and handoff

Before any device-side smoke, Astra must authorize the target and confirm
target-local runtime/API availability, disk/dependency budget, target-native
engine provenance, binding/output contract, final stream/event completion,
timestamp/clock evidence, and the future output tolerance manifest. A missing
power boundary or clock alignment makes energy unavailable; it does not block a
separately labelled correctness/latency smoke. E3/E5 are not scored arms by
this package, and no official-test/AP result is implied.

**Git:** branch `luna1/e2l1-006-jetson-adapter-smoke`; scoped commit/push follows
after final checks. No PR or merge.

## L1A-007 — adapter ownership repair and executable CPU/mock smoke

**Status:** J1–J3 complete locally; **NO-GO** for SSH, model transfer,
target-native export/build, device inference, benchmark, installation, or
device configuration changes. The E2L1-007 inbox entry is committed unchanged
alongside this report.

### J1 — coherent buffer ownership

`scripts/edge_readiness/jetson_adapter.py` no longer invents device pointers.
The injected buffer manager supplies exactly one validated allocation per
binding, with positive unique pointer, owner, lifetime ID, shape/dtype/byte
count, host staging, and copy-stream handle. The same allocation map is used
for H2D, E2 binding-list v2, E5 named-address v3, and D2H. Host output payloads
are returned only after `after_output_copy` synchronization and an explicit
`mark_outputs_ready` boundary. Host-location bindings, missing/duplicate
pointers, ownership/lifetime omissions, wrong stream, copy failure, enqueue
failure, and named-address failure produce structured errors.

### J2 — harness integration and model contract

`AdapterHarnessBridge` runs through the accepted `run_session` contract for
both `inference_only` and `decoded_image_to_detections`. Tests verify warmup
exclusion, measured image sequence, both timing boundaries, final-copy
completion, synchronous postprocess, backend/synchronization metadata, and
failure propagation. The three generated 4x4 PPMs are now labelled
`synthetic_mock_fixture`; they are not CCTSDB train evidence. The proposed real
smoke requires the first three U42 CCTSDB train images in canonical order with
accepted source content hashes; that material is not currently present and is
recorded as a prerequisite rather than fabricated.

The YOLO11n no-NMS engineering validator freezes one native output `output0`
with shape `[1,7,8400]`, explicit source-to-engine mapping, raw comparison
before decode/NMS, and one later NMS owner. The former generic `[1,84,8400]`
fixture is rejected. Tolerance policy remains pre-observation and distinguishes
FP16/FP32 output domains; it is an engineering smoke threshold, not an
accuracy or precision claim.

### J3 — executable local orchestration

`scripts/edge_readiness/jetson_adapter_smoke.py` is an executable CPU/mock
dry-run only. It exercises the adapter through the accepted harness for both
timing boundaries and writes `manifest.json`, fixture/output contract files,
per-boundary session/event files, and a structured `failure.json` on mock
orchestration failure. It does not load a model, perform a forward pass, read
SSH, or enter a device mode. The future real target entry point is intentionally
not implemented at this gate. Remote collector lifecycle and per-source
clock/error persistence are explicitly listed as unimplemented real-session
prerequisites; no telemetry result is fabricated.

Power boundary or clock alignment is no longer an unconditional correctness
stop: those omissions make energy unavailable. Missing final synchronization,
wrong runtime/binding/model output, or missing accepted CCTSDB fixture remains a
correctness-smoke stop.

### Verification

    python -m unittest discover -s tests -p 'test_edge*.py' -v
    Ran 39 tests ... OK
    python -m py_compile scripts/edge_readiness/collect_inventory.py scripts/edge_readiness/collect_telemetry.py scripts/edge_readiness/measurement_harness.py scripts/edge_readiness/jetson_adapter.py scripts/edge_readiness/jetson_adapter_smoke.py tests/test_jetson_adapter.py tests/test_edge_jetson_adapter.py
    git diff --check
    python scripts/edge_readiness/jetson_adapter_smoke.py --target E2 --out-dir <temporary-dir> --dry-run
    exit 0; manifest status completed_mock_dry_run; real_device_execution false

No GPU runtime was imported and no edge command was executed. Previous
E2L1-005 telemetry artifacts and prior mock artifacts remain immutable.

### Remaining prerequisites and handoff

Before a real E2 smoke, Astra must authorize target-local runtime/API checks,
the target-native engine provenance, accepted U42 fixture materialization,
binding/output manifest, explicit stream/event completion, and bounded output
comparison. Future real telemetry also needs bounded remote child lifecycle,
per-source status/error records, and clock conversion evidence. Missing power
or clock alignment remains `energy: unavailable`, not a reason to recollect
telemetry or block a separately labelled correctness/latency smoke.

**Git:** branch `luna1/e2l1-006-jetson-adapter-smoke`; commit/push follows after
final checks using the established `NADUNGVN` account. No PR or merge.

## L1A-008 — E2 real-runtime preparation and verified source fixture

**Status:** CPU/mock acceptance and read-only E2 preparation complete;
**NO-GO** for TensorRT/CUDA execution, model transfer, source export, target
build, inference, benchmark, installation, power measurement, or device
configuration changes. The E2L1-008 inbox entry is staged unchanged with this
report.

### Runtime path and dependency evidence

Added `scripts/edge_readiness/jetson_runtime_provider.py`. The provider is
lazy: importing the module is CPU-safe and TensorRT is loaded only by explicit
engine deserialization. It verifies the engine SHA-256, introspects the
TensorRT 8 binding-list contract, keeps the CUDA memory/stream owner injected,
uses one allocation map for H2D/enqueue/D2H, returns fresh synchronized host
payloads, and frees prior allocations if a later allocation fails. Local fake
TensorRT/memory doubles cover load laziness, hash rejection, allocation/free
lifetime, repeated-call freshness and partial-failure cleanup; they do not
establish real TensorRT correctness.

Because the existing inventory did not expose decisive Python API details, one
bounded read-only E2 probe was run through alias `nx`. It reported host
`arar-desktop`, Python `3.8.10`, NumPy `1.17.4`, TensorRT `8.5.2.2`, CUDA
`11.4.19`, cuDNN `8.6.0.166`, and 94 GB free on `/`. `pycuda`, `cuda.cudart`,
ONNX, ONNX Runtime, Torch and Ultralytics were not discoverable. No CUDA or
TensorRT import/initialization was performed. The probe returned exit 0,
`timed_out=false`, elapsed `1.047 s`, command SHA-256
`89198aaeb9aacf2bd237506ee36c422c7ba07b37b97d936bce51fd9b0e29ea98`, and
remote script SHA-256
`5b84148b10a383d55fc6252af4b6599a0c2ec004f417aa619a6747c843f8641d`.
Raw stdout/stderr and the manifest are under
`results/edge_readiness_v1/e2l1-008/dependency_probe_20260917/`; no device
data was written.

### Fixture and source contract

`scripts/edge_readiness/e2_source_fixture.py` verified the actual train-only
JPEG bytes in canonical order `00006`, `00009`, `00028` at
`D:/Research/paper/data/processed/cctsdb2021_clean/train/images/`. Sizes are
63,138, 120,736 and 314,856 bytes; each accepted hash matches, with sequence
SHA-256
`7bfaaa99c4ed6b8ea2c92695f68747496a2ebb24400cf978b8630c27f41a6fc9`.
The edge-owned manifest is
`results/edge_readiness_v1/e2l1-008/fixture_binding_20260917/fixture_manifest.json`
(SHA-256
`51de19e309367f99a71a458a941b27d60d8e54fa8da5155bfd63495e6e154523`). No
image, checkpoint or large tensor was copied or staged. The helper records the
future decoder/resizer boundary and explicit BGR→RGB, letterbox pad 114,
NCHW float32 `/255` preprocessing; local Pillow is absent, so no decoded
tensor is fabricated.

The source reference is the frozen YOLO11n checkpoint
`results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt`, SHA-256
`3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8`, with
deployment policy `configs/deployment/yolo11n_fp16_trt10.json` and source
FP16 evidence under
`results/calibration_method_v1/rtx8000/yolo11n/eval/yolo11n_fp16_reference/`.
A future source reference must run in its own recorded Python/PyTorch/
Ultralytics/decoder environment and persist versions plus output hash; the
RTX TensorRT engine is provenance only and is not transferable to E2.

### Comparator, compatibility and staged workflow

`scripts/edge_readiness/e2_output_compare.py` compares native `output0`
`[1,7,8400]` before decode/NMS. The policy distinguishes source
`fp32_reference`, target `fp16` compute, and target binding/I/O `float32`.
It uses `abs(reference-target) <= absolute + relative*abs(reference)`;
decoded boxes (`x,y,w,h`) are pixels of the 640x640 letterboxed input and use
`5e-3/1e-2`; scores use `2e-3/1e-2`, all values must be
finite, and mismatch summaries are bounded. Score class order is pinned to
`prohibitory`, `mandatory`, `warning`. These are pre-observation engineering
thresholds, not an accuracy or precision bound.

The explicit future conversion contract is: source export, separately
authorized, from the accepted checkpoint with `imgsz=640`, batch 1, static
`[1,3,640,640]`, `opset=17`, `simplify=true`, no Q/DQ; retain ONNX SHA-256,
exporter versions, checker/schema and operator evidence. After review, build
FP16 only on E2 with its existing TensorRT 8.5.2.2 runtime, validate the
target-local API/bindings and write a target-native engine/hash; do not copy a
TensorRT 10 engine. Required target resources are the existing runtime,
target-supported CUDA allocation/copy/stream owner, private ONNX/engine,
accepted checkpoint/fixture/source-reference outputs and a fresh scoped
artifact directory. Missing Python bindings are reported as prerequisites,
not repaired by installation.

The executable staged workflow is
`scripts/edge_readiness/e2_correctness_workflow.py`: `preflight` verifies the
checkpoint/config hashes, `source-artifacts` verifies the fixture, while
`target-build` and `inference-compare` write structured
`blocked_not_authorized` manifests and disable transfer, CUDA allocation,
deserialize, build, forward, timing, power and SSH execution. Existing output
roots are rejected to prevent appending failure markers to prior success.
Power/clock evidence is not a correctness gate: without aligned boundaries,
energy remains unavailable.

### Verification and remaining prerequisites

    python -m unittest discover -s tests -p 'test_edge*.py' -v
    Ran 58 tests ... OK
    python -m py_compile scripts/edge_readiness/probe_e2_dependencies.py scripts/edge_readiness/jetson_runtime_provider.py scripts/edge_readiness/e2_source_fixture.py scripts/edge_readiness/e2_output_compare.py scripts/edge_readiness/e2_correctness_workflow.py tests/test_edge_e2_runtime.py
    git diff --check

The local checks use only CPU doubles and actual fixture hash reads. Remaining
prerequisites are a separately authorized source forward/export, target-local
TensorRT 8.5.2.2 API and CUDA-owner validation, target-native FP16 build and
engine binding/hash manifest, then a bounded correctness compare with final
stream completion. No scored timing/power/energy result is claimed; telemetry
collector lifecycle and clock conversion remain future prerequisites.

### Post-review correction: asynchronous D2H completion

Follow-up review reproduced a real boundary defect despite the previous
`51/51` pass: the fake device completed with byte value `42`, but the bridge
returned a zero payload because `OwnedBuffers.copy_device_to_host` snapshotted
the destination before asynchronous copy completion. The fix keeps each
mutable D2H destination pending and creates the immutable `HostTensor` only in
`mark_outputs_ready`, after the adapter's
`stream.synchronize("after_output_copy")`. The regression test now models that
ordering and asserts the returned output contains `42`. This closes the
buffer/runtime interface bug; it is not evidence of real CUDA correctness and
does not change the **NO-GO edge inference** status.

**Git:** branch `luna1/e2l1-006-jetson-adapter-smoke`; E2L1-008 code/docs,
scoped evidence and this report are pushed in the completion commit using the
established `NADUNGVN` account. No PR or merge; prior E2L1-005/E2L1-007
artifacts and other worktrees remain untouched.

## L1A-009 — asynchronous runtime-boundary repair

**Status:** R1–R3 local repair complete; **NO-GO** for edge inference, CUDA
allocation, engine deserialize/build, model transfer/forward, timing, energy,
installation or device configuration. This correction addresses a correctness
blocker discovered after the prior 51/51 result; it is not a GPU result.

### R1 — D2H completion and per-call ownership

`OwnedBuffers.copy_device_to_host` now retains the mutable destination instead
of immediately converting it to immutable bytes. The adapter's existing
`stream.synchronize("after_output_copy")` is the completion boundary; only
then does `mark_outputs_ready` snapshot host payloads. `begin_inference` clears
pending/ready/output state before H2D, so a failed second call cannot expose a
previous output, and the owner rejects reuse after `free()` while keeping
cleanup idempotent.

The deferred CPU double queues D2H and writes `42` at synchronization, then
writes `43` on the next call. The provider-plus-owner-plus-adapter regression
asserts both values, proving the reproduced `42 → 0` stale-snapshot failure is
closed at the interface boundary. Tests also cover output access before
readiness, after free, partial allocation cleanup, descriptor mismatch and
failed second call.

### R2 — runtime-owned evidence and JSON-safe diagnostics

`TensorRTProvider` now requires the imported TensorRT module's observed
`__version__` to match the target profile before deserialization. It queries
actual legacy binding/tensor locations, rejects missing/unknown location data
and dynamic/host bindings through the existing descriptor validator, retains
logger/runtime/engine/context lifetimes, rejects a null context and invokes
the YOLO11n native `output0 [1,7,8400]` contract before adapter creation.
The E2 Python environment still has no discoverable concrete CUDA owner
(`pycuda`/`cuda.cudart` absent); `CudaMemoryOwner` therefore remains an
injected contract tested by doubles, not a claim of installed real CUDA
execution.

The comparator now validates finite nonnegative tolerances and emits JSON-safe
`null` fields plus explicit nonfinite counters/reasons; `json.dumps(...,
allow_nan=False)` is covered. Box coordinates are documented as decoded `xywh`
pixels in the 640x640 letterboxed input, separate from the three pinned score
classes and from compute/I/O dtype labels.

### R3 — bounded artifact/error behavior

The staged workflow now binds the accepted deployment config SHA-256
`86973fe56b850cb5b119773b402243d36a98809b894e8e8a13493fb0edd30628`, schema,
YOLO11n model family and 640/3-channel input before reporting preflight. The
source-artifact stage explicitly records model output/ONNX/forward as not
executed. CLI failures write `failure.json` only inside a root created by that
invocation; a second invocation against a successful or partial root refuses
without mutation and preserves the original evidence.

The probe implementation now uses E2 alias validation and a remote `timeout
50s` child under a local timeout greater than 50 seconds. The retained
E2L1-008 probe artifact was **not rerun**; its original 1.047-second evidence
and hashes remain immutable, and the new remote-bound behavior is unclaimed
for that historical capture.

### Verification and handoff

    python -m unittest discover -s tests -p 'test_edge*.py' -v
    Ran 58 tests ... OK
    python -m py_compile scripts/edge_readiness/jetson_adapter.py scripts/edge_readiness/jetson_runtime_provider.py scripts/edge_readiness/e2_output_compare.py scripts/edge_readiness/e2_correctness_workflow.py scripts/edge_readiness/probe_e2_dependencies.py tests/test_edge_e2_runtime.py
    git diff --check

No edge command was executed in this correction. The E2L1-009 inbox entry is
committed unchanged. Remaining prerequisites are a separately authorized
concrete CUDA-owner decision, target-local TensorRT 8.5.2.2 API/binding check,
source reference/export, target-native engine and final synchronized smoke.
Missing aligned power/clock evidence keeps energy unavailable and does not
change the correctness NO-GO.

**Git:** branch `luna1/e2l1-006-jetson-adapter-smoke`; correction commit/push
uses the established `NADUNGVN` account. No PR or merge; historical results
and other worktrees remain untouched.

## L1A-010 — concrete CUDA runtime owner and default-disabled allocation smoke

**Status:** local implementation complete; **NO-GO** for real CUDA calls,
allocation smoke execution, TensorRT/model load, build, inference, benchmark,
installation or device configuration. The owner is ready for Astra review and
separate authorization of the small E2 memory roundtrip.

### Concrete route and native evidence

Implemented `scripts/edge_readiness/cuda_runtime_owner.py` as a lazy,
injected-loader `ctypes` binding to the CUDA 11.4 Runtime C API. Importing the
module does not load `libcudart`; `CudaRuntime.open()` is the explicit load
boundary. The owner configures and checks raw return codes for `cudaMalloc`,
`cudaFree`, `cudaMemcpy`, `cudaStreamCreate`, `cudaStreamSynchronize`,
`cudaStreamDestroy` and `cudaGetErrorString`, with structured `AdapterError`
codes. It owns one stream, validates positive pointers/sizes and rejects a
non-owned stream. Its context policy is explicit: use the caller's current
CUDA primary context, without creating or switching a context.

The selected copy mode is synchronous `cudaMemcpy` for both H2D and D2H. This
is deliberate: ordinary Python `bytes`/`bytearray` are not silently treated as
pinned memory. The one-stream completion call remains part of the contract,
and `OwnedBuffers.free()` synchronizes before attempting every device free,
including after an earlier cleanup error. The TensorRT provider release order
keeps the logger alive until the runtime is released. The implementation does
not claim target compatibility until the separately authorized smoke loads the
target library.

The official ABI/API references are the [CUDA Runtime API v11.4
reference](https://docs.nvidia.com/cuda/archive/11.4.0/pdf/CUDA_Runtime_API.pdf)
and [CUDA stream synchronization
documentation](https://docs.nvidia.com/cuda/archive/11.4.0/cuda-runtime-api/group__CUDART__STREAM.html).

One bounded read-only E2 inspection was performed through alias `nx` only. It
recorded host `arar-desktop`, no `nvcc` (`nvcc: not found`), loader discovery of
`libcudart.so.11.0`, and both symlinks resolving to
`/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so.11.4.298`. The
installed header resolves to
`/usr/local/cuda-11.4/targets/aarch64-linux/include/cuda_runtime_api.h` and
declares the observed allocation/free/stream/error APIs plus
`cudaMemcpyAsync`; symbol inspection observed `cudaMalloc`, `cudaFree`,
`cudaMallocHost`, `cudaFreeHost`, `cudaMemcpyAsync` and
`cudaStreamSynchronize`. No library was loaded and no CUDA function was
called. The read-only artifact is under
`results/edge_readiness_v1/e2l1-010/cuda_runtime_probe_20260917/`:
returncode `0`, `timed_out=false`, local elapsed `1.734s`, remote child bound
`50s`, local bound `60s`, command SHA-256
`ba098b7bacc1354775e27828fa3f60a002756f4fe4787e3256681703ce7503ff`, remote
script SHA-256
`1910e2a351b4d37041b3459281b797e91e5f70cc8fa821bed679e1f9cc7ebb21`, manifest
SHA-256 `4d21d95cbd473c5c6e389d1945bb53af1dc6e3358d1ea7af6ad944f1654c4f29`,
stdout SHA-256
`8357c470ec17ad3351a4e285cd3994805d73c7944bf14dc2e09a9b59ea13e3a0`, and
empty stderr SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

### Allocation-only smoke and source handoff

`scripts/edge_readiness/e2_cuda_allocation_smoke.py` is default-disabled and
prints a plan without loading CUDA:

    python scripts/edge_readiness/e2_cuda_allocation_smoke.py --target E2

The candidate command, **not executed in L1A-010**, is:

    python scripts/edge_readiness/e2_cuda_allocation_smoke.py --target E2 --execute-real-device --out-dir results/edge_readiness_v1/e2l1-010/<run_id>/allocation_smoke

It is E2-only, one owned stream, one deterministic 4096-byte payload, one
allocation capped at 1 MiB, one synchronous H2D/D2H roundtrip, exact-byte
comparison, synchronization and cleanup. It must write only a scoped JSON
manifest (or failure JSON); it must not touch an engine, model, TensorRT,
inference, benchmark, power/thermal collection, package state or device
configuration. Astra must authorize this command separately before any real
device invocation.

The later source/reference handoff remains unchanged and is explicitly future
work: frozen YOLO11n checkpoint SHA
`3e5fc7a2148c16539cd9fb7cc7cacd81a4eec1dfc28143cdf9b6dcd872ba4ab8`, the
verified train fixture sequence `00006`, `00009`, `00028` with sequence SHA
`7bfaaa99c4ed6b8ea2c92695f68747496a2ebb24400cf978b8630c27f41a6fc9`, input
`[1,3,640,640]`, letterbox/RGB/`[0,1]` recipe, ONNX `opset=17`, static shape,
`simplify=true`, no Q/DQ, and native output `output0 [1,7,8400]`. No source
forward/export was run or claimed; no TRT10 engine is transferred.
The intended source-run location is the Windows local worktree
`D:\Research\paper`, not a Linux server path (CPU/reference/export only, after
separate authorization); E2 is reserved for the later target-native TensorRT
build and synchronized compare.
Artifact roles are separated: checkpoint SHA identifies frozen weights, the
fixture manifest/sequence SHA identifies the three source inputs, the future
ONNX SHA identifies source export, and a future E2 engine SHA identifies only
the target-native TensorRT artifact. No large image, tensor, ONNX or engine
bytes are added to this handoff.

### Verification

    python -m unittest discover -s tests -p 'test_edge*.py'
    Ran 63 tests ... OK
    python -m unittest tests.test_edge_e2_runtime
    Ran 24 tests ... OK
    python -m py_compile scripts/edge_readiness/cuda_runtime_owner.py scripts/edge_readiness/e2_cuda_allocation_smoke.py scripts/edge_readiness/probe_e2_cuda_runtime.py scripts/edge_readiness/jetson_runtime_provider.py tests/test_edge_e2_runtime.py
    git diff --check (source/docs; the raw vendor-header capture is retained verbatim)

The tests use injected C-function doubles only and cover allocation/copy/sync/
free failures, exact small-copy roundtrip, repeated stream use and close,
partial cleanup, and provider integration. The full repository discovery was
not used as acceptance evidence because this workstation lacks unrelated
NumPy/PIL dependencies and canonical fixtures. No E2 CUDA call, build or
inference was performed.

**Git:** L1A-010 includes the concrete owner, default-disabled smoke, scoped
read-only E2 evidence, tests and this report. Push uses the `NADUNGVN` account;
the E2L1-010 inbox entry remains unchanged.

## L1A-011 — Python 3.8 lifecycle repair and one E2 allocation roundtrip

**Status:** C1/C2 repair complete; the conditional one-shot E2 allocation-only
GO completed successfully. NO-GO remains in force for source/model
forward/export, TensorRT engine build/deserialization/inference, benchmarking,
telemetry sampling, installation and device configuration.

### C1/C2 implementation

Commit `aa31c42825fa9995a6a8d1048a87d9c4fb8385f6` adds the Python 3.8-safe
`edge_errors.py` import boundary and exclusive strict-JSON writer using
`open(..., "x", encoding="utf-8", newline="\n")`; the executable smoke path no
longer imports the Python-3.10-only writer API. Runtime load, stream creation
and owner construction now occur after the new output-root check. Cleanup
records stream/runtime failures, preserves the primary error alongside cleanup
errors, and writes terminal PASS only after allocation free, stream destruction
and runtime close succeed. The owner tracks live allocation generations, so a
reused CUDA address is freed again in its new generation while repeated free
within one generation remains idempotent.

Local verification: `68/68` edge tests PASS, including `29/29`
`test_edge_e2_runtime`; Python compile and diff checks pass. Tests cover
load/stream-create/allocation/copy/sync/free/destroy/close failures, combined
primary-plus-cleanup errors, successful roundtrip followed by cleanup failure,
existing-root preservation and reused allocation addresses. The local tests
use injected C doubles only; they are not Python 3.8 execution evidence.

### E2 preflight, transfer and one-shot result

Before dispatch, alias `nx` was verified read-only as user `arar`, cwd
`/home/arar`, hostname `arar-desktop`, architecture `aarch64`, Python
`3.8.10`, and libcudart realpath
`/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so.11.4.298`, matching
the accepted E2 probe. Only these three committed sources were transferred to
the new user-owned directory
`/tmp/luna1-e2l1-011-cKHdoJn/scripts/edge_readiness/`:

    e2_cuda_allocation_smoke.py  6dd971ac4bfc90cb782da81af0ede12b65eff9bedf76bc933af873c98ce16a85
    cuda_runtime_owner.py        0512964f58c4bffae85be9d83bb6c7748041e581434ed88bff7c87b2277f283b
    edge_errors.py               c13691982d80dc714b935d367ef35bb371b07dfd591a8856be3d23ac1aaa7602

The remote output root was new and user-owned:
`/tmp/luna1-e2l1-011-cKHdoJn/e2l1-011-allocation-v1`. The exact sanitized
dispatch used remote child timeout `60s` and local wait timeout `75s`:

    ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes nx timeout 60s python3.8 /tmp/luna1-e2l1-011-cKHdoJn/scripts/edge_readiness/e2_cuda_allocation_smoke.py --target E2 --execute-real-device --out-dir /tmp/luna1-e2l1-011-cKHdoJn/e2l1-011-allocation-v1

The single dispatch returned `0`, was not timed out, and took `1.109s` local.
The result is `executed_allocation_only`: exactly one 4096-byte allocation,
synchronous H2D/D2H, completion and cleanup. Expected and observed payload
SHA-256 both equal
`4e441a3533bb2c10cd5649981d395744213e09a336746b5a3458fee4057205ec`.
The manifest records library identity
`/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so.11.4.298`,
`stream_close=ok` and `runtime_close=ok`. No CUDA reset, TensorRT/model
operation, build, inference, benchmark, telemetry, install, sudo or device
configuration was performed, and no retry was made.
This is a copy-correctness result only; it makes no performance or isolated-GPU
claim and does not require a globally idle device.

Sanitized evidence is retained under
`results/edge_readiness_v1/e2l1-011/`: `manifest.json` SHA-256
`e2c29422dc49af47cbeacad8e727b955764cc0bb33157db423a7dac829dde823`,
`remote.stdout` SHA-256
`e2c29422dc49af47cbeacad8e727b955764cc0bb33157db423a7dac829dde823`, empty
`remote.stderr` SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, and
`dispatch.json` SHA-256
`df8c59a7309f5bcb2db581ba6aca414c55d24e90bc9e36b24df843558b333599`.
The dispatch file includes the full commit, target verification and all
source-transfer hashes. No model, ONNX, engine, image or payload file was
transferred or persisted.

The runtime call path can initialize/use the process CUDA context; this smoke
does not claim “no context initialization”. The 4096-byte/1 MiB limit bounds
only the requested device allocation, not total CUDA driver/context memory.
It creates no workload in another process and does not reset or reconfigure
the device. The source/reference export location remains the Windows local
`D:\Research\paper` worktree; source forward/export is future work and was not
run here.

**Git:** L1A-011, scoped code/docs and sanitized evidence are ready to push
with the `NADUNGVN` account. Stop after this allocation result; a model smoke
requires separate review.

## L1A-012 — CPU source-bundle runner implemented; awaiting complete source environment

**Status:** P1 implementation and tests complete; local real bundle execution
is **BLOCKED by missing dependencies**, so no local bundle was dispatched. No
E2 SSH/transfer/build/deserialization/model inference/benchmark was performed
in this entry. The accepted E2 copy result remains unchanged.

### P1 implementation

Commit `05624d7333966de0c16e1a33a79359fb0a3663dc` adds
`scripts/edge_readiness/e2_source_bundle.py` and
`tests/test_edge_source_bundle.py`. The runner is CPU-only and lazy-imports
the source stack. It verifies the frozen checkpoint SHA and the three accepted
train fixtures in order, protects fixture hashes before/after, copies the
checkpoint only into a private output area, freezes each input tensor once,
records little-endian contiguous float32 bytes, and keeps all ONNX/reference
tensor bytes under private retention paths.

It uses Ultralytics `LetterBox` semantics with BGR decode, RGB conversion,
normalization by `255.0`, NCHW `[1,3,640,640]`, CPU float32 and no test/dev or
negative split. Native FP32 forwards occur before a separate exporter model is
created. Export is constrained to CPU, float32 ONNX, opset 17, static batch-1
640, no NMS/end2end wrapper, `simplify=true`, no Q/DQ/calibration/TensorRT;
the private checkpoint copy is the only exporter input. ONNX checker/I/O
shape/dtype/name validation and ORT `CPUExecutionProvider` validation are
required. Native/ORT `[1,7,8400]` outputs are compared with
`abs(observed-reference) <= 1e-5 + 1e-4*abs(reference)`, bounded failures and
finite checks. Exporter internal forward calls are counted separately from the
three native and three ORT fixture forwards.

The public output contract is `public/plan.json`, terminal
`manifest.json` or `failure.json`, `report.md`, `run.log` and `index.json`.
The private `checkpoint/`, `onnx_export/`, `inputs/`, `native_reference/` and
`onnx_reference/` paths are explicit in the manifest and must not be published
to Git. New roots are exclusive; an existing or partial root is refused.
Missing dependencies fail closed without package download/install, and every
failure records attempted/completed counters.

Verification is `74/74` edge tests PASS, including six new source-bundle tests:
synthetic helper/producer byte equivalence, injected end-to-end bundle,
private-copy protection, dependency fail-closed behavior, semantic/shape
mismatch, and output-root collision. `py_compile` and `git diff --check` pass.

### Local environment decision

The inspected local environment is
`D:\Research\paper\local\measurement_audit_env`: Ultralytics `8.4.102`,
Torch `2.8.0+cu129`, ONNX Runtime `1.24.3`, OpenCV `4.13.0` and Pillow
`12.3.0` are present, but `onnx` and `onnxslim` are absent. Therefore the
complete source/export gate is not met. No automatic installation was attempted
and `results/edge_readiness_v1/e2l1-012-source-v1/` was not created. The
local checkpoint path exists at `D:\Research\paper\results\yolo11n_cctsdb_clean_s42_v2\weights\best.pt`,
but that does not overcome the missing exporter dependencies.

### Server operator command

Run once in the existing complete main-server environment, preserving the main
worktree by using a detached branch-scoped worktree. This is the exact command
sequence; it does not alter the main checkout or install anything:

    cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new
    git fetch origin luna1/e2l1-006-jetson-adapter-smoke
    git worktree add --detach /tmp/luna1-e2l1-012-source 05624d7333966de0c16e1a33a79359fb0a3663dc
    /home/ubuntu/Dung_TDTU/nighttime-tsd-new/local/g0_size_env/bin/python /tmp/luna1-e2l1-012-source/scripts/edge_readiness/e2_source_bundle.py --source-root /home/ubuntu/Dung_TDTU/nighttime-tsd-new --out-dir /home/ubuntu/Dung_TDTU/nighttime-tsd-new/results/edge_readiness_v1/e2l1-012-source-v1 --commit 05624d7333966de0c16e1a33a79359fb0a3663dc

Before running, the operator must verify that the output root is absent and
that this interpreter has Ultralytics 8.4.102 plus Torch, `onnx`, `onnxslim`,
ONNX Runtime, OpenCV and Pillow. The runner itself enforces CPU visibility,
two CPU threads, no downloads/auto-install and ORT CPU provider. It writes
private ONNX/checkpoint/tensor bytes only under the new output root; publish
only the `public/` JSON/text artifacts after reviewing hashes and counters. Do
not use another environment, add export options, retry a failed run, or reuse
the other lane's output root. The runner is source CPU/reference preparation,
not a complete camera/preprocess or E2 deployment claim.

The intended source-run location is the Linux main-server repository
`/home/ubuntu/Dung_TDTU/nighttime-tsd-new`; the Windows path
`D:\Research\paper` is only the local development/source-reference location.
Actual source forward/export and any numerical PASS/FAIL remain pending the
operator's one server run. **Git:** code, this report and the unchanged
E2L1-012 inbox are pushed using `NADUNGVN`; stop after the source-bundle result.

## L1A-013 — source-bundle integration repairs complete; conditional server GO

**Status:** B1–B3 implementation and actual-runtime-boundary CPU tests are
complete. No local dependency installation, frozen-checkpoint load/forward,
export, inference, benchmark, E2 SSH or device operation was performed.

Commit `5ad2eb166a0cf1a6c67edb1353ad6ee614453f83` repairs the source runner:

- NumPy is part of the required prepared module contract and is bound in
  `self.modules`; native conversion, native output packaging and ORT dispatch
  are exercised through `UltralyticsSourceRuntime`.
- The exporter receives a hash-verified `best.pt` inside its owned
  `private/onnx_export/` workspace, follows the installed Ultralytics
  `pt_path` sibling-output semantics, records separate export flags/counters,
  and checks the exact resolved ONNX path plus source/private checkpoint hashes
  before and after export.
- Native model binding is CPU/eval/float32 and enforces the frozen YOLO11n
  Detect/nc/class-order/end2end/export/training/xyxy/decoded-output contract.
  ONNX validation enforces `images`/`output0`, opset 17, static float32
  `[1,3,640,640]` to `[1,7,8400]`, and no Q/DQ/NMS wrappers.
- `YOLO_AUTOINSTALL=false` is set before imports; Ultralytics is pinned to
  `8.4.102`; Torch is set to intra-op 2/inter-op 1; ORT uses explicit
  SessionOptions and only `CPUExecutionProvider`; executable HEAD and helper
  hashes are recorded and explicit commit/dirty-worktree mismatches fail.
- One contiguous little-endian float32 NumPy array supplies both saved input
  bytes and runtime input. Decode, non-square LetterBox, padding, RGB/layout,
  normalization and input hashes are recorded and byte equality is asserted
  before each consumer.
- Numerical comparison counts all mismatches, separates box/score counts and
  maxima, caps examples at 20, rejects nonfinite/rehashed-invalid bytes, and
  writes complete terminal provenance for numerical FAIL and partial failures.
  Fixture/checkpoint before/after protection, attempted/completed counters,
  exporter internal-forward preservation and terminal writer errors are tested.

Verification after the commit: **82/82 edge tests PASS**, including 14 source
bundle tests. The actual-runtime tests use the existing local measurement
environment for NumPy `2.4.2`, Torch `2.8.0+cu129`, Ultralytics `8.4.102` and
OpenCV `4.13.0` on deterministic non-square synthetic input; injected model,
exporter, ONNX checker and ORT session doubles cover the missing external
export/session boundaries. `py_compile`, `git diff --check`, and explicit
HEAD/clean-executable-source provenance verification also pass. The original
inputs, fixture order and comparison equation are unchanged.

The local environment still lacks `onnx` and `onnxslim`, so the conditional GO
is the single user-operated foreground CPU source-bundle run in the existing
complete server environment. The old `05624d7` command is superseded and must
not be run. After pushing this commit, run exactly once:

    cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new
    git fetch origin luna1/e2l1-006-jetson-adapter-smoke
    test ! -e /tmp/luna1-e2l1-013-source
    test ! -e /home/ubuntu/Dung_TDTU/nighttime-tsd-new/results/edge_readiness_v1/e2l1-012-source-v1
    git worktree add --detach /tmp/luna1-e2l1-013-source 5ad2eb166a0cf1a6c67edb1353ad6ee614453f83
    test "$(git -C /tmp/luna1-e2l1-013-source rev-parse HEAD)" = "5ad2eb166a0cf1a6c67edb1353ad6ee614453f83"
    /home/ubuntu/Dung_TDTU/nighttime-tsd-new/local/g0_size_env/bin/python /tmp/luna1-e2l1-013-source/scripts/edge_readiness/e2_source_bundle.py --source-root /home/ubuntu/Dung_TDTU/nighttime-tsd-new --out-dir /home/ubuntu/Dung_TDTU/nighttime-tsd-new/results/edge_readiness_v1/e2l1-012-source-v1 --commit 5ad2eb166a0cf1a6c67edb1353ad6ee614453f83

The command assumes the existing main-server interpreter already has all
required dependencies; it installs nothing, retries nothing, uses no nohup or
GPU-idle guard, and leaves the main worktree untouched. Publish only the new
`public/` JSON/text evidence after reviewing it; retain private checkpoint,
ONNX and tensor files on the server. Report any numerical FAIL honestly and
stop for Astra review. **Git:** this entry, scoped code/tests and the
unchanged E2L1-013 inbox are to be pushed with `NADUNGVN`; no E2 SSH/transfer,
build, inference or benchmark is authorized here.

### Operator attempt audit and corrective dispatch

The first operator run was pushed as commit
`348bd475b53712eeacbf128d04cf2bc19efc3668`. Its public evidence is retained at
`results/edge_readiness_v1/e2l1-012-source-v1/public/` and reports a
fail-closed `NATIVE_MODEL_DEVICE_DTYPE_MISMATCH` before any input/native/export
operation (`inputs_attempted=0`, `export_invocations_attempted=0`). The
checkpoint and all three fixtures were verified unchanged; the server
environment was complete (`numpy 2.4.4`, `torch 2.5.1+cu121`,
`ultralytics 8.4.102`, `onnx 1.21.0`, `onnxruntime 1.24.4`, `onnxslim 0.1.94`,
OpenCV 5.0.0, Python 3.11.15), with CPU threads 2/1 and auto-install false.
The failure was in the runner's validation: YOLO's CPU model contains
non-floating `torch.int64` bookkeeping buffers, which are valid and must not
be treated as floating model weights.

Corrective executable commit
`7dd178af4c001d074d1efbd4bbbafcb767dc6263` allows integer bookkeeping buffers
while still requiring all parameters and floating buffers to be CPU float32;
it adds the regression for this exact boundary. Verification is now
**83/83 edge tests PASS**, with `py_compile` and `git diff --check` passing.
The prior failed output root is preserved. The single corrected operator run
must use a new absent output root:

    cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new
    git fetch origin luna1/e2l1-006-jetson-adapter-smoke
    test ! -e /tmp/luna1-e2l1-013-source-v2
    test ! -e /home/ubuntu/Dung_TDTU/nighttime-tsd-new/results/edge_readiness_v1/e2l1-013-source-v2
    git worktree add --detach /tmp/luna1-e2l1-013-source-v2 7dd178af4c001d074d1efbd4bbbafcb767dc6263
    test "$(git -C /tmp/luna1-e2l1-013-source-v2 rev-parse HEAD)" = "7dd178af4c001d074d1efbd4bbbafcb767dc6263"
    /home/ubuntu/Dung_TDTU/nighttime-tsd-new/local/g0_size_env/bin/python /tmp/luna1-e2l1-013-source-v2/scripts/edge_readiness/e2_source_bundle.py --source-root /home/ubuntu/Dung_TDTU/nighttime-tsd-new --out-dir /home/ubuntu/Dung_TDTU/nighttime-tsd-new/results/edge_readiness_v1/e2l1-013-source-v2 --commit 7dd178af4c001d074d1efbd4bbbafcb767dc6263

This is the replacement conditional GO command; do not rerun the old
`05624d7` or `5ad2eb1` commands and do not overwrite
`e2l1-012-source-v1`. Publish only the corrected run's public JSON/text
evidence, then stop for Astra audit. No E2 SSH/transfer/build/inference/
benchmark is authorized.

### Corrective operator result — stop for Astra review

The corrected run's public evidence was pushed in commit
`34a542b2f787d7ef60dc3d3125cecad78e0c16f9`. Execution completed under the
requested code commit `7dd178af4c001d074d1efbd4bbbafcb767dc6263`, with the
detached worktree HEAD matching exactly. Checkpoint and all three fixture
hashes matched before/after. The server environment was complete and recorded
as NumPy 2.4.4, Torch 2.5.1+cu121, Ultralytics 8.4.102, ONNX 1.21.0,
ONNX Runtime 1.24.4, onnxslim 0.1.94, OpenCV 5.0.0, Python 3.11.15; Torch
threads were 2/1, auto-install was false, and ORT was CPU-only.

The source bundle completed all three native forwards, one export, four
exporter-internal forwards, all three ORT forwards and all three comparisons.
ONNX validation passed: `images`/`output0`, static `[1,3,640,640]` to
`[1,7,8400]`, float32, opset 17, and no Q/DQ/NMS wrappers. Comparisons were:

- `00006`: PASS, 0 failing elements.
- `00009`: PASS, 0 failing elements.
- `00028`: FAIL, 1 box element at flat index 16402; observed absolute error
  `1.52587890625e-05` versus the unchanged limit
  `1.0292587280273439e-05`.

Terminal status is therefore `execution_complete_numerical_fail`, not a
runtime failure. The complete public manifest preserves model semantics,
input/output hashes and bindings, comparison counters, private inventory and
the exact equation. **NO-GO for tolerance changes, rerun, second environment
or device deployment; stop for Astra review of this single numerical
discrepancy.**

## L1A-014 — E2 model diagnostic workflow implemented; target execution held

**Status:** local implementation and protocol tests complete; server retrieval,
transfer, TensorRT build/deserialization/inference and benchmark remain
NO-GO pending Astra review. The accepted source bundle is consumed as a
provenance-bound diagnostic input even though its strict source verdict is
FAIL; this workflow never converts that result into PASS.

Executable commit: `d9423f21d24ff0217e9424c9d6b7037602fd3596`.
It adds `scripts/edge_readiness/e2_model_smoke.py` and its tests. The workflow
is default-disabled and has one target profile, E2 / Jetson Xavier NX /
TensorRT `8.5.2.2`.

### Implemented contracts

- The input bundle binds exactly ten private binaries from the accepted source
  manifest: one ONNX plus `inputs/{00006,00009,00028}.bin`,
  `native_reference/{00006,00009,00028}.bin` and
  `onnx_reference/{00006,00009,00028}.bin`. The copied checkpoint and any
  arbitrary manifest path are rejected. Paths are normalized under `private/`,
  traversal and symlink components are rejected, and every size/SHA-256,
  shape, little-endian float32 byte contract and finite value is revalidated.
  The accepted ONNX hash is
  `bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7`.
- The production target path is parser/builder → TensorRTProvider →
  OwnedBuffers/CUDA owner → JetsonRuntimeAdapter. It requires a target-local
  TensorRT parser, one fixed 1 GiB builder workspace, FP16 builder flag,
  disabled TF32 when available, no timing cache, no retries, and fresh private
  engine/output paths. It validates E2 runtime identity, engine hash, binding
  order/name/location/shape/dtype and raw `output0 [1,7,8400]` before dispatch.
- E2 uses exact frozen float32 input bytes, with no decode/preprocess. It
  executes one owned context/stream and exactly three sequential enqueues,
  with no warmup/repeat/latency/energy benchmark. Full finite output checks,
  attempted/completed counters and partial/unknown failure states are
  recorded; cleanup does not kill unrelated work or claim completion after a
  timeout.
- The public manifest keeps three results separate:
  `export_discrepancy.source_native_vs_source_onnx` (immutable strict source
  result), `tensorrt_discrepancy.target_vs_source_onnx` (export-fixed target
  diagnostic), and `tensorrt_discrepancy.target_vs_source_native` (total
  target-vs-native diagnostic). The source strict FAIL remains visible even if
  both target comparisons PASS. Target comparisons retain the frozen
  `e2_output_compare.py` thresholds: boxes absolute `5e-3`, relative `1e-2`;
  scores absolute `2e-3`, relative `1e-2`. These are engineering thresholds,
  not accuracy claims. The channel-1/anchor-8002 neighborhood is recorded
  without score cutoff or NMS.

### Verification and boundary

The new suite has 11 tests and the complete edge suite is **94/94 PASS**;
`py_compile`, Python-3.8 AST compatibility checks and `git diff --check` pass.
Tests cover exact allowlisting/traversal/tampering, parser errors and runtime
mismatch, bounded FP16 builder settings, production parser/provider/
owner/adapter orchestration with injected TRT/CUDA calls, distinct outputs for
three images, source FAIL coexisting with target PASS, second-enqueue failure,
unknown timeout, default-disabled execution and output collision. No real
TensorRT/CUDA import or target call was made locally.

### Proposed later operator staging and command (not authorized in L1A-014)

After Astra review only, retrieve into a fresh scoped staging root the public
source manifest plus exactly the ten allowlisted private files above. Do not
retrieve/transfer `private/checkpoint/best.pt`, the whole repository, secrets,
or arbitrary files. Keep the manifest at
`/tmp/luna1-e2l1-014-source-bundle/public/manifest.json` and preserve its
`private/` relative layout. The workflow emits the explicit allowlist in
`public/plan.json`; model binaries and full output tensors remain private.

Proposed foreground command after review:

    cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new
    test ! -e /tmp/luna1-e2l1-014-model-smoke
    test ! -e /home/ubuntu/Dung_TDTU/nighttime-tsd-new/results/edge_readiness_v1/e2l1-014-model-smoke-v1
    git fetch origin luna1/e2l1-006-jetson-adapter-smoke
    git worktree add --detach /tmp/luna1-e2l1-014-model-smoke d9423f21d24ff0217e9424c9d6b7037602fd3596
    test "$(git -C /tmp/luna1-e2l1-014-model-smoke rev-parse HEAD)" = "d9423f21d24ff0217e9424c9d6b7037602fd3596"
    /home/ubuntu/Dung_TDTU/nighttime-tsd-new/local/g0_size_env/bin/python /tmp/luna1-e2l1-014-model-smoke/scripts/edge_readiness/e2_model_smoke.py --bundle-root /tmp/luna1-e2l1-014-source-bundle --manifest /tmp/luna1-e2l1-014-source-bundle/public/manifest.json --out-dir /home/ubuntu/Dung_TDTU/nighttime-tsd-new/results/edge_readiness_v1/e2l1-014-model-smoke-v1 --target E2 --execute-real-device --build-timeout-seconds 900 --inference-timeout-seconds 180

This command is intentionally withheld from execution in this entry. No
source re-export, additional allocation smoke, E2 SSH, transfer, build,
inference, benchmark, package installation, device reset or configuration
change is authorized. Publish only public JSON/text evidence after a later
authorized run; Astra then audits and decides the next gate.

## L1A-015 — repaired E2 dispatch boundary, bounded stages and cleanup evidence

**Status:** local implementation and CPU/mock verification complete. No E2
SSH, source retrieval/transfer, TensorRT import, engine build or
deserialization, model inference, benchmark, installation or configuration
change was performed.

Executable repair commit: `1603732` (`fix(luna1): enforce bounded E2 model
smoke lifecycle`). The unchanged Astra inbox entry E2L1-015 is included in
that commit.

### Repairs delivered

- `load_source_bundle()` now checks the raw-byte SHA-256
  `df2e81e943c471cb540dd837151e87f7799510b01c050be1b2d14e2869f760be` for
  `results/edge_readiness_v1/e2l1-013-source-v2/public/manifest.json` from
  accepted commit `34a542b2f787d7ef60dc3d3125cecad78e0c16f9`, before deriving
  the ten-file private allowlist. Relocation preserves the original manifest
  and each private-file hash; jointly edited manifest/reference bytes and a
  same-ONNX wrong manifest are rejected.
- E2 identity is checked before TensorRT/CUDA loading: SSH alias `nx`, host
  `arar-desktop`, `aarch64`, and a device-tree model containing `Jetson Xavier
  NX`, with expected and observed values recorded separately. Code provenance
  records the executing Git revision and hashes for the runner and boundary
  helpers; a caller label is not treated as provenance.
- Real execution uses separate owned child processes for build and inference,
  each with a finite positive independent deadline, its own process group,
  durable JSONL stage events, and no retry/global kill. A timeout records
  termination confirmation separately and leaves completion `unknown`; partial
  target outputs and counters remain available to the parent failure artifact.
  Adapter stage observers distinguish enqueue and D2H-copy attempted/completed
  events at their actual boundary.
- Cleanup attempts buffer, provider, stream and CUDA ownership in order even
  if one close fails. The first parse/load/inference/copy error remains primary;
  cleanup failures are attached separately. Cleanup-only failure remains
  primary.

### Verification boundary

The focused E2 workflow suite is **16/16 PASS**. It covers canonical-manifest
binding, wrong-host rejection before TensorRT import, a blocking CPU subprocess
deadline with persisted timeout event and no retry, invalid timeout values,
all-resource cleanup failure collection, parser/build/runtime contracts,
partial enqueue failure, source strict FAIL coexisting with target PASS,
output collision, and Python 3.8 AST/compile checks. `py_compile` and
`git diff --check` pass. Full repository discovery was also attempted but is
not a gate here: the local environment lacks existing `numpy`/`PIL` and one
dataset fixture, causing unrelated import/fixture failures.

### Replacement command sequence (proposed only; not run)

The old L1A-014 command that invoked the SERVER-01 Python interpreter with
`--target E2` is superseded and must not be reused. After separate review and
GO, the operator must stage and transfer only this manifest/allowlist:

    public/manifest.json
    private/onnx_export/best.onnx
    private/inputs/00006.bin
    private/inputs/00009.bin
    private/inputs/00028.bin
    private/native_reference/00006.bin
    private/native_reference/00009.bin
    private/native_reference/00028.bin
    private/onnx_reference/00006.bin
    private/onnx_reference/00009.bin
    private/onnx_reference/00028.bin

Proposed source staging and local transfer, with the already accepted source
bundle root supplied from the existing SERVER-01 inventory (no new server
authority or private-file discovery):

    SOURCE=<existing-accepted-e2l1-013-source-bundle>
    STAGE=/tmp/luna1-e2l1-015-source-bundle
    test ! -e "$STAGE" && mkdir -p "$STAGE/public" "$STAGE/private"
    cp "$SOURCE/public/manifest.json" "$STAGE/public/manifest.json"
    for f in onnx_export/best.onnx inputs/00006.bin inputs/00009.bin inputs/00028.bin native_reference/00006.bin native_reference/00009.bin native_reference/00028.bin onnx_reference/00006.bin onnx_reference/00009.bin onnx_reference/00028.bin; do mkdir -p "$STAGE/private/$(dirname "$f")"; cp "$SOURCE/private/$f" "$STAGE/private/$f"; done
    sha256sum "$STAGE/public/manifest.json"
    rsync -a --files-from=- "$STAGE/" <local-approved-transfer-root>/ <<'EOF'
    public/manifest.json
    private/onnx_export/best.onnx
    private/inputs/00006.bin
    private/inputs/00009.bin
    private/inputs/00028.bin
    private/native_reference/00006.bin
    private/native_reference/00009.bin
    private/native_reference/00028.bin
    private/onnx_reference/00006.bin
    private/onnx_reference/00009.bin
    private/onnx_reference/00028.bin
    EOF

Then transfer the reviewed code commit separately and execute only on E2 with
the existing Python 3.8 interpreter identified by the operator; the
placeholder is intentional until that existing inventory is selected:

    git archive --format=tar 1603732 scripts/edge_readiness configs/deployment | ssh nx 'mkdir -p /tmp/luna1-e2l1-015-code && tar -xf - -C /tmp/luna1-e2l1-015-code'
    ssh nx 'test "$(hostname)" = arar-desktop && test "$(uname -m)" = aarch64 && grep -a "Jetson Xavier NX" /proc/device-tree/model && test ! -e /tmp/luna1-e2l1-015-model-smoke'
    ssh nx '<E2_PYTHON_3_8> /tmp/luna1-e2l1-015-code/scripts/edge_readiness/e2_model_smoke.py --bundle-root <E2_SOURCE_BUNDLE> --manifest <E2_SOURCE_BUNDLE>/public/manifest.json --out-dir <E2_OUTPUT_ROOT> --target E2 --execute-real-device --commit 1603732 --build-timeout-seconds 900 --inference-timeout-seconds 180'

The proposed run retains the existing one-build/three-sequential-enqueue
limit, zero warmup/retry/benchmark, FP16 and 1 GiB workspace, exact frozen
float32 input bytes, output0 `[1,7,8400]` contract, and frozen source/target
comparison policies. This entry remains **NO-GO for the sequence above** until
the combined review explicitly authorizes the real E2 run.

## L1A-016 — canonical Git binding, archive provenance and production-child evidence repair

**Status:** local CPU/mock implementation and verification complete; E2 and
SERVER-01 remain NO-GO. No SSH, source transfer, TensorRT/CUDA load, engine
build/deserialization, model inference, benchmark, installation or device
configuration was performed in this entry.

Implementation commit: `6cce03e7125dd507702f168f98ec99fe0a8189c8`.
The E2L1-016 inbox is included unchanged in that commit.

### P1 — canonical source artifact

The source binding is now the raw Git blob from accepted commit
`34a542b2f787d7ef60dc3d3125cecad78e0c16f9`. The exact blob is 20,346 bytes
with SHA-256
`60744680973a73d2986bdf59ce3c6bc956119aeeebbd2106960df57947665c08`.
The production constant and regression now use this value without newline
normalization or test-time hash substitution. The original manifest bytes
remain required before deriving the ten-file allowlist; private tensor hashes
and finite float32 contracts remain enforced.

### P2 — archive-safe code provenance

The runner now supports `e2l1-code-export-v1`: a reviewed full 40-character
commit plus exact SHA-256 hashes for the runner, comparison, adapter, provider,
CUDA owner and error helper. `create_code_export_manifest()` creates it, and
`code_provenance(..., archive_manifest_path=...)` verifies executing bytes
without calling Git. This is recorded as `reviewed_archive_manifest`, not
observed `git HEAD`. A normal checkout may use `git_head`, but `--commit` uses
the exact full revision. Provenance is present in public plan, success
manifest and failure evidence; changed helper bytes are rejected before
runtime work.

### P3/P4 — child lifecycle and evidence

The parent now exercises the same child-stage contract used for production E2:
child build and inference return durable result JSON plus JSONL events,
including parser/build boundaries and per-enqueue/D2H-copy counters. Parent
failure handling reconciles result/event files, counters and private partial
output hashes on every exit. Parent deadline/termination facts are never
replaced by a child result error.

Timeout teardown is bounded at every wait/drain step. On POSIX, TERM then KILL
are sent to the owned process group/session; termination confirmation and
cleanup-unconfirmed are separate fields. No retry or unrelated-process kill is
performed. Cleanup preserves generic or structured primary errors, records all
cleanup failures, and reports cleanup-only failure as primary.

### Verification

The focused workflow suite is **21/21 PASS**. It includes the raw Git blob
regression, a real no-`.git` archive rehearsal with changed-helper rejection,
child parse/build success and parse-success/build-failure counters, parent
production-child success, timeout partial-output/counter reconciliation,
bounded timeout escalation, identity-before-runtime, cleanup aggregation,
manifest tamper rejection, and Python 3.8 compile/AST checks. The operator-
reported edge regression is **99/99 PASS**. Repository-wide discovery remains
separately environment-limited by pre-existing missing `numpy`/`PIL` and
dataset fixtures; those unrelated tests are not counted as E2 evidence.

### Package rehearsal and gate

The supported route is a scoped archive plus export manifest; no `.git`
directory is required on E2. The reviewed package revision is concrete:
`6cce03e7125dd507702f168f98ec99fe0a8189c8`. The export manifest binds the
six required helper files and is passed with `--archive-manifest`; the runner
verifies the extracted bytes before runtime work.

The exact existing E2 Python 3.8 interpreter path and accepted transferred
source-bundle root are still operator inputs. Therefore no placeholder command
is authorized for execution and no new server/device authority is inferred.
The L1A-015 source staging/transfer allowlist, FP16/1 GiB workspace,
three-enqueue, zero-warmup, zero-retry and frozen comparison policies remain
unchanged. Real E2/server transfer/build/inference stays **NO-GO** pending
combined review and those two path confirmations.

## L1A-017 — final child-failure repair, canonical package staging and conditional smoke gate

**Status:** C1/C2 local repairs and package preparation complete. The
conditional one-smoke GO was **not entered** because the authorized read-only
E2 check could not connect and the scoped private source bundle was not
available locally. No E2 build/inference/benchmark or SERVER-01 access was
performed.

Implementation commit: `b7fedfaf95bfbd9b1ee1248ec330ee5f7ac7d3f4`, pushed on
`luna1/e2l1-006-jetson-adapter-smoke` using NADUNGVN. The local focused suite
is **23/23 PASS**; the operator-reported edge regression is **104/104 PASS**.

### C1/C2 repairs

- Child cleanup/counter state is initialized before provenance and runtime
  preflight, so provenance or identity/runtime failure writes its original
  structured error JSON with zero model calls; it cannot become an
  `UnboundLocalError`.
- Event rows are reconciled even when the child result JSON is absent. Valid
  counters are merged from durable JSONL events, a truncated/invalid final row
  is retained as evidence but never treated as completion, and incomplete
  enqueue/copy/parse/build completion is marked unknown. Parser and builder
  dispatch/completion events are emitted at their actual builder boundaries;
  enqueue acceptance remains distinct from stream synchronization.
- The parent production-child path and child entrypoint are covered with
  injected CPU doubles, including three distinguishable outputs, second-stage
  failure, event-only timeout, partial output preservation, and structured
  pre-runtime failure. Generic primary errors plus cleanup failures remain
  separate, and cleanup-only failure is primary.
- Deadline teardown remains bounded through TERM/KILL escalation; leader/group
  termination confirmation and cleanup-unconfirmed are separate facts.

### Package evidence

The reviewed executable revision is the full commit above. A fresh package was
materialized outside the worktree from canonical Git blob bytes (the Windows
`tar.exe` CRLF conversion path was not used):

    D:\Research\luna1-e2l1-017-code-b7fedfaf95bfbd9b1ee1248ec330ee5f7ac7d3f4-blob

The package export manifest SHA-256 is
`aa136f72b9d395569e70944e1c67d1dbd1bdb23621912f9dfbed856297764507`.
Independent comparison of all six provenance helpers plus
`scripts/edge_readiness/__init__.py` against `git cat-file blob` for the exact
commit is **PASS**. The package contains no `.git`; execution must pass
`--archive-manifest` and the same full 40-character `--commit`.

The accepted source manifest remains the 20,346-byte raw Git blob with SHA-256
`60744680973a73d2986bdf59ce3c6bc956119aeeebbd2106960df57947665c08`. The local
working source path has only its public files; `private/` and all ten scoped
private binaries are absent, so no source package was fabricated or re-exported.

### Read-only E2 staging check and gate

I issued one bounded read-only check through the existing `nx` alias for host,
architecture, device-tree model, Python 3.8, `/tmp` disk, competing process
summary and fresh owned paths. It resolved to `192.168.10.93:22` but ended with
`Connection timed out`; therefore the observed interpreter/path/workload facts
are **unresolved**, not inferred from the inventory. No retry, install, reset,
configuration change, allocation or model operation was attempted.

The user must supply the narrowly scoped source archive from the recorded
SERVER-01 root `/home/ubuntu/Dung_TDTU/nighttime-tsd-new/results/edge_readiness_v1/e2l1-013-source-v2`, preserving the original manifest bytes and exactly the ten private allowlisted files. The user-only SERVER-01 read-only existence/hash check and packaging command remain the required next input; if that root is absent, report that exact fact. No checkpoint, raw image, credential or whole-repository transfer is allowed.

After the E2 read-only check succeeds, the source/package hashes are verified
on both sides, and the exact E2 Python 3.8 interpreter and fresh paths are
known, C4 authorizes exactly one foreground diagnostic: one FP16/1 GiB build and
exactly three ordered application enqueues (`00006`, `00009`, `00028`), with
900-second build and 180-second inference deadlines. No additional permission
round is required for that exact smoke, but this entry is currently **HOLD**
because the two concrete prerequisites above are unmet. No automatic rerun is
permitted after timeout/disconnect.

## L1A-018 — E2 connectivity restored; source archive handoff remains pending

**Status:** local package and read-only E2 staging checks are complete. The
conditional C4 smoke remains **HOLD** because the user-operated SERVER-01
source archive has not yet been supplied. No E2 build, deserialization,
inference, benchmark, allocation, installation or configuration change was
performed.

Implementation remains `b7fedfaf95bfbd9b1ee1248ec330ee5f7ac7d3f4`; report and
inbox changes are pushed in the current branch. The operator-reported edge
regression is **106/106 PASS**; local focused tests remain **23/23 PASS**.

### Connectivity result

The initial `nx` SSH timeout was reproduced. After the lab route became
available, one bounded read-only retry through the existing `nx` alias
(`192.168.10.93`) succeeded. Observed output:

    hostname: arar-desktop
    architecture: aarch64
    device-tree model: NVIDIA Jetson Xavier NX Developer Kit
    Python: /usr/bin/python3.8, Python 3.8.10
    /tmp filesystem: /dev/mmcblk0p1, 122054784 KiB total, 97123816 KiB available, 17% used
    /tmp/luna1-e2l1-017-code: absent
    /tmp/luna1-e2l1-017-source: absent
    /tmp/luna1-e2l1-017-model-smoke: absent

A bounded process-name check found no `trtexec`, `tensorrt`,
`e2_model_smoke`, `python3.8` or `cuda` process. The observed process summary
was not used as an isolation/latency claim. No route, VPN, SSH configuration
or device state was changed by Luna1.

### Package/source result

The canonical code package remains
`D:/Research/luna1-e2l1-017-code-b7fedfaf95bfbd9b1ee1248ec330ee5f7ac7d3f4-blob`.
Its export manifest SHA-256 is
`aa136f72b9d395569e70944e1c67d1dbd1bdb23621912f9dfbed856297764507`, with
7/7 raw Git file comparisons passing. The first Windows tar extraction was
discarded from the execution plan because it converted LF to CRLF; the valid
package was rebuilt directly from `git cat-file blob` bytes.

The local source path still has only public files and no private/allowlisted
source binaries. Luna1 did not access SERVER-01. The user must run the two
foreground commands already specified in E2L1-018: first verify the recorded
source root and canonical manifest SHA-256, then create exactly one
`/tmp/luna1-e2l1-017-source-v2.tar.gz` containing the manifest, ONNX and nine
reference/input binaries. The user supplies that archive and its SHA-256 via an
approved transfer location; it must not be committed or uploaded to Git.

Once supplied, Luna1 will verify archive SHA-256, exactly 11 members, no links
or traversal, raw manifest bytes and all private hashes, then transfer only the
verified scoped package to fresh absent E2 paths. C4 then permits exactly one
foreground smoke: one FP16/1 GiB build and three ordered enqueues with the
900s/180s child deadlines. Until the archive is supplied and reverified, no
smoke dispatch is authorized by the conditional gate.

## L1A-019 — HF transfer completed; local private-repo download auth remains

**Status:** The user uploaded the existing archive from SERVER-01 using the
authorized `DungJD` HF session. Local download and E2 execution remain pending;
no E2 build/inference was performed. The local HF session is still
authenticated as `Dung-trivita` and cannot see the private destination.

The upload must be run by the user on SERVER-01 using an HF session authorized
for `DungJD`. It checks the existing archive hash, creates the dataset repo
private if absent, confirms an existing repo is private, refuses to overwrite
an existing artifact path, uploads exactly the existing archive, and prints the
returned commit revision without exposing credentials:

    cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new
    python - <<'PY'
    import hashlib
    from pathlib import Path
    from huggingface_hub import HfApi, HfHubHTTPError
    repo = "DungJD/nighttime-tsd-artifacts"
    artifact = "transfers/e2l1-017/luna1-e2l1-017-source-v2.tar.gz"
    expected = "bfbd48198faad06dc45a3c1c969e14ace3f1d7c34500ef4715d3bb6997edb28b"
    local = Path("/tmp/luna1-e2l1-017-source-v2.tar.gz")
    if not local.is_file(): raise SystemExit("ARCHIVE_MISSING")
    observed = hashlib.sha256(local.read_bytes()).hexdigest()
    if observed != expected: raise SystemExit("ARCHIVE_HASH_MISMATCH:" + observed)
    api = HfApi()
    me = api.whoami()["name"]
    if me != "DungJD": raise SystemExit("HF_ACCOUNT_MISMATCH:" + me)
    try:
        info = api.repo_info(repo, repo_type="dataset")
        if getattr(info, "private", None) is not True: raise SystemExit("HF_REPO_NOT_PRIVATE")
    except HfHubHTTPError as exc:
        if getattr(exc, "response", None) is None or exc.response.status_code != 404: raise
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=False)
    if api.file_exists(repo, artifact, repo_type="dataset"):
        raise SystemExit("HF_ARTIFACT_ALREADY_EXISTS_REFUSE_OVERWRITE")
    commit = api.upload_file(path_or_fileobj=str(local), path_in_repo=artifact, repo_id=repo, repo_type="dataset", commit_message="Upload E2L1-017 source bundle")
    print("HF_REPO=" + repo)
    print("HF_COMMIT=" + commit.oid)
    print("HF_PATH=" + artifact)
    print("ARCHIVE_SHA256=" + observed)
    print("ARCHIVE_BYTES=" + str(local.stat().st_size))
    PY

After the user supplies `HF_COMMIT`, Luna1 downloads that pinned revision via
`hf_hub_download`, verifies the expected SHA-256, exactly eleven regular-file
members, no symlinks/traversal, the canonical manifest and all ten private
hashes. Only then is the verified bundle transferred to fresh E2 paths. The
existing conditional authorization remains exactly one foreground smoke:
one FP16/1 GiB build and three ordered inferences, with no retry or benchmark.

### L1A-019 receipt addendum — upload completed (2026-09-20)

The user confirmed the upload completed after authenticating SERVER-01 as
`DungJD`. Receipt: repo `DungJD/nighttime-tsd-artifacts`, dataset/private;
HF commit `0f909057a861a2b7fd887963a71c1a3f8f516007`; path
`transfers/e2l1-017/luna1-e2l1-017-source-v2.tar.gz`; archive size
`12430222` bytes; SHA-256
`bfbd48198faad06dc45a3c1c969e14ace3f1d7c34500ef4715d3bb6997edb28b`.

The local agent verified the local account is `Dung-trivita`; pinned private
repo lookup returned 404, so the archive has not yet been downloaded or
transferred to E2. No smoke was dispatched.

The local session was subsequently authenticated as `DungJD`. At pinned HF
revision `0f909057a861a2b7fd887963a71c1a3f8f516007`, the agent downloaded and
verified the archive SHA-256, all eleven approved regular-file members, the
canonical manifest SHA-256, and the ten-file private allowlist. The verified
source is staged at `D:/Research/luna1-e2l1-017-source-v2` and the archive at
`D:/Research/luna1-e2l1-017-source-v2.tar.gz`. Two bounded read-only SSH
attempts to `nx` then timed out; no E2 transfer, build, inference, or smoke
dispatch occurred.

## L1A-020 — source verified; E2 network-only hold

**Status:** Complete for the current network-held scope; no E2 execution was
performed. The pinned private HF revision
`0f909057a861a2b7fd887963a71c1a3f8f516007` was downloaded under `DungJD` and
the archive was verified locally at
`D:/Research/luna1-e2l1-017-source-v2.tar.gz`: size `12430222` bytes, exactly
11 approved regular-file members, canonical manifest SHA-256
`60744680973a73d2986bdf59ce3c6bc956119aeeebbd2106960df57947665c08`, and all
10 private binary hashes matched the manifest. The corrected archive SHA-256
is `bfbd48198faad06dc45a3c1c969e14ace3f1d7c34500ef4715d3bb6997edb28b`.

Two bounded read-only SSH attempts to the existing `nx` alias timed out before
E2 path checks/transfer. This is a network prerequisite, not a failed smoke;
no E2 transfer, build, inference, retry, warmup, benchmark, or extra device
was run. When VPN/lab connectivity is reported restored, perform one bounded
read-only identity/runtime/resource check, transfer the already verified code
and source packages to fresh E2 paths, recheck their hashes, and use the
existing E2L1-017 conditional authorization for exactly one FP16/1 GiB build
and three ordered inferences. No new approval is needed for that continuation.

## L1A-021 — read-only audit of consumed E2 build timeout

**Status:** Diagnosis complete; no second build or inference was run. A single
read-only audit inspected the unchanged owned run
`/tmp/luna1-e2l1-017-model-smoke` on E2 at UTC
`2026-09-21T16:12:52.169931+00:00`.

Observed: no matching old-run process remained; the engine path and
`build_result.json` were absent; and no engine was deserialized. The retained
`build_events.jsonl` is 1,997 bytes with SHA-256
`1b2bad411c45b001086ccfcffd462d811fd85575f3ac9121994622a38accff81`. It
records parse 1/1, build 1/0, engine load/enqueue/copy/compare 0/0, TERM sent,
KILL not sent, child return code `-15`, and confirmed termination. Attempt-time
timestamps were unavailable; file mtimes are artifact-write times only.

The current snapshot showed 4,928,136 kB available RAM, 3,509,072 kB free
swap, approximately 99.4 GB free on `/tmp`, and load average `0.29 0.11
0.03`. `nvpmodel -q` reported `MODE_20W_6CORE` (index 8) without changing
state; its read-only query also reported permission errors for some EMC/VDDIN
limit paths. A bounded 10-second `tegrastats` sample showed no current GPU
activity and temperatures around 30 C. These are current observations, not
reconstructed build-time conditions. `journalctl -k` exposed recent/current
throttle-device lines and `NVRM: No NVIDIA GPU found`, but no timestamp linkage
to the build; `dmesg` was unavailable due to permission denial.

Evidence does not establish whether the 900s deadline was insufficient, the
runtime/driver stalled, or a resource condition occurred during the build. No
OOM, thermal failure, or workspace insufficiency is claimed. The full public
diagnostic is at
`results/edge_readiness_v1/e2l1-021-timeout-diagnostic/`; binary/tensor bytes
were not published and the old private run was left unchanged.

**Recommendation:** because no completed engine or actionable error exists,
propose (not authorize) one future fresh-root diagnostic with unchanged
source/runtime/FP16/1 GiB/input contract, durable builder logs, and candidate
build ceiling `3600s`. It must be separately reviewed, counted alongside the
consumed attempt, and must never resume or overwrite the old root. E2L1-021
does not run it.

## L1A-022 — implementation_verified milestone

The E2L1-022 observability/lifecycle repair is locally verified. The parent
stage boundary now binds durable per-stage stdout/stderr paths, SHA-256/tail
metadata, UTC and monotonic event timestamps, and additive timeout details;
the child emits low-rate heartbeat events while a stage is live and requests
TensorRT INFO logging with an explicit legacy ERROR fallback. Existing
unknown-completion and partial-counter semantics remain unchanged.

Focused durable-log/timeout/child-boundary tests pass, and the full edge
regression is **108/108 PASS**. No local GPU, TensorRT execution, E2 build or
inference occurred during this milestone. Next milestones are canonical
package creation, E2 transfer/hash recheck and the single conditional attempt
authorized by E2L1-022.

### L1A-020 execution addendum — one E2 smoke consumed by build timeout (2026-09-21)

The bounded read-only E2 gate passed: `arar-desktop`, `aarch64`, NVIDIA Jetson
Xavier NX Developer Kit, Python 3.8.10, expected fresh paths, approximately
97 GB available on `/tmp`, and no relevant workload process. The verified code
and source packages were transferred to `/tmp/luna1-e2l1-017-code` and
`/tmp/luna1-e2l1-017-source`; the E2 recheck matched the source manifest,
all ten private binaries, code commit `b7fedfaf95bfbd9b1ee1248ec330ee5f7ac7d3f4`,
and code-export manifest SHA-256
`aa136f72b9d395569e70944e1c67d1dbd1bdb23621912f9dfbed856297764507`.

The single authorized foreground smoke was dispatched with build deadline
900s, inference deadline 180s, FP16 and 1 GiB workspace, zero warmup/retry/
benchmark, and input order `00006`, `00009`, `00028`. It ended with
`status=failed`, `error=STAGE_TIMEOUT`, `stage=build`, and
`execution_state=unknown`; termination was confirmed. Counters: parse
`1/1`, build `1/0`, engine load `0/0`, enqueues `0/0`, output copies `0/0`,
comparisons `0/0`; `unknown_completions=["build_completion"]`. No inference
was attempted and no retry is authorized. The source strict diagnostic was
preserved separately (`00006`/`00009` pass, `00028` has one mismatch); this is
not an E2 target numerical result because target inference never started.

Public evidence is retained at
`results/edge_readiness_v1/e2l1-017-model-smoke-public/`; private engine/build
partials remain on E2. Public artifact hashes are: `failure.json`
`9962a13e6c1b336fabca8267b8e9a1ed5ead0c1c3350e2a6a5fa9664244ab3ed`,
`index.json` `0a0b4e3d1342e11a87eb8dfc40fbfa18532a48a319665cb1ada24bd333dfd6a9`,
`plan.json` `050e3e9b4408b12c0e503bdd1833ee9ea6bd165bf212d1ffb757667d25f19a5d`,
and `run.log` `d8215558f3bfeadd117e42f22cdb7802263d3a62131825d7905369c96ac3a346`.

### L1A-022 execution milestones — one instrumented E2 attempt completed (2026-09-22)

**staged.** The local observability repair and CPU/mock regression were
verified before staging: focused durable-log tests passed and the full edge
regression was `108/108 PASS`. The executable package was built from canonical
Git blobs of commit `f56c5262943df6c1d436adcc11d22422b47d89c0`; its seven-file
code manifest was verified on E2 with SHA-256
`89b91cf51aedfe83378feca2d5726a506a466d980e5c2733baaaf97486e80778`.
The accepted source was rechecked against manifest
`60744680973a73d2986bdf59ce3c6bc956119aeeebbd2106960df57947665c08` and ONNX
`bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7`.

**running.** The fresh E2 roots were `/tmp/luna1-e2l1-022-code` and
`/tmp/luna1-e2l1-022-model-smoke`; attempt1 at
`/tmp/luna1-e2l1-017-model-smoke` was not modified. The read-only identity
gate remained `nx`/`arar-desktop`, `aarch64`, NVIDIA Jetson Xavier NX
Developer Kit, TensorRT `8.5.2.2`, Python `3.8.10`. Exactly one foreground
attempt was dispatched with one FP16 builder, `1073741824` workspace bytes,
build ceiling `3600s`, inference ceiling `180s`, zero warmup/retry/benchmark,
and ordered fixtures `00006`, `00009`, `00028`. No third attempt or benchmark
was run.

**artifact_audited.** Attempt2 parsed and built successfully, loaded the
engine, and completed all three ordered enqueues and synchronized output
copies. Counters were parse `1/1`, build `1/1`, engine load `1/1`, enqueues
`3/3` with synchronization `3/3`, output copies `3/3` with synchronization
`3/3`, and comparisons `3/3`; no unknown completions remained. The private
engine is 7,158,097 bytes, SHA-256
`581a9ea2eafdae690f25f57ab88ba1bcffdafa99e5322ea50ee43678520e7f56`.
The build event window was approximately 16m35s and the inference event
window approximately 13.3s; these are stage wall times, not benchmark or
inference-latency claims. Durable evidence hashes are:

| Evidence | Bytes | SHA-256 |
| --- | ---: | --- |
| `private/engine/build_events.jsonl` | 100950 | `93074ef910d64cbcdba864807f14ae8a356ce65517c9744cc9ccc1927ffbf2f2` |
| `private/engine/build_stdout.log` | 24143 | `494d324db36d900857824e4831396f55844bcab5bbd5244e8865a1f757eb285d` |
| `private/inference_events.jsonl` | 15232 | `28dfdb6814f57498584a1d409b5de81d2c79af4e7fbbe2faa39b14000c8a5dc1` |
| `private/inference_result.json` | 1296 | `c5e019fbba9b3907e141666dae77875fc2cbf530dfc920027c10f4e4a570c00d` |

The build log contains TensorRT warnings about tactics requiring more than
the available workspace and FP16 subnormal weights; these are observed log
facts, not a causal diagnosis. A post-run read-only snapshot found no matching
smoke process, `MODE_20W_6CORE` (index 8), about 4.7 GiB available memory,
3.3 GiB unused swap, and 93 GiB free on `/tmp`. The snapshot is not evidence
of resource conditions throughout the build. No binary engine or raw target
tensor was published to Git.

The public handoff is
`results/edge_readiness_v1/e2l1-022-model-smoke-public/` with five JSON/text
files. Their local hashes are: `manifest.json`
`29954570c5b1f44af928e283ce2ab89885c224b165ab85b4f69de6d5ae6a4e74`,
`plan.json` `85cae7fea27e7dce65bb9ef39292db0dc62cab14920c24221b3dfeb15843b19e`,
`index.json` `d1c99f0f3bba11461bc2e3f1390116a1a4d10f688d8159599ef96a583e0d9cd7`,
`report.md` `b010475ec19ae2846a332ac22fffc0e5143c08d81b6522cc6b43c7e405249990`,
and `run.log`
`832cfaaf3e189a599cdefee109a4f7a92c9210a0db1c768ca82fcb6378a94f78`.

**Comparison and interpretation.** The source/export discrepancy is kept
separate from target comparison: source native versus source ONNX passed for
`00006` and `00009`, while `00028` retained the known single box mismatch.
TensorRT versus source ONNX/native both failed for every target fixture:

| Fixture | TRT vs ONNX mismatches / max abs | TRT vs native mismatches / max abs | Target output SHA-256 |
| --- | ---: | ---: | --- |
| `00006` | 965 / 4.2297821045 | 966 / 4.2314453125 | `d22fdb65d0affaae165c527c2f7d067e989911ae3f42391ada6387aed596bd9e` |
| `00009` | 1226 / 1.3426208496 | 1226 / 1.3427810669 | `d67bba2dbcc840a09f3eca943352f653f15160caefb3ff6d7c350236d7837320` |
| `00028` | 564 / 0.9544372559 | 564 / 0.9544982910 | `cab60f96c43af4835493d90b73052334e121b2b285be41216a088c73347b9a3c` |

The terminal result is therefore
`edge_smoke_completed_review_required`, specifically
`execution_complete_target_mismatch`. This establishes a real FP16 TensorRT
build/load/copy path and a reproducible target discrepancy; it does not
establish accuracy, latency, energy, deployment readiness, or a root cause.
The next bounded scientific decision is to review the FP16 target/reference
contract and the recorded workspace/tactic warnings, then define an explicit
FP16-aware acceptance diagnostic before any future edge run; no tolerance was
relaxed and no rerun was made here.

## L1A-023 — saved-output CPU diagnosis completed (2026-09-22)

**acquisition.** No E2 build, inference, benchmark, device change or engine
download was performed. The three already-existing target tensors were copied
read-only from E2 into private local staging outside the repository. Each was
exactly `235200` bytes and matched the accepted manifest: `00006`
`d22fdb65d0affaae165c527c2f7d067e989911ae3f42391ada6387aed596bd9e`, `00009`
`d67bba2dbcc840a09f3eca943352f653f15160caefb3ff6d7c350236d7837320`, and
`00028`
`cab60f96c43af4835493d90b73052334e121b2b285be41216a088c73347b9a3c`. The
retained build/inference result, event and stdout/stderr files also matched
the previously recorded sizes and hashes. Engine and raw tensors remain
private; none were added to Git.

**implementation_verified.** Added the dependency-free CPU analyzer
`scripts/edge_readiness/e2_output_diagnostic.py` and five focused tests. It
rejects wrong size/hash, replays the existing raw comparator, reports
channel/anchor strata and threshold/class crossings, uses immutable
little-endian float32 reads, and applies deterministic class-aware single-
label NMS at `conf=0.001`, `IoU=0.7`, `max_det=300` in 640x640 input pixels.
Same-origin lineage is preferred; supplementary matching is deterministic,
one-to-one and descriptive only. Analyzer tests pass `5/5`; the existing edge
regression remains `108/108 PASS`. A broader repository discovery was not a
clean gate because this checkout lacks unrelated NumPy/data prerequisites;
that does not affect the CPU analyzer or edge suite.

**analyzed.** Raw replay preserved the accepted verdicts: source native versus
ONNX passed for `00006` and `00009`, and retained the one known `00028` source
box mismatch. TRT versus source ONNX had box/score mismatch counts `965/0`,
`1226/0`, and `564/0` for `00006`, `00009`, and `00028`; score `0` means no
policy violation, not exact score equality. At source-ONNX max-score strata
`0.001` and `0.25`, source/target selected-anchor membership was unchanged for
all three fixtures. The selected-stratum box mismatch counts were respectively
`0`, `5`, and `0`.

CPU postprocess retained the same candidate and NMS counts: `00006` source
`10 -> 1`, target `10 -> 1`; `00009` source `16 -> 2`, target `16 -> 2`;
`00028` source `20 -> 2`, target `20 -> 2`. All kept detections had the same
anchor/class lineage. Nevertheless, same-lineage box coordinates differed;
the maximum per-detection coordinate deltas were `0.0971832`, `0.212311`, and
`0.263626` pixels. The corresponding per-fixture IoU ranges were
`0.992440–0.992440`, `0.979394–0.987351`, and `0.993441–0.994679`; the
maximum coordinate delta and IoU extrema are not asserted to belong to the same
detection. These observations do not establish final-detection or ground-truth
accuracy impact. No AP, recall, safety or deployment claim is made from three
train fixtures.

Public scoped evidence is at
`results/edge_readiness_v1/e2l1-023-output-diagnostic/`:

The analyzer/test source bindings are
`e2_output_diagnostic.py` SHA-256
`06efff09adca9543fa4e4f34be303783bd92d94b2da4e3522aa5bb187858f702` and
`test_e2_output_diagnostic.py` SHA-256
`e6f03893df06917f2375b353b06b94ccb6cd837b206eeea3b770e38390cd75d4`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `analysis.json` | 137627 | `b9df6015385a5595997e1eefdc5253553638bd2a39237ac22c1163cd7c8c2058` |
| `inventory.json` | 3579 | `7977637e8abfc0e1c248689ccd8d9c7eda14f3b1a87b9d95f718a2c74d92cd79` |
| `report.md` | 1586 | `39fb049f8fda17b92cf7406dd30eee39f8743d4c7b8ae20213baa7b152e1c704` |
| `index.json` | 276 | `42d9a9ac53b42081cca26502371890b4462a6ed14369d2a3e4634f1334d9a3b2` |

**terminal.** `output_diagnostic_completed_review_required`. The evidence
supports an observed raw box discrepancy with unchanged score-threshold
membership and unchanged post-NMS count/lineage, but it cannot determine
accuracy impact. One proposed next experiment is retained for scientific
review only: a separately authorized FP16-reference/target postprocess
comparison on a predeclared evaluation slice. No third build, new inference,
benchmark or tolerance relaxation was performed.

## L1A-024 — saved-output closure and prospective dev packet (2026-09-22)

**implementation_verified.** The E2L1-023 analyzer was tightened to accept
source expectations only from the pinned source manifest
`60744680973a73d2986bdf59ce3c6bc956119aeeebbd2106960df57947665c08`, verify
the target public manifest
`29954570c5b1f44af928e283ce2ab89885c224b165ab85b4f69de6d5ae6a4e74`, bind all
eight retained log hashes, and recheck all three input hashes before/after
analysis. The accepted postprocess binding is Ultralytics `8.4.102`,
`ultralytics.utils.ops.non_max_suppression`, CPU, `return_idxs=True`; the local
dependency-free helper remains explicitly marked as a descriptive fallback,
not exact helper equivalence. Tests cover positive-area same-class suppression,
different-class retention, confidence/IoU/max-det boundaries, tie handling,
immutability, nonfinite/shape/hash rejection, wrong log hash, timeout partial
publication and cleanup failure. Focused analyzer tests pass `7/7`; the edge
regression remains `108/108 PASS`.
The analyzer/test source hashes are respectively
`e0598851b19a28acd3140b77b7f9100c2456fcd39c9c3bc0ed1ce12f20441a11` and
`ace3ef994c9c1b545cf424972e637a6b079084ec9c3f36923a387068adb14dcc`; the
packet/test hashes are
`d37923feb5b44a2941107248b00b35cea7b003075d4073db78b573b90779fddd` and
`87637ce6928602da9a2d40b19eb2bcf34ec8540962c19d04a626a0cf8973fa68`.

**analyzed.** A fresh audit root was generated without touching E2 or the
published L1A-023 v1 directory. It retains all five same-origin rows and
correctly reports separate extrema: for `00006`, `00009`, `00028`, maximum box
coordinate deltas are `0.0971832`, `0.212311`, `0.263626` input pixels, while
IoU ranges are `0.992440–0.992440`, `0.979394–0.987351`,
`0.993441–0.994679`. The independent accepted-helper review found custom
float32 conversion differences up to `1.52587890625e-05` pixels. This is
replay evidence only; raw FAILs remain and no box correction, tolerance change,
new inference or accuracy claim was made.

**packet_ready.** The prospective packet is
`results/edge_readiness_v1/e2l1-024-dev-packet/`. It freezes the existing
engine hash `581a9ea2eafdae690f25f57ab88ba1bcffdafa99e5322ea50ee43678520e7f56`
and source ONNX identity, exactly 1636 dev-image target passes and 2706
instances, zero builder/warmup/probe/retry calls, and reference calls either
zero when reusable predictions are identity-bound or 1636 in a separately
counted CPU reference capture. It specifies paired COCO/XML AP50/AP50-95,
XS/S and class/coordinate diagnostics with the accepted 1000-resample paired
image bootstrap (`seed=20260916`), without inventing a noninferiority margin.
The runbook divides server CPU materialization, Luna1 E2 execution and Astra
review; public artifacts contain hashes/metadata only, while engine/ONNX/input
and raw predictions remain private. The CPU/mock packet tests pass `7/7`,
including successful synthetic artifact-to-analysis, package/hash/dataset
guards, output shape, partial timeout, no-retry and cleanup failure behavior.

Public audit hashes are:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `e2l1-024-saved-output-audit/analysis.json` | 142251 | `0eedd680c2dd1c14426bfdcfcb67273341ae10e405dcae7c77d33ee380adc053` |
| `e2l1-024-saved-output-audit/inventory.json` | 4563 | `634574dcb9fb7576fa6e3277f5d7e52d2eb60b6514020d6c50d8231de74f419b` |
| `e2l1-024-saved-output-audit/report.md` | 2324 | `3c6c2b374ebf856393376b0d48f100d5daa1e047dc80b1cbb7d6293b28c9d8d0` |
| `e2l1-024-saved-output-audit/index.json` | 276 | `42d9a9ac53b42081cca26502371890b4462a6ed14369d2a3e4634f1334d9a3b2` |
| `e2l1-024-dev-packet/contract.json` | 3926 | `0df5a3a2874762bc901f22d41a93af0162f664dc597f8a80cbc2bae90e077978` |
| `e2l1-024-dev-packet/runbook.md` | 2426 | `589b1cd7a5a632e5780ded9566ed639d868a64dc79ff7de6f5bbcb0835523862` |
| `e2l1-024-dev-packet/index.json` | 243 | `96a463c275cfaeab8d0aaf809131516f443ff4ac9329becd8352d129bf9e2f9e` |

**terminal.** `edge_dev_packet_implementation_review_required`. E2L1-024
does not authorize the proposed development study; later execution remains
NO-GO pending integrated review of source identity, dev membership/XML,
private package, engine availability and call accounting. No SSH, transfer,
source forward, export, build, E2 inference or benchmark was performed in
this packet.

## L1A-025 — ST-EDGE-03 integrated review repair (2026-09-22)

**implementation_verified.** E2L1-025 repairs the gaps identified in
`docs/ST_EDGE_03_INTEGRATED_REVIEW_R1.md` without touching E2. The dev packet
now uses contract schema v2 and compares every frozen identity/runtime/budget
field, including the canonical engine hash, `warmup=0`, zero probes/debug/
retries/builds, the canonical `ultralytics.utils.nms.non_max_suppression`
binding, and streaming input storage. Package validation checks complete
1636-ID ordered membership, shape/dtype/little-endian/finiteness, source/
reference/XML identity, and actual declared file path/size/SHA256, symlink and
undeclared-file policy. The engine remains E2-private and is not loaded locally.

The CPU/mock workflow now has AP50/AP50-95 synthetic evaluation with nonzero
ground truth and paired duplicate-image bootstrap CI, plus an external child
process with append+flush+fsync events, bounded timeout, durable partial
counters, unknown completion state and no retry. The runtime boundary records
the existing `TensorRTProvider` / `CudaRuntimeMemoryOwner` /
`JetsonRuntimeAdapter` ownership; no new device framework or model call was
introduced. Saved-output diagnosis binds the corrected NMS module through an
executable resolver, compares a copied CPU tensor, hashes input after fixture
processing, and records native/ONNX/TRT output bytes and hashes separately
from input audit.

**analyzed.** Focused packet and diagnostic tests pass `18/18`, including wrong
engine hash, nonzero warmup, wrong reference count, actual file hash,
unexpected file, AP/CI, durable child timeout, corrected helper binding and
the prior shape/hash/cleanup regressions. A fresh public packet root was
generated at `results/edge_readiness_v1/e2l1-025-dev-packet-r1/`; it contains
only `contract.json`, `runbook.md` and `index.json`. Prior E2L1-024 roots
remain unchanged. No build, inference, benchmark, SSH, transfer or device
configuration was performed.

**packet_ready.** The local implementation is ready for later integrated
review of the private package and owner-side E2 preflight. This is not an E2
GO: the packet remains `prepared_not_executed`, and the canonical XML/AP
evaluator still belongs to the future approved workflow.

**terminal.** `edge_dev_packet_implementation_review_required`.

R1 public packet hashes: `contract.json` 4165 bytes /
`60a2bacc0b8828d71322572485d9ceb75a35a51d0a336e4643392f63d836868a`;
`runbook.md` 2003 bytes /
`b9402fe631d323c1e2020d124540219e0659455774ad8806f1e8aa1290174309`;
`index.json` 609 bytes /
`f1a56d68398bb4ff810a9c564752b1fce87b1cd19df144a77cb67fe4f83a075d`.

## L1A-026 — ST-EDGE-03 R2 provenance and executable pipeline (2026-09-22)

**implementation_verified_local.** R2 corrects the provenance defect: hash
`5c23d0fa7d4bbf09858b2f1a4dbf45c35650c474aaedebe40d845e3c9acc410a` is now
named only as the SERVER TensorRT FP16 reference. It is explicitly not source
ONNX CPU evidence. Source CPU reference generation is represented by
`e2l1-026-source-reference-pending/source_reference.json`, with planned 1,636
passes and executed `0`; no source model forward occurred.

The accepted NMS adapter now creates a copied CPU float32 tensor of shape
`[1,7,8400]`, preserves and rechecks input bytes, calls the exact
`ultralytics.utils.nms.non_max_suppression` symbol with `return_idxs=True`,
and records detections/output hashes. The production evaluator path delegates
to the repository's locked `coco_xml_paired_image_bootstrap_v1` implementation
(`pycocotools==2.0.10`, PCG64 seed `20260916`, 1,000 paired sorted image
draws, duplicate occurrences retained) and returns source, target and
target-minus-source AP50/AP50-95 across all/xs/s/m/l/xl. With local
dependencies absent (`numpy`, `torch`, `ultralytics`, `pycocotools`), this
canonical path remains dependency-pending rather than producing substitute
metrics.

The child path now constructs and uses the existing `JetsonRuntimeAdapter`
with `MockBuffers`/`MockContext`/`MockStream`; the future real factory remains
lazy through `TensorRTProvider`, and no TensorRT import/load is performed.
Streaming input records have bounded shape/dtype/hash/finiteness checks and
are consumed one tensor at a time. Durable child events are still append,
flush and fsync; timeout/cleanup/unknown completion remain explicit.

**analyzed.** Focused R2 packet/diagnostic tests pass `21/21`; they cover
source-reference pending status, canonical-image binding, streaming producer,
actual adapter lifecycle through the existing boundary, helper tensor shape
and immutability, empty AP behavior, strict contract/package negatives and
durable timeout. Historical saved-output FAILs and E2L1-024/R1 evidence remain
unchanged. Fresh public roots are
`results/edge_readiness_v1/e2l1-026-dev-packet-r2/` and
`results/edge_readiness_v1/e2l1-026-source-reference-pending/`; only JSON/
Markdown metadata is public.

R2 public artifact hashes are: `contract.json` 4823 bytes /
`313bb7632d056fbafe192663c16bd6723c493fd96bf8f24c92417cdce09b71e8`;
`index.json` 770 bytes /
`b37e8910b4880d6317ee8b0867640358227f771dd9f4f826e4d0972d4554f7ee`;
`runbook.md` 2779 bytes /
`61c9025c5abdcfe875e9725c210536b67f93ac45ade858ac9b919a810006c382`;
`source_reference.json` 549 bytes /
`cc727258578e9b082efc972871051b7d0e494a8d953ec8de49569b79c2e3091e`.

**packet_ready.** The local executable boundaries and prospective commands
are ready for review, but this is not an E2 GO. E2 execution remains
`prepared_not_executed`; source CPU reference production and canonical
COCO/XML evaluation must occur later in their approved dependency environment
and be separately counted.

**terminal.** `edge_dev_packet_canonical_reference_pending`.
