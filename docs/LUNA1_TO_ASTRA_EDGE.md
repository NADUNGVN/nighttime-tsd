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
