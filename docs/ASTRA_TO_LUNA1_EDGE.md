# Astra to Luna 1 — Edge deployment lane

## E2L1-001 — GO SSH read-only inventory and local harness preparation

### Context and ownership

Paper: Edge INT8 Traffic-Sign Detection, CCTSDB2021 three classes. Fifteen trained FP32 checkpoints are frozen; no retraining or external training datasets. Original VCSC superiority hypothesis was not supported. Current question: accuracy loss and detection-head precision sensitivity, versus deployment cost. YOLO11n dev diagnostics found improvements from head FP32 protection, but this is not yet cross-model or edge evidence; do not advertise universal speedup or accuracy guarantees.

Main Luna is implementing graph preparation for YOLOv8n and YOLO26n confirmation (3 Uniform selections, 4 INT8 arms, 3 builds plus FP16). This is independent of edge inventory. Main Luna cannot SSH into servers; the user is server operator. **Luna 1 may SSH into the user's designated edge devices for the bounded read-only task below**, because user has confirmed they are ready and accessible. Do not use or discover unrelated hosts/accounts.

### Access source supplied by the user

Read `D:/Research/Teacher_Vu/docs/SHARED_INFRASTRUCTURE.md`, especially Sections3.1/3.2, for existing SSH user/host mappings. This is a local shared-infrastructure reference, not a file to copy into this paper repository. Do not ask the user to repeat endpoints already documented. Use existing authorized keys/config; ask only for missing key/alias access details or workspace permission, never private key/password contents. The recorded addresses/versions are historical observations, not live reachability evidence. Do not disable SSH host-key verification.

Inventory targets mapped from that reference:

| Infra ID | Device / access section | Role in current task |
|---|---|---|
| E1 | Raspberry Pi5 + Hailo-8; `urlab`, Tailscale endpoint in E1 table | GO read-only inventory. CPU-only and Hailo may be two runtime conditions on this same physical board, not two independent devices. No accelerator disabling or hardware removal needed for inventory. |
| E2 | Xavier NX; `arar`, LAN endpoint in E2 table | GO read-only inventory. |
| E3 | AGX Xavier; `arar`, LAN endpoint in E3 table | GO read-only inventory; verify exact RAM/SKU instead of assuming32/64GB developer kit. |
| E5 | Orin Nano Super; `huy`, Tailscale endpoint in E5 table | GO read-only inventory as available candidate hardware; **not legacy Jetson Nano** and not silently substituted into paper scope. Await user/reviewer choice before scored study. |
| Legacy Nano | Not identified in supplied infrastructure file | Missing mapping, not a failed SSH host. Do not scan networks to find it; ask operator whether present. |
| Separate Pi5 CPU board | Not identified separately | Record unknown; may reuse E1 CPU condition after protocol approval, do not claim a fifth independent physical board. |

E4 RUBIK Pi3/Qualcomm is documented for another project but **out of this task's scope**; do not add it automatically. Existing segmentation/audio compiler smoke results in the source document are not evidence for CCTSDB YOLO model compatibility or measured performance. Do not modify those projects' environment/output directories or reuse their engines/calibration as ours. Existing server SSH information does not change main Luna's operator-mediated server workflow.

### GO now

1. Inspect existing edge/benchmark code in repo before creating another implementation. Use a dedicated branch and separate worktree/clone if agents share filesystem. Proposed owned paths: `scripts/edge_readiness/`, `tests/test_edge_readiness.py`, `docs/EDGE_READINESS_V1.md`, `docs/LUNA1_TO_ASTRA_EDGE.md`, `results/edge_readiness_v1/<device_id>/<run_id>/`. Coordinate reuse without editing main Luna's confirmation files. No broad staging or branch switching in another agent's working tree.
2. Create a small schema-versioned read-only collector and CPU fixture tests; reuse scripts when suitable. Run inventory via SSH once local tests establish commands have no mutating flags. No need to wait for source-model confirmation. Record actual commands, exit codes, timestamp, code hash and unknown/unavailable/permission-denied states rather than fabricating compatibility.
3. Collect available hardware model, RAM, CPU architecture, GPU/accelerator identity, OS/kernel, installed JetPack/L4T/TensorRT/CUDA/driver/Python/runtime package versions as relevant, disk availability, existing power mode/clocks, idle thermal state, cooling/power-supply description (operator supplied where unobservable). No assumption a device must support server TensorRT10.16.1.11. Missing tool is a reportable prerequisite, not authorization to install.
4. Check availability of existing telemetry sources and power measurement: describe onboard/software/external meter, measurement boundary (chip/module/whole device), units, sampling limitations. A short bounded read-only sample is allowed; no stress test, inference loop, clock/power-mode change, or calibration. Do not equate nominal TOPS/TDP with measured power or throughput.
5. Produce compatibility matrix separating observed installed capability from documentation/inference/unverified export support. For version-specific support conclusions consult official vendor documentation and record source/version; do not decide compatibility from model name alone. Record likely risks for native YOLO26 end2end output, precision constraints, operator coverage, and backend conversion without forcing unsupported paths.
6. Draft a benchmark harness interface/protocol and CPU mocked tests only: future manifest binds source-weight hash, calibration IDs/hash/order, device/backend-native quantization, runtime versions, preprocessing/postprocessing, batch/input shape, warmup/measured counts, latency boundary, synchronization, percentiles, memory, measured-power boundary and thermals. Mark counts/backend options as proposed until reviewed; do not silently copy server timings into edge results.

### NO-GO without separate review

- No training, model download, ONNX export, TensorRT build, Hailo compilation/calibration, scored inference or benchmark matrix.
- No apt/pip/firmware/OS changes, sudo mutation, reboot, service stop, process kill/pause, power/clock/fan changes or automatic dependency repair.
- No source-weight/calibration resampling, policy selection, official-test-guided tuning, or changes to main confirmation protocol.
- Do not copy RTX8000 engine to an edge and treat it as a portable model. Cross-backend comparison means **same frozen source weights + same calibration image IDs where logically applicable + backend-native quantization**, not numerically identical INT8 engines. CPU FP32/other unsupported precision can be legitimate reference; never relabel mixed precision as all-INT8.

Missing privileges/devices should block only that device's affected check. Continue read-only checks on other authorized devices and document limitations. Do not wait for all mapped devices or unresolved legacy-Nano mapping before reporting the first useful inventory; do not expand access to make them pass.

### Deliverables and gates

- Device inventories JSON + concise report, machine-readable compatibility matrix, local tests, proposed measurement protocol and dependency/action list with scope/risk. New output directories, no overwrite. Keep host credentials/private network addresses out of committed public artifacts; use stable device aliases and retain hardware identity bindings in the agreed private artifact scope if sensitive.
- Append **L1A-001** to `docs/LUNA1_TO_ASTRA_EDGE.md` with actual contacted device aliases, evidence, incomplete checks, local tests, commands and Git commit. Preserve this instruction file and commit it with scoped work. Push through authorized Git account; tell Astra which branch/commit, do not merge main Luna changes blindly or force-push.
- Astra reviews inventory and proposed setup before authorizing installations/conversion/smoke tests. **GO inventory/preparation now; NO-GO scored edge study.** Final representative model/arm/backend selection waits for confirmation review, but setup planning need not wait.

Completion is honest device readiness evidence, not successful deployment or paper performance results. These are parallel work packages, not permission to bypass scientific gates.

## E2L1-002 — accept inventory snapshot; GO local measurement harness/protocol

Reviewed branch `luna1/e2l1-001-edge-readiness`, commit `5d407f1a9c285d274c9009ac99511afb539817c4`, L1A-001. Astra inspected 6 canonical JSON artifacts (4 device inventories + compatibility/protocol), each device records 24 commands and SSH transport exit 0. Collector hash in all 4 inventories matches canonical collector source `b9abe9272cae4c015d6094248e559f854f417840b9ac1abe877745b6ed87b212`. Full combined suite 166/166 passed, including 5 edge tests. This confirms recorded provenance/structure, not independent re-SSH execution; unknown/permission-denied checks remain unknown. Accept as **initial live inventory**, not YOLO deployment support or performance validation.

Observed useful constraints: E1 root free 4.9GB (83% used), E2 free 94GB, E3 free 11GB, E5 free 792GB. E5 is Orin Nano Super, not legacy Nano. No cleanup/install/clock modification authorized. Existing dependency and board observations can guide planning, but documentation-derived JetPack labels remain separate from observed L4T/package versions. External power meter/cooling/supply boundary still unresolved; no actual telemetry sample recorded. Do not treat inventory temperatures as controlled idle without workload evidence.

### Next independent task (does not wait for Luna graph)

1. Draft concrete measurement protocol and implement backend-neutral **local CPU/mock harness**, no actual device inference/build. Reuse existing project latency/statistics helpers where appropriate; preserve old accepted results. Include explicit inference-only versus decoded-image-to-detections boundaries, synchronization responsibility per backend, preprocessing/postprocessing, batch 1/input 640, warmup/measurement/session design, device-native model/hash bindings and negative/error states. Propose 200 warmup/1000 measured samples × 3 sessions as starting design consistent with prior study, clearly pending review and thermal/power sampling suitability; preserve per-sample/session data, distinguish FPS reciprocal latency versus measured pipelined throughput, no pooled 39k-independent-samples claim. Cross-device parser implementations/quantization need future correctness checks, not just timing.
2. Define measured power/energy integration with time-aligned samples and actual boundary; report unavailable rather than infer watts from TDP/TOPS. Memory metrics backend-specific and unit-labeled. Thermal/throttle and current power-mode observations before/after; do not prescribe max clocks or change modes without operator approval. Physical E1 CPU versus Hailo conditions may be proposed on same board, not two independent devices; no E5 scored scope substitution without confirmation.
3. Publish a setup action matrix for E1/E2/E3/E5: current observed prerequisites, missing packages/tools, proposed actions, disk budget, offline/compiler host needs, permission/risk. All installs/conversions remain proposed only. Prioritize E2/E3 GPU-native path and E1 CPU/Hailo feasibility planning; do not add DLA/Qualcomm or all 15 models. This is prioritization for planning, not final model/arm selection.
4. Harden collector for future use without rerunning all devices: validate SSH alias as safe non-option alias (current check permits leading `-`), reject duplicate/unexpected/missing command markers before declaring complete; keep transport success separate from inventory completeness. Preserve timeout partial evidence; `trtexec --help | head` exit 0 currently reports wrapper success, not reliable executable exit code—document existing record limitation and fix future collection if code touched. Use exclusive no-overwrite writer, source/script hash and tests. None of these findings alone invalidates the observed 24-label snapshots.
5. Append **L1A-002** with files/tests/protocol/options and outstanding operator questions. Push scoped branch changes, keep main-lane files unchanged. Don't open/merge PR automatically; supplied `/pull/new/...` is only creation URL.

### Concurrency safeguard

Astra found shared cwd on edge branch after Luna main work. Use a separate worktree/clone for each lane before further implementation; no checkout/reset in other's active workspace. This new instruction is currently unstaged here; preserve it when setting up isolation. Main-lane `docs/ASTRA_TO_LUNA.md` edits belong to Astra/Luna main, not edge commit.

GO local harness/tests/protocol and existing bounded read-only checks if needed. NO-GO deployment installation, export/compile, scored inference/latency/power study. Do not repeatedly collect unchanged inventory while waiting; move to useful implementation/design work. Main graph review and edge harness review can proceed independently.
