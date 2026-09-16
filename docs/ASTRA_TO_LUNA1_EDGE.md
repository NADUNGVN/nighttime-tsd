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

## E2L1-003 — review bc611c9; fix energy math before real telemetry

Reviewed `bc611c9d8c58ba0aa5234308b4b29c415a48ba36` / L1A-002. Astra independently reran15/15edge testsPASS. Accept local/mock-only separation, named measurement boundaries, raw latency/session storage and collector hardening as preparation. **GO bounded local fixes below; NO-GO real benchmark/conversion/install remains.** Main Luna receives independent CPU ONNX collection authorization; do not wait for its graphs to fix harness.

### E1 — reproduced wrong energy results

`integrate_power_energy` currently labels partial overlap as measured full interval, and when clipping a segment uses original endpoint powers rather than interpolating powers at clipped boundaries.

Astra reproductions (timestamps in seconds, code inputs ns):

- requested[0,10], samples(2,10W),(8,10W) => returns `measured`,60J. Must not claim full-session energy: missing coverage[0,2]and[8,10]. Return unavailable/partial with coverage metadata, no full averagepower/Jperimage.
- linear samples(0,0W),(10,10W), requested[0,2] => returns10J; correct trapezoidal clipped integral is2J (boundary powers0Wand2W).

Require strictly increasing finite timestamps or explicit validated sorting policy; reject duplicate/conflicting timestamps, invalid/NaN/inf/negative power, mixed boundaries/units and unverified clock alignment. Full measured result needs samples bracketing entire interval. Interpolate power at clipped interval endpoints, don't extrapolate missing ends. Record coverage, duration, clock identity/alignment source, sample gap limitations and integration method; do not invent a validated max-gap threshold after observing real results. Tests full constant/ramp, clipped ramp, partial-left/right/interior, no overlap, duplicate/out-of-order, mixedboundary/nonfinite. If averagepower/energyperimage are added, denominator is covered full-duration/imagecount with exact time boundary, not reciprocal latency.

### E2 — measurement binding and persistence

`run_session` measures energy over outer session wall time including Python/provider overhead whereas latency excludes provider; persist absolute monotonic start/end/session duration and label energy boundary separately. Postprocess may be asynchronous for a future adapter: declare CPU synchronous or provide final synchronization before ending decoded-image timer. Make boundary enum validation explicit; unknown strings cannot silently choose end-to-end via else branch. Bind pool IDs/hash/order/size instead of hidden `%256` assumption for real adapters; mock may retain explicit mockpool.

Use atomic/exclusive no-overwrite open (`x`) in harness writer; current exists+write_text has a race. Preserve mock-only provenance and don't let config claim pipelined-throughput while harness reports serial. Future actual model/backend/device bindings must be checked, not self-declared labels only; no actual runtime adapter needed in this task.

### Handoff

Luna1 adds tests, updates measurement protocol with corrected integration and timing/clock scope, records **L1A-003**, commits/pushes scoped edge branch. Do not rewrite existing mock or inventory artifacts; new mock outputs if needed. No newSSH inventory required; no device inference or power-mode modification. Exact SKU/meter/cooling/operator questions remain open but do not prevent local math fixes. Astra leaves this entry unstaged in the isolated edge worktree for Luna1 to commit.

## E2L1-004 — accept integration math fixes; close actual session/pool/clock bindings

Reviewed `51bbecafa4420c1b1f5decd3a1dc80042db784f9` / L1A-003 on 2026-09-16. Astra independently reran **18/18 PASS** with `python -m unittest discover -s tests -p 'test_edge*.py' -v`. The slash-module invocation reported by Luna did not resolve the local `tests` namespace in Astra's environment; discovery did, without code changes. Accept clipped trapezoid correction (2J ramp case), partial coverage labeling, input-order checks, boundary enum validation and exclusive writer. No actual edge runtime/telemetry/energy measurement was performed by Astra.

**Decision: GO bounded local completion below; NO-GO scored/device inference or benchmark.** These are concrete contract failures, not a reason to wait for main Luna's graph. Do not rewrite earlier mock or inventory artifacts.

### E3 — measured image sequence does not use declared pool

Warmup now uses `pool_indices`, but measured calls still use `image_provider(index % 256)` at `run_session`. Astra reproduction: pool IDs `(a,b)`, order `(1,0)`, warmup=2, measured=4 gives provider calls `[1,0,0,1,2,3]`; required sequence with independent phase restart is `[1,0,1,0,1,0]`. Current output claims pool size 2 while calling indices 2 and 3 outside that pool.

Use declared order/size in both loops, explicitly define phase restart vs continuation, and persist/check actual consumed IDs/order or a deterministic sequence hash. Test non-256 pool, reversed order, wraparound and a provider that raises for out-of-range indices. Validate unique IDs and integer non-boolean permutation indices. Mock ID hash remains mock-only; future actual image-content provenance must not be replaced by hashing IDs alone.

### E4 — warmup included in energy, missing measured-window evidence

`session_start_ns` is captured before warmup and is passed to energy integration; `measured_start` is unused. `warmup_in_timing=false` is true for latency but not this energy interval. This risks later dividing warmup-inclusive joules by only measured calls.

Persist total-session window **and** post-warmup measured-loop outer window. The primary energy interval for measured-image energy must use measured-loop start/end, including provider/Python overhead for those calls but excluding warmup. Label `includes_warmup=false`, image count and distinct latency/energy boundaries. Optional warmup-inclusive energy must be separately named with its own count/window, never silently mixed. No need to add an energy-per-image statistic yet. Add deterministic mocked-clock tests with a large warmup duration to prove exclusion; do not rely only on wall-time greater than zero.

### E5 — clock consistency within samples is not binding to session clock

Astra reproduced `integrate_power_energy(rows, 0, 10, clock_identity='session-clock', alignment_source='same-process')` returning `measured` while both rows declare `clock_identity='other-clock'`. The keyword currently acts as a default and silently loses to row metadata. `run_session` supplies no expected clock identity at all. Equal strings among rows do not establish alignment to `perf_counter_ns` boundaries.

Define explicit requested/session clock binding. If an expected clock/alignment is supplied, contradictory sample metadata must fail; if conversion is needed, require an explicit validated conversion/evidence before integration, not an arbitrary nonempty string. Bind host/session monotonic clock and method in the session record; do not claim independent verification merely because caller supplied metadata. Mock evidence is explicitly mock. Tests: internally consistent rows from a wrong clock versus the expected session, conflicting alignment evidence, valid same-session clock, and missing/unverified binding. Raw unaligned telemetry may be saved as diagnostic evidence but must not produce measured session energy.

Also finish E2's postprocessing contract: current decoded-image timer ends immediately after callback. For now explicitly support CPU-synchronous postprocessing only and reject an async mode, or implement/test an explicit final synchronization hook for async adapters. Do not silently assume all future device callbacks are synchronous. No real runtime adapter is required in this repair.

### Parallel work and handoff

Luna1 fixes E3-E5 together, adds behavioral regression tests, updates protocol and records **L1A-004**. Tests must exercise the actual provider calls, time boundaries and expected-clock mismatch; a manifest containing the new fields alone is not validation. Push scoped branch changes including this inbox entry; no automatic merge, no changes to main graph worktree. Astra leaves the doc unstaged for Luna1.

Existing E2L1-001 permission for bounded **read-only raw telemetry availability samples** on E1/E2/E3/E5 remains in force, independently of this mock repair. If useful, Luna1 may collect a short bounded sample (at most 30 seconds per device, no workload generation or mode changes) with host monotonic receive timestamps, exact source/units and unavailable/permission states. Use existing tools/access only; no installation/sudo mutation. Such samples establish parser/source availability, not benchmark power, idle control, clock alignment accuracy or measured inference energy. No need to repeat unchanged inventory or wait for all devices. Preserve exclusive new output paths and do not commit private endpoints. Scored inference, TensorRT/Hailo conversion and benchmarking still need separate review.

## E2L1-005 — accept a1d62cf mock corrections; GO bounded raw telemetry feasibility

Reviewed commit `a1d62cf556434e197abf97da39e90aa618fc485d`, L1A-004. Astra independently ran `python -m unittest discover -s tests -p 'test_edge*.py' -v`: **23/23 PASS**; py_compile of harness/tests and git diff --check PASS. Worktree was clean before this entry. Review used API/interface-design boundary checks. No SSH, actual device inference, telemetry collection or benchmark was performed by Astra.

**Decision: ACCEPT E3/E4/E5 for the current CPU/mock harness.** Observed tests exercise non-256 pool permutation/wraparound and actual provider calls, measured-window energy excluding warmup (50J versus separately labeled 150J total-session fixture), expected-clock mismatch/conversion checks, and awaitable postprocess rejection. These complete the requested local repairs; do not open another generic mock-hardening cycle before collecting useful device evidence.

Acceptance is scoped: `completed_mock` remains mock-only. Callback return not being awaitable does not prove a future GPU callback has completed; a real adapter needs explicit stream/event synchronization. Injected fixture clock is not a real host clock measurement. A caller's `validated=true` conversion/evidence string is supplied provenance, not independent verification. Before a real energy benchmark, retain raw timestamps plus actual source/target clock identities, conversion parameters/evidence/uncertainty in persisted artifacts; the current integration result does not itself preserve the whole conversion mapping. These are real-adapter prerequisites, not blockers for raw telemetry sampling below.

### Next independent task: telemetry feasibility, not model benchmarking

Owner **Luna1**, same isolated edge worktree/branch workflow. GO existing authorized SSH aliases from `D:/Research/Teacher_Vu/docs/SHARED_INFRASTRUCTURE.md` for **E1/E2/E3/E5 only**. Main Luna still does not SSH server. No need to wait for YOLO26 graph preparation or for all devices to be reachable.

1. Reuse the accepted inventory/collector and obtain a short **raw read-only telemetry sample**, up to 30 seconds per device, using already installed tools and readable sensor files. Prioritize E2/E3; E1/E5 can proceed independently. No synthetic load, model load/forward, TensorRT/Hailo conversion, calibration, stress test or benchmark. No package installation, sudo mutation, fan/clock/power-mode changes, reboot or modification of another project's environment. Avoid repeated inventory of unchanged facts.
2. Use a bounded foreground child and capture raw stdout/stderr, exact command, return status, duration, device alias and source/code hash. Keep lifecycle scoped to the collector's own child; no global process-name stop command (including a global telemetry stop), no signals to other users' processes. If a tool's bounded execution/lifecycle is unclear, use simpler read-only sensor queries or record unavailable. SSH host-key checking remains enabled. Do not publish private endpoints/credentials.
3. Timestamp samples on the **device** with an identified monotonic clock where feasible; separately record local receive timestamps if used. SSH receipt time is not sensor acquisition time and must not be silently treated as aligned. Record buffering, sampling gaps, timestamp origin and unknowns. Raw power sources may have different rails/units and instantaneous/averaged fields: preserve original labels and units, do not sum overlapping rails or rename a module reading whole-device power. Missing power does not block thermal/memory collection; report unavailable.
4. Produce a small CPU parser/fixture validation for actual captured formats (if a new parser is necessary), preserving raw evidence alongside parsed values. Report observed memory, thermal/throttle fields and current power mode where available; do not call the device controlled-idle or steady-state from these snapshots. **Do not feed this run into measured inference energy/FPS/AP**: it has no model workload and establishes only source/parser feasibility.
5. Write exclusive new artifacts under `results/edge_readiness_v1/e2l1-005/<device_id>/<run_id>/`, with a concise feasibility summary: available channels, units/boundaries, clock limitations, privileges/tools missing, and which next adapter prerequisites remain. Preserve all earlier mock/inventory outputs. Artifact hashes must be computed from actual stored bytes; keep canonical Git hash handling explicit on Windows.

No external meter purchase/setup, Hailo compiler host change or scored device selection is implied. Unknown cooling/power-supply/external-meter information may remain unknown in this feasibility report; do not hold all progress for it. E1 CPU and Hailo remain possible conditions on one physical board, E5 is Orin Nano Super, and no legacy Nano/E4 substitution is authorized.

### Completion and gates

Luna1 may implement/test the small read-only capture/parser plumbing locally and execute this bounded task via existing edge SSH access under this authorization. If it would require installation or changing device state, stop only that affected operation and report the prerequisite. Append **L1A-005** with actual contacted aliases, exact action scope, artifact locations/hashes, tests and limitations; push scoped code/docs/artifacts including this entry using the established account. Do not merge another lane or stage unrelated files. Astra leaves this entry unstaged for Luna1 to commit/push.

GO raw telemetry feasibility now. NO-GO model inference, export/compile/build, benchmark, energy-per-image claims or matrix expansion. Those need source-model/runtime adapter correctness and a reviewed measurement protocol; mock acceptance alone cannot authorize them. Main Luna's G4/G5 work proceeds in parallel without a dependency on this task.
