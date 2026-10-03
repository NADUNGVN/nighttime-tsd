# Edge evidence ledger

Scope: existing E2 correctness evidence and one bounded read-only inventory on each currently reachable approved alternative device. No source-server access, new model call, build, inference, benchmark, install or device change occurred in this checkpoint.

## Frozen experiment identity

- Model family: YOLO11n CCTSDB2021 engineering/evaluation lane; batch 1, 640×640 input; native output contract `[1,7,8400]`.
- Accepted source ONNX SHA-256: `bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7`.
- Three-fixture source-package manifest SHA-256: `60744680973a73d2986bdf59ce3c6bc956119aeeebbd2106960df57947665c08`. This is the earlier three-fixture package, not the outstanding full-dev source reference.
- E2: Jetson Xavier NX; historical execution runtime TensorRT 8.5.2.2. The accepted private engine SHA-256 is `581a9ea2eafdae690f25f57ab88ba1bcffdafa99e5322ea50ee43678520e7f56`, 7,158,097 bytes. This identity is historical until rechecked on E2.
- Canonical full-dev scope: 1,636 ordered images / 2,706 instances. Current full-dev source and target row counts remain `0/1636` each.

## E2 execution history and boundaries

| Attempt | Observed evidence | Interpretation / limitation |
| --- | --- | --- |
| Attempt 1, E2L1-017 | Parse 1/1; build attempted 1, completed 0; build deadline 900 s; TERM sent, child exit `-15`, termination recorded; no engine or build result. Event hash `1b2bad411c45b001086ccfcffd462d811fd85575f3ac9121994622a38accff81`. | Failure/timeout, not evidence of model incompatibility or root cause. Event-time timestamps were unavailable; the later resource snapshot is not a reconstruction of build-time conditions. Preserve the attempt unchanged. |
| Attempt 2, E2L1-022 | One FP16 TensorRT build, 1 GiB workspace, ceiling 3,600 s; parse/build/load all 1/1; ordered fixtures `00006`, `00009`, `00028` completed 3/3 with synchronized copies/comparisons 3/3 and no unknown completion. Engine hash above. Public manifest hash `29954570c5b1f44af928e283ce2ab89885c224b165ab85b4f69de6d5ae6a4e74`. | The approximately 16m35s build and 13.3s inference event windows are stage wall times, not latency/FPS or benchmark results. No third build or target call is authorized here. |

The user previously reported `31,109 s` as the aggregate E2 time across the two attempts and said it need not be reacquired. This ledger preserves that as a user-reported accounting value, not an independently recomputed benchmark. Attempt 1's public diagnostic lacks event timestamps, so no attempt-level reconstruction is claimed.

## Source/export versus target/runtime discrepancies

Keep these comparisons separate. Source-native versus source-ONNX was strict-pass for `00006` and `00009`, but retained one box mismatch for `00028`. TensorRT-vs-ONNX raw output failed on all three fixtures; all counted violations were box elements, while score-domain violations under the frozen absolute-plus-relative policy were zero.

| Fixture | Source export | TRT-vs-ONNX box / score violations | Same-origin detections after NMS | Max score Δ | Max box Δ (input px) | Same-origin box IoU range |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `00006` | pass | 965 / 0 | 1 → 1 | 0.000801444 | 0.0971832 | 0.992440–0.992440 |
| `00009` | pass | 1,226 / 0 | 2 → 2 | 0.000768185 | 0.212311 | 0.979394–0.987351 |
| `00028` | fail (one source box element) | 564 / 0 | 2 → 2 | 0.000089407 | 0.263626 | 0.993441–0.994679 |

The NMS replay used the accepted Ultralytics CPU helper at version 8.4.102 with confidence 0.001, IoU 0.7 and max detections 300. Five retained same-origin rows were audited. Score differences are within the declared tolerance, not exact equality. The three fixtures are diagnostic examples, not an AP sample or evidence of ground-truth accuracy impact.

Accepted diagnostic artifacts are [`e2l1-023-output-diagnostic`](../../results/edge_readiness_v1/e2l1-023-output-diagnostic/) and [`e2l1-024-saved-output-audit`](../../results/edge_readiness_v1/e2l1-024-saved-output-audit/). Public raw-result FAIL status is intentionally preserved. No box correction, tolerance change, image selection or new inference was made.

## Current bounded alternative-device inventory

Collection UTC: `2026-10-03T18:19:39Z` (2026-10-04 Asia/Saigon). One bounded SSH inventory was made for each authorized alias `pi5`, `rubik`, `nano`; transport exit 0 and 27/27 markers were recorded for each. Full sanitized evidence and hashes are in [`edge_contingency_20261004`](../../results/edge_readiness_v1/edge_contingency_20261004/README.md). Raw command captures remain outside the repository. No power mode, clock, fan or runtime state was changed.

## Gates still open

- User-operated full source CPU reference and its saved-artifact audit; source expected 1,636 ONNX forwards and zero native forwards.
- POSIX TERM-ignore/kill-escalation mock test must report exactly one test, zero skips, zero failures/errors before E2 dispatch. The current Windows host cannot establish this gate.
- Private package/hash audit and per-image input binding before any target enqueue.
- Fresh E2 identity, exact existing-engine hash, runtime/dependency, storage/memory and workload check after access returns. No E2 check was attempted while the user-reported offline state held.
- Only after all gates: the already-authorized single existing-engine target pass, maximum one pass/image, no build/warmup/retry/benchmark; then paired COCO/XML AP and 1,000-resample analysis with seed 20260916.

Power/energy measurement is not present in this correctness evidence. Available telemetry commands on alternative devices do not establish a calibrated measurement boundary.
