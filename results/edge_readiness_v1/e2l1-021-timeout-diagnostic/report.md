# E2L1-021 timeout diagnosis

Collection: 2026-09-21, read-only, existing run only. No smoke, build,
engine load, inference, retry, workspace change, power-mode change or runtime
change was performed.

## Observed facts

- No process matching the owned run path, `e2_model_smoke.py` or `--child-stage`
  remained when inspected.
- The engine path is absent. `build_result.json` is also absent; no engine was
  deserialized or treated as valid.
- `build_events.jsonl` is present and hash-bound:
  `1b2bad411c45b001086ccfcffd462d811fd85575f3ac9121994622a38accff81`.
  It records parse `1/1`, build `1/0`, TERM dispatched, KILL not dispatched,
  child return code `-15`, and termination confirmed. It does not contain
  timestamps for the attempt.
- Current resources were healthy-looking at collection time: 4,928,136 kB
  available RAM, 3,509,072 kB free swap, and approximately 99.4 GB free on
  `/tmp`. The current load average was `0.29 0.11 0.03`.
- `nvpmodel -q` reported `MODE_20W_6CORE` (mode index 8). The query also
  reported permission errors reading some EMC/VDDIN limit paths; no setting was
  changed.
- A bounded 10-second `tegrastats` sample showed no current GPU activity
  (`GR3D_FREQ 0%`) and temperatures around 30 C, with PMIC around 50 C. This
  is not evidence of conditions during the old build.
- `journalctl -k` exposed recent/current keyword lines for throttle cooling
  devices and `NVRM: No NVIDIA GPU found`, but timestamps were not aligned to
  the build attempt. `dmesg` was unavailable due to permission denial.

## Missing evidence and hypotheses

There is no completed engine, result file, build stdout/stderr, or attempt-time
resource timeline. Therefore the evidence does not distinguish among a build
that legitimately needs more than 900 seconds, a TensorRT/runtime stall or
driver problem, and a resource condition that existed only during the build.
The current healthy-looking snapshot does not disprove any of these. The
`NVRM` journal line is a candidate lead only, not a causal finding.

## Recommendation

Because no completed engine or actionable error exists, the next candidate is
one new diagnostic attempt using a fresh owned output root, unchanged ONNX,
FP16/1 GiB workspace, source/runtime and input contract, but an explicit
candidate build ceiling of 3600 seconds plus durable informative builder logs.
`3600s` is an operational proposal, not an expected duration and is not
authorized by E2L1-021. Any new attempt must remain separate from and counted
alongside this consumed attempt; the old root and private partial evidence
must not be resumed, erased or overwritten. No second build/inference is run
by this entry.
