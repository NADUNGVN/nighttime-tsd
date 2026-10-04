# ST-SERVER-01 operator runbook — active A2L-054 conditional GO

## Active dispatch addendum — 2026-10-04

The older HOLD language below is preserved as decision history and is
superseded by the accepted `docs/ST_SERVER_01_RECOVERY_GO_20261004.md`.
Astra accepted reviewed executable `57faee011413686792ff654727418909371580f5`
and granted conditional GO for **one** fresh v2 attempt. This is not permission
for a second attempt. Luna does not SSH; the user alone runs SERVER-01 commands.
Preserve v1 and the unrelated dirty checkpoint. For this confirmation runner,
desktop PID/path confirmations identify known desktop processes only; they do
not authorize competing background/GPU compute. An unknown workload or
unassessed contention blocks dispatch. The accepted confirmation executable
explicitly rejects any nonempty `--confirm-background-process` list; operator
confirmation cannot bypass that check. A2L-044's shared-workload exception
belongs to the completed feasibility smoke only and is not applicable to this
repeated-build confirmation. Do not alter code, kill another user's process,
or mislabel compute as desktop to obtain dispatch.

### A. Operator preflight — run and return the complete output

This read-only command checks the reviewed executable/helper bytes, current
GPU/process paths, runtime, disk, frozen inputs, v1 presence and v2 absence.
It creates no output directory and starts no GPU work.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && \
env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master && \
env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse HEAD && \
test "$(env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse 57faee011413686792ff654727418909371580f5)" = "57faee011413686792ff654727418909371580f5" && \
env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git diff --exit-code 57faee011413686792ff654727418909371580f5 -- \
  configs/precision_head_confirmation_v1.json \
  configs/precision_head_confirmation_execution_v1.json \
  scripts/precision_head_confirmation_contract.py \
  scripts/precision_head_calibration_order.py \
  scripts/run_precision_head_confirmation.py \
  scripts/run_precision_head_confirmation_server.py \
  scripts/analyze_precision_head_confirmation.py \
  scripts/prepare_precision_head_confirmation.py \
  scripts/prepare_precision_head_confirmation_graph.py \
  scripts/run_precision_head_source_export_dev_bridge.py \
  scripts/capture_cctsdb_validator.py \
  scripts/verify_cctsdb_capture.py \
  scripts/analyze_dev_quantization.py \
  scripts/audit_cctsdb_measurement.py \
  scripts/run_architecture_matrix.py \
  scripts/uniform_build_repeat.py && \
env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git status --short --branch && \
hostname && \
nvidia-smi --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader && \
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader && \
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk '$1 ~ /^[0-9]+$/ {print $1}'); do printf 'PID %s | ' "$p"; ps -o user=,comm=,args= -p "$p"; done && \
df -h /home/ubuntu && \
conda activate nighttime-tsd && \
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && \
local/g0_size_env/bin/python -c "import importlib.metadata as m,numpy as np,torch,ultralytics,tensorrt as trt; print({'torch':torch.__version__,'ultralytics':ultralytics.__version__,'tensorrt':trt.__version__,'numpy':np.__version__,'pycocotools':m.version('pycocotools'),'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})" && \
sha256sum \
  configs/precision_head_confirmation_v1.json \
  configs/precision_head_confirmation_execution_v1.json \
  results/yolov8n_cctsdb_clean_s42_v1/weights/best.pt \
  results/yolo26n_cctsdb_clean_s42_v1/weights/best.pt \
  results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolov8n/model.onnx \
  results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolo26n/model.onnx \
  scripts/precision_head_confirmation_contract.py \
  scripts/precision_head_calibration_order.py \
  scripts/run_precision_head_confirmation.py \
  scripts/run_precision_head_confirmation_server.py \
  scripts/analyze_precision_head_confirmation.py \
  scripts/prepare_precision_head_confirmation.py \
  scripts/prepare_precision_head_confirmation_graph.py \
  scripts/run_precision_head_source_export_dev_bridge.py \
  scripts/capture_cctsdb_validator.py \
  scripts/verify_cctsdb_capture.py \
  scripts/analyze_dev_quantization.py \
  scripts/audit_cctsdb_measurement.py \
  scripts/run_architecture_matrix.py \
  scripts/uniform_build_repeat.py && \
test -f results/measurement_audit_v1/server_precision_head_confirmation_v1/execution_manifest.json && \
test ! -e results/measurement_audit_v1/server_precision_head_confirmation_v2 && \
echo OUTPUT_ABSENT
```

Required frozen hashes: config `2a7f07e145d7561fd929e53eb930309e54a952da150d1e0b87f3eeb1412baca8`; YOLOv8n checkpoint `b2b7a1c77a19499ded33c9cc11c621757077aa871f4e7f7a1fcdbf94f53b383b`; YOLO26n checkpoint `2bb49f85f581469fc7942652d5fda4da44278d57fa8363e8f7295daa49f0d01e`; accepted ONNX v8n `e22d53bbeb333f44783535d911d5e318ebb7d500e8fcb3d1cbb1284f5248d603`; v26n `1b2467ccd62bd1e53f3bde3e3f22e1b42129711d3e368a4b4666d025099ce5cc`. Any mismatch, missing runtime, existing v2 root, unknown/incompatible compute workload or inadequate resources stops dispatch. Do not clean, kill or alter another user's workload. Record current desktop PID/path identities if present, but their confirmation does not authorize competing compute.

### B. Fresh v2 plan — CPU-only after preflight passes

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && test ! -e results/measurement_audit_v1/server_precision_head_confirmation_v2 && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 YOLO_AUTOINSTALL=0 ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1 PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 local/g0_size_env/bin/python scripts/run_precision_head_confirmation.py --phase plan --xml /home/ubuntu/Dung_TDTU/nighttime-tsd/data/raw/CCTSDB2021/xml.zip --out-dir results/measurement_audit_v1/server_precision_head_confirmation_v2
```

Require XML validation 1,636 images / 2,706 instances and plan accounting of
84 builders / 78 captures / 127,608 image-model passes. This step must not
create model, engine, calibration or timing binaries. Preserve any plan
failure; do not overwrite or rerun it.

### C. Scored command — fill only after returned snapshot and valid plan

Luna will return the complete foreground command with exact current
confirmations after reading the user's output. Do not run the placeholder
below. Do not include `--confirm-background-process`: the reviewed runner
rejects it, and the historical smoke exception is not part of this study.
Desktop confirmations identify desktop processes only. If competing compute
is present or any workload is unknown or incompatible, remain
`operator_pending`; do not retry on another device or kill processes.

```bash
conda activate nighttime-tsd && cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && test -f results/measurement_audit_v1/server_precision_head_confirmation_v2/confirmation_plan.json && test ! -f results/measurement_audit_v1/server_precision_head_confirmation_v2/execution_manifest.json && local/g0_size_env/bin/python scripts/run_precision_head_confirmation.py --phase scored --go-token ASTRA_INTEGRATED_GO_REQUIRED --out-dir results/measurement_audit_v1/server_precision_head_confirmation_v2 --device 0 --confirm-desktop-process CURRENT_PID=CURRENT_EXACT_PATH
```

The authorized ceiling is one attempt: 84 builders / 78 captures / 127,608
passes; cumulative ceiling including the failed v1 builder attempt is 85.
Foreground only. No resume, retry, canary, replacement cell, new export,
training, native forward or benchmark. Any failure stops and preserves partials.
The `--confirm-desktop-process` values bind only the exact current desktop
process identity. They are not a workload exception and do not permit
competing background compute.

### D. Read-only progress and terminal evidence

From a second terminal, inspect the manifest/states and a supplementary GPU
snapshot only. A missing live child log or quiet foreground terminal does not
diagnose a hang; the parent publishes each job log after child return/timeout.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python -c "import json,pathlib; r=pathlib.Path('results/measurement_audit_v1/server_precision_head_confirmation_v2'); p=r/'execution_manifest.json'; m=json.loads(p.read_text()) if p.exists() else {}; jobs=m.get('jobs',[]); print({'status':m.get('status','manifest_not_created'),'jobs_returned':len(jobs),'jobs_completed':sum(j.get('state',{}).get('status')=='completed' for j in jobs),'expected_jobs':m.get('expected_jobs',84)}); [print(f.relative_to(r).as_posix(),json.loads(f.read_text()).get('stage'),json.loads(f.read_text()).get('status'),json.loads(f.read_text()).get('forward_completed'),json.loads(f.read_text()).get('error')) for f in sorted(r.glob('jobs/*/child_state.json'))]" && nvidia-smi --query-gpu=uuid,pstate,temperature.gpu,power.draw,memory.used --format=csv,noheader && nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
```

Do not launch a second runner based on partial status or a stale `running`
file. Return the foreground exit status and preserve all output.

### E. Scoped artifact publication and CPU analysis

Before staging, enumerate the output and inspect the plan/manifest. Stage only
public plan/schedule/execution manifests, public per-job states, cell metrics,
predictions, logs, sanitized inspector JSON, reports and failure/partial
inventories. Never stage `private/`, engines, ONNX, checkpoints, calibration or
timing caches, raw tensors or temporary files. Use explicit allowlisted paths
from the returned inventory (not `git add` on the whole study root), then
inspect `git diff --cached --name-only` and push only scoped evidence.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && find results/measurement_audit_v1/server_precision_head_confirmation_v2 -type f -printf '%P\n' | sort && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git status --short --branch
```

After the operator pushes the artifact, Luna audits canonical Git blobs,
provenance, all schedule/counters, bindings, cache behavior, metrics, telemetry,
lifecycle and inventory. Only after that audit, run the locked analyzer:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/analyze_precision_head_confirmation.py --root results/measurement_audit_v1/server_precision_head_confirmation_v2 --xml /home/ubuntu/Dung_TDTU/nighttime-tsd/data/raw/CCTSDB2021/xml.zip --out-dir results/measurement_audit_v1/server_precision_head_confirmation_analysis_v2
```

An incomplete/invalid cell remains incomplete; no replacement run is started.
Update L2A-053 and the task board/manuscript only from audited evidence.

Luna does not SSH and does not run TensorRT locally. The HOLD statement in this
historical A2L-052 record was superseded by A2L-054, which accepted the reviewed
executable and granted conditional GO for one fresh v2 attempt. It is not the
current dispatch decision; use the active addendum and Sections A–E above.
Do not use `nohup`. Preserve the failed v1 root byte-for-byte; use only the
fresh v2 root after current preflight and plan checks pass.

## Historical A2L-052 recovery notes — superseded by the active A2L-054 dispatch above

After Astra review, replace `FULL_COMMIT` with the full SHA printed in the
current L2A-053 handoff. The pre-review revision is not a dispatch revision.

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git pull --ff-only origin master && test "$(env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git rev-parse HEAD)" = "FULL_COMMIT" && env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git status --short --branch && sha256sum configs/precision_head_confirmation_v1.json configs/precision_head_confirmation_execution_v1.json scripts/precision_head_confirmation_contract.py scripts/precision_head_calibration_order.py scripts/run_precision_head_confirmation.py scripts/run_precision_head_confirmation_server.py scripts/analyze_precision_head_confirmation.py scripts/prepare_precision_head_confirmation.py scripts/prepare_precision_head_confirmation_graph.py scripts/capture_cctsdb_validator.py scripts/verify_cctsdb_capture.py scripts/analyze_dev_quantization.py scripts/audit_cctsdb_measurement.py scripts/run_architecture_matrix.py scripts/uniform_build_repeat.py && test -f results/measurement_audit_v1/server_precision_head_confirmation_v1/execution_manifest.json && test ! -e results/measurement_audit_v1/server_precision_head_confirmation_v2
```

The operator must also record GPU UUID/name/driver, temperature/power/clocks,
compute PIDs and exact `ps` commands, runtime package versions, checkpoint/
ONNX/config hashes, disk space, and output absence. The accepted bindings are:
YOLOv8n checkpoint `b2b7a1c77a19499ded33c9cc11c621757077aa871f4e7f7a1fcdbf94f53b383b`,
YOLO26n checkpoint `2bb49f85f581469fc7942652d5fda4da44278d57fa8363e8f7295daa49f0d01e`,
YOLOv8n ONNX `e22d53bbeb333f44783535d911d5e318ebb7d500e8fcb3d1cbb1284f5248d603`,
and YOLO26n ONNX `1b2467ccd62bd1e53f3bde3e3f22e1b42129711d3e368a4b4666d025099ce5cc`.
Desktop exceptions use the current exact PID/path confirmation. Unknown or
non-desktop compute remains a block; do not kill, pause, reprioritize, or
change another user's process. A sampled idle state is not isolation proof.

## Read-only plan preparation (historical A2L-052 wording; current command is in section B above)

Do not rerun graph preparation or export. The accepted ONNX and graph-audit
artifacts are read-only inputs. Only after Astra accepts this recovery packet,
create the fresh v2 study plan with the CPU-only parent. It verifies the
checkpoint/ONNX bytes, nested graph mapping, calibration readiness, XML and
schedule, and binds exact executable-helper hashes prospectively:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 YOLO_AUTOINSTALL=0 ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1 PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 local/g0_size_env/bin/python scripts/run_precision_head_confirmation.py --phase plan --xml /home/ubuntu/Dung_TDTU/nighttime-tsd/data/raw/CCTSDB2021/xml.zip --out-dir results/measurement_audit_v1/server_precision_head_confirmation_v2
```

This command must finish before the scored command, must verify the XML archive
against the exact 1,636-image/2,706-instance dev contract, and must not create ONNX,
engine, calibration or timing binaries.

## Scored execution — historical HOLD, superseded by the active A2L-054 conditional GO above

There is deliberately no scored command in this recovery handoff. The old R5
command points at immutable failed v1 and must not be reused. After Astra's
integrated review/GO and a fresh operator snapshot, Luna will provide the exact
foreground v2 command using current desktop confirmations. The reviewed v2
attempt is one 84-builder/78-capture run, with a cumulative ceiling of 85
attempted builders across v1+v2. Any v2 failure stops; no retry, resume,
canary, replacement, or additional cell.

Multi-host partition is not implemented in this runner. Do not split or merge
the study across hosts; run both complete model blocks sequentially on one
compatible GPU so the locked model-local GPU/runtime comparison is preserved.

## Scoped publication

After completion or failure, publish only the allowlist generated by the
runner: `confirmation_plan.json`, `schedule.json`, `execution_manifest.json`,
`jobs/*/child_state.json`, `jobs/*/cell_metrics.json`,
`jobs/*/predictions.json`, `public/inspectors/*.json`, reports, logs,
telemetry, hashes, and failure/partial inventories. Inspector JSON is
sanitized, nonbinary evidence only. Never publish checkpoint, ONNX, engine,
calibration/timing cache, private tensor, or private temporary files. Push the
scoped artifact commit and send Luna the full SHA.

## Local post-run audit

Luna pulls the artifact commit and checks the exact 84/78 schedule, every
identity link, cache-only counters, fresh timing-cache evidence, mapping and
output semantics, capture membership, XML/hash bindings, telemetry/workload
violations, finite outputs, deadlines, lifecycle release and partial files.
Then Luna runs the CPU analysis command on the pulled artifact only:

```bash
cd /home/ubuntu/Dung_TDTU/nighttime-tsd-new && local/g0_size_env/bin/python scripts/analyze_precision_head_confirmation.py --root results/measurement_audit_v1/server_precision_head_confirmation_v2 --xml /home/ubuntu/Dung_TDTU/nighttime-tsd/data/raw/CCTSDB2021/xml.zip --out-dir results/measurement_audit_v1/server_precision_head_confirmation_analysis_v2
```

The analyzer reconstructs pooled COCO/XML AP from detection records and uses
the shared PCG64 image bootstrap. Any
missing or invalid cell is reported as incomplete; no replacement run is
started. The final packet contains machine JSON, all-cell/contrast/variation
tables and manuscript Methods/Results/Limitations text for Astra review.
