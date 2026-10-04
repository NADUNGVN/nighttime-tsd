#!/usr/bin/env bash
set -euo pipefail

repo=/home/ubuntu/Dung_TDTU/nighttime-tsd-new
reviewed=57faee011413686792ff654727418909371580f5
cd "$repo"

safe_git() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD PATH=/usr/bin:/bin /usr/bin/git "$@"
}

code_paths=(
  configs/precision_head_confirmation_v1.json
  configs/precision_head_confirmation_execution_v1.json
  scripts/precision_head_confirmation_contract.py
  scripts/precision_head_calibration_order.py
  scripts/run_precision_head_confirmation.py
  scripts/run_precision_head_confirmation_server.py
  scripts/analyze_precision_head_confirmation.py
  scripts/prepare_precision_head_confirmation.py
  scripts/prepare_precision_head_confirmation_graph.py
  scripts/run_precision_head_source_export_dev_bridge.py
  scripts/capture_cctsdb_validator.py
  scripts/verify_cctsdb_capture.py
  scripts/analyze_dev_quantization.py
  scripts/audit_cctsdb_measurement.py
  scripts/run_architecture_matrix.py
  scripts/uniform_build_repeat.py
)

printf '\n== checkout ==\n'
safe_git rev-parse HEAD
test "$(safe_git rev-parse "$reviewed")" = "$reviewed"
safe_git diff --exit-code "$reviewed" -- "${code_paths[@]}"
safe_git status --short --branch

printf '\n== host / GPU / compute processes ==\n'
hostname
nvidia-smi --query-gpu=uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk -F, '{gsub(/[[:space:]]/, "", $1); if ($1 ~ /^[0-9]+$/) print $1}')
if ((${#gpu_pids[@]})); then
  for pid in "${gpu_pids[@]}"; do ps -o pid=,user=,comm=,args= -p "$pid"; done
else
  printf 'NO_COMPUTE_APP_PIDS\n'
fi
df -h /home/ubuntu

printf '\n== runtime ==\n'
local/g0_size_env/bin/python -c "import importlib.metadata as m,numpy as np,torch,ultralytics,tensorrt as trt; print({'torch':torch.__version__,'ultralytics':ultralytics.__version__,'tensorrt':trt.__version__,'numpy':np.__version__,'pycocotools':m.version('pycocotools'),'cuda':torch.version.cuda,'available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"

printf '\n== frozen input and source hashes ==\n'
sha256sum \
  configs/precision_head_confirmation_v1.json \
  configs/precision_head_confirmation_execution_v1.json \
  results/yolov8n_cctsdb_clean_s42_v1/weights/best.pt \
  results/yolo26n_cctsdb_clean_s42_v1/weights/best.pt \
  results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolov8n/model.onnx \
  results/measurement_audit_v1/precision_head_confirmation_graph_prep_v2/models/yolo26n/model.onnx \
  "${code_paths[@]}"

test -f results/measurement_audit_v1/server_precision_head_confirmation_v1/execution_manifest.json
test ! -e results/measurement_audit_v1/server_precision_head_confirmation_v2
printf 'OUTPUT_ABSENT\n'
