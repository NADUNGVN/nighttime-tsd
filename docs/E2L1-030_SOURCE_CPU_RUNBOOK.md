# E2L1-030 source ONNX CPU reference — operator runbook

This is the user-operated SERVER-01 step only. It performs one bounded source
ONNX CPU reference pass over the canonical 1,636 CCTSDB2021/dev images after
all image, ONNX, environment, scope and provider checks succeed. It performs
zero native-model forwards, exports, builds, target calls or benchmarks. Do
not retry or resume a failed/partial stage; preserve its output and report the
partial counters.

The reviewed executable code is full commit
`988bd6f715df99da6998914404ce507204e10790`. Source ONNX identity is
`bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7`.
The current actual helper hashes at that revision are:

| File | SHA-256 |
| --- | --- |
| `scripts/edge_readiness/e2_dev_evaluation_packet.py` | `aaf0cc4ea7ef7a4eee52c920cea87beedd1b1afce89fed7e0f42cc75744c81c6` |
| `scripts/edge_readiness/e2_image_preprocess.py` | `3676634a0cca2c71b887b30e7ff734b77a04f0f0776776946ce49c58db2d169c` |
| `scripts/edge_readiness/e2_source_bundle.py` | `a40c074380d8123df63c69e922128ff8bb918e6c46ebf73320d0980be33399a4` |
| `scripts/edge_readiness/e2_output_diagnostic.py` | `df146f212de7afbe830ffe2c7d830bdead055afa85628dbbe7836df4323b2989` |
| `scripts/edge_readiness/e2_model_smoke.py` | `e2e7a63fb1da13ef2274ea7d65fa603b74139d3708a9e06139184779be9ba2e4` |
| `scripts/edge_readiness/jetson_runtime_provider.py` | `d79185a0bb0bfb31b2938c6fc9f18cf19720194b197b5c85d313743fe46ccb61` |
| `scripts/edge_readiness/jetson_adapter.py` | `2ee34aff5c1cdf446ca845417e75c62d4f0c444860be6c2efdc906f4387b8c42` |
| `scripts/edge_readiness/cuda_runtime_owner.py` | `0512964f58c4bffae85be9d83bb6c7748041e581434ed88bff7c87b2277f283b` |
| `scripts/edge_readiness/edge_errors.py` | `c13691982d80dc714b935d367ef35bb371b07dfd591a8856be3d23ac1aaa7602` |
| `tests/test_e2_dev_evaluation_packet.py` | `eb6c82c55bccfd1fad10289c85eaa08da5ec197dbfb5d62dada6cb9e718919c7` |
| `tests/test_e2_output_diagnostic.py` | `6ab4c2c0805275ddce49b2f38b01d06dff4e7db14279bd6a625277b46fd7027e` |

This runbook intentionally uses a detached worktree, leaving the operator's
existing checkout and other work untouched. It stops if either fresh path
already exists or any prerequisite fails. Do not remove or overwrite an
existing path to make it pass.

## One foreground command

Paste this whole block into the SERVER-01 shell. The environment and paths
match the previously used source environment and the accepted absolute dev
YAML. No SSH from Luna1 to SERVER-01 is part of this step.

```bash
set -euo pipefail

REPO=/home/ubuntu/Dung_TDTU/nighttime-tsd-new
BRANCH=luna1/e2l1-006-jetson-adapter-smoke
REV=988bd6f715df99da6998914404ce507204e10790
CODE=/tmp/luna1-e2l1-030-code-${REV}
ROOT=${REPO}/results/edge_readiness_v1/e2l1-030-source-reference-v1
PY=${REPO}/local/g0_size_env/bin/python
ONNX=${REPO}/results/edge_readiness_v1/e2l1-013-source-v2/private/onnx_export/best.onnx
CANON=${REPO}/results/measurement_audit_v1/server_fp16_capture_v1/validator_predictions.json
DEV_YAML=${REPO}/results/measurement_audit_v1/server_fp16_capture_v1/dev_absolute.yaml

test -d "$REPO"
test -x "$PY"
test -f "$ONNX"
test -f "$CANON"
test -f "$DEV_YAML"
test ! -e "$CODE" && test ! -L "$CODE"
test ! -e "$ROOT" && test ! -L "$ROOT"
printf '%s  %s\n' bd20b36d640c502358c44edbbde51f05267eda2518b0c0a4cfe84ca18a00d4b7 "$ONNX" | sha256sum -c -
printf '%s  %s\n' ea8f40ba75920b2a67c2b5e1fd18c3d2eaf760686f11d0efea9bb7f26d5f5479 "$DEV_YAML" | sha256sum -c -

git -C "$REPO" fetch --no-tags origin "$BRANCH"
git -C "$REPO" worktree add --detach "$CODE" "$REV"
test "$(git -C "$CODE" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$CODE" status --porcelain)"
(
  cd "$CODE"
  sha256sum -c <<'HASHES'
aaf0cc4ea7ef7a4eee52c920cea87beedd1b1afce89fed7e0f42cc75744c81c6  scripts/edge_readiness/e2_dev_evaluation_packet.py
3676634a0cca2c71b887b30e7ff734b77a04f0f0776776946ce49c58db2d169c  scripts/edge_readiness/e2_image_preprocess.py
a40c074380d8123df63c69e922128ff8bb918e6c46ebf73320d0980be33399a4  scripts/edge_readiness/e2_source_bundle.py
df146f212de7afbe830ffe2c7d830bdead055afa85628dbbe7836df4323b2989  scripts/edge_readiness/e2_output_diagnostic.py
e2e7a63fb1da13ef2274ea7d65fa603b74139d3708a9e06139184779be9ba2e4  scripts/edge_readiness/e2_model_smoke.py
d79185a0bb0bfb31b2938c6fc9f18cf19720194b197b5c85d313743fe46ccb61  scripts/edge_readiness/jetson_runtime_provider.py
2ee34aff5c1cdf446ca845417e75c62d4f0c444860be6c2efdc906f4387b8c42  scripts/edge_readiness/jetson_adapter.py
0512964f58c4bffae85be9d83bb6c7748041e581434ed88bff7c87b2277f283b  scripts/edge_readiness/cuda_runtime_owner.py
c13691982d80dc714b935d367ef35bb371b07dfd591a8856be3d23ac1aaa7602  scripts/edge_readiness/edge_errors.py
eb6c82c55bccfd1fad10289c85eaa08da5ec197dbfb5d62dada6cb9e718919c7  tests/test_e2_dev_evaluation_packet.py
6ab4c2c0805275ddce49b2f38b01d06dff4e7db14279bd6a625277b46fd7027e  tests/test_e2_output_diagnostic.py
HASHES
)

mkdir -p "$(dirname "$ROOT")"
mkdir "$ROOT"
mkdir "$ROOT/input"
export LUNA030_REPO="$REPO" LUNA030_ROOT="$ROOT" LUNA030_ONNX="$ONNX" LUNA030_CODE="$CODE" LUNA030_PY="$PY"
export PYTHONPATH="$CODE/scripts"
export CUDA_VISIBLE_DEVICES= YOLO_AUTOINSTALL=false ULTRALYTICS_AUTOUPDATE=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

# Bind the canonical ordered IDs to actual image bytes and decoded shapes.
# This is metadata/preprocessing only; it makes no model call.
"$PY" - <<'PY'
import hashlib, json, os
from pathlib import Path
from edge_readiness.e2_dev_evaluation_packet import CANONICAL_DEV_IMAGE_IDS_SHA256, DEV_IMAGES, canonical_image_ids_digest
from edge_readiness.e2_image_preprocess import inspect_image_bytes

repo = Path(os.environ['LUNA030_REPO']).resolve()
root = Path(os.environ['LUNA030_ROOT'])
canonical = repo / 'results/measurement_audit_v1/server_fp16_capture_v1/validator_predictions.json'
images_root = repo / 'data/processed/cctsdb2021_clean/dev/images'
payload = json.loads(canonical.read_text(encoding='utf-8'))
ids = sorted(str(row['image']) for row in payload['records'])
if len(ids) != DEV_IMAGES or len(set(ids)) != DEV_IMAGES or canonical_image_ids_digest(ids) != CANONICAL_DEV_IMAGE_IDS_SHA256:
    raise SystemExit('CANONICAL_DEV_ID_SCOPE_MISMATCH')
records = []
for image_id in ids:
    if Path(image_id).name != image_id or image_id in ('', '.', '..'):
        raise SystemExit('CANONICAL_IMAGE_NAME_INVALID:' + image_id)
    path = images_root / image_id
    if path.is_symlink() or not path.is_file():
        raise SystemExit('DEV_IMAGE_MISSING_OR_SYMLINK:' + image_id)
    data = path.read_bytes()
    decoded = inspect_image_bytes(data, image_id=image_id)
    records.append({'image_id': image_id,
                    'path': path.relative_to(repo).as_posix(),
                    'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest(),
                    'orig_shape': decoded['orig_shape']})
out = root / 'input/image_records.json'
with out.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump({'records': records}, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write('\n')
print('IMAGE_RECORDS=' + str(out))
print('IMAGE_RECORD_COUNT=' + str(len(records)))
print('IMAGE_RECORDS_SHA256=' + hashlib.sha256(out.read_bytes()).hexdigest())
print('CANONICAL_ID_SHA256=' + canonical_image_ids_digest(ids))
PY

# Fail closed before any forward unless the pinned source stack, CUDA-hidden
# CPU environment, ONNX contract and actual ORT provider all pass.
"$PY" - "$ONNX" <<'PY'
import json, sys
from pathlib import Path
from edge_readiness.e2_source_bundle import UltralyticsSourceRuntime

runtime = UltralyticsSourceRuntime()
versions = runtime.prepare()
onnx_path = Path(sys.argv[1])
contract = runtime.validate_onnx(onnx_path)
session = runtime.open_ort(onnx_path)
providers = list(session.get_providers())
if providers != ['CPUExecutionProvider']:
    raise SystemExit('SOURCE_PROVIDER_MISMATCH:' + repr(providers))
print(json.dumps({'preflight': 'pass', 'versions': versions,
                  'onnx_contract': contract, 'observed_providers': providers},
                 sort_keys=True))
del session
PY

test ! -e "$ROOT/source"
sha256sum \
  "$CODE/scripts/edge_readiness/e2_dev_evaluation_packet.py" \
  "$CODE/scripts/edge_readiness/e2_image_preprocess.py" \
  "$CODE/scripts/edge_readiness/e2_source_bundle.py" \
  "$CODE/scripts/edge_readiness/e2_output_diagnostic.py"

# The only model-call command in this block: at most 1,636 ONNX CPU forwards.
# Keep complete stdout/stderr. A nonzero exit or failed_partial is terminal;
# do not rerun, resume, select images or reuse this output root.
set +e
"$PY" "$CODE/scripts/edge_readiness/e2_dev_evaluation_packet.py" \
  --stage source \
  --source-root "$REPO" \
  --onnx "$ONNX" \
  --image-records "$ROOT/input/image_records.json" \
  --out-dir "$ROOT/source" \
  --allow-source-forward \
  --commit "$REV" \
  >"$ROOT/source_stdout.json" 2>"$ROOT/source_stderr.log"
SOURCE_EXIT=$?
set -e
printf 'SOURCE_STAGE_EXIT=%s\n' "$SOURCE_EXIT"
if [ "$SOURCE_EXIT" -ne 0 ]; then
  printf 'SOURCE_STAGE_FAILED_PARTIAL: preserve this root; no retry/resume\n'
  if [ -f "$ROOT/source/manifest.json" ]; then
    "$PY" - "$ROOT/source/manifest.json" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(json.dumps({'status': m.get('status'), 'error': m.get('error'),
                  'counters': m.get('attempted_completed'),
                  'executed_image_passes': m.get('executed_image_passes'),
                  'cleanup': m.get('cleanup')}, sort_keys=True))
PY
  fi
  exit "$SOURCE_EXIT"
fi

# Artifact-only audit; it rehashes every saved output and executes no model.
"$PY" - "$CODE" "$ROOT/source/manifest.json" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts'))
from edge_readiness.e2_dev_evaluation_packet import DEV_IMAGES, SOURCE_ONNX_SHA256, validate_reference_artifact

manifest_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
audit = validate_reference_artifact(manifest_path)
summary = {'audit_status': audit['status'],
           'manifest_status': manifest.get('status'),
           'commit': manifest.get('commit'),
           'source_onnx_sha256': manifest.get('source_onnx_sha256'),
           'onnx_sha256_before': manifest.get('onnx_sha256_before'),
           'onnx_sha256_after': manifest.get('onnx_sha256_after'),
           'providers': manifest.get('ort_session_providers'),
           'image_count': len(manifest.get('image_ids', [])),
           'executed_image_passes': manifest.get('executed_image_passes'),
           'counters': manifest.get('attempted_completed'),
           'predictions_sha256': manifest.get('predictions_sha256'),
           'manifest_sha256': audit['artifact_sha256']}
print(json.dumps(summary, sort_keys=True))
if audit.get('status') != 'verified' or manifest.get('status') != 'verified' or len(manifest.get('image_ids', [])) != DEV_IMAGES or manifest.get('executed_image_passes') != DEV_IMAGES or manifest.get('source_onnx_sha256') != SOURCE_ONNX_SHA256:
    raise SystemExit('SOURCE_REFERENCE_AUDIT_FAILED')
PY
```

Return only sanitized status/counters, observed versions/provider, code/helper
hashes and manifest/index/events/stdout/stderr hashes. Keep images, ONNX,
raw outputs and private package bytes out of Git. Do not upload a transfer
archive until Luna1 verifies the exact allowlist and archive hash.
