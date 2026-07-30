# Jetson continuation checklist

This file records the state of the CCTSDB2021 study before Jetson hardware is
available. It is deliberately a checklist, not a claim of final deployment
performance.

## Current evidence

- Clean CCTSDB2021 protocol: 14,720 train / 1,636 deterministic development /
  1,500 untouched official-test images; seed 42.
- FP32 checkpoints and official-domain JSON results are versioned for YOLO11n,
  YOLOv8n, and YOLO26n.
- YOLO11n TensorRT 10 results are versioned for FP16, INT8 uniform calibration,
  and INT8 low-luminance calibration.
- Low-luminance INT8 retains +0.74 mAP50 on `night` relative to uniform INT8,
  but is not consistently better across all domains.
- On the Quadro RTX 8000 server, FP16 is both more accurate and faster than
  either current INT8 engine. This is a server-only observation.

## Claims that must wait for Jetson

- Do not claim latency, throughput, memory, power, energy, or speed-up on
  Orin from the RTX 8000 benchmarks.
- Do not copy an x86 RTX 8000 `.engine` to Jetson. TensorRT engines are
  hardware-specific and must be rebuilt on the target.
- Do not claim INT8 is faster than FP16: the current server measurements show
  the opposite for this small model with end-to-end preprocessing/postprocess.
- Do not claim low-luminance calibration wins generally; it is only a positive
  night-domain signal in one seed.

## Artifacts to bring to Jetson

- Repository at a tagged/recorded Git commit.
- `best.pt` checkpoint(s), at minimum
  `results/yolo11n_cctsdb_clean_s42_v2/weights/best.pt`.
- CCTSDB processed data or a readable path to it. Only `train/images` may be
  used for INT8 calibration; `dev/images` is used for benchmark images;
  official test remains evaluation-only.
- Versioned manifests in `results/paper_artifacts/v1/metadata/`:
  dataset split manifest and the two calibration manifests.
- Deployment contract:
  `configs/deployment/yolo11n_fp16_trt10.json`. Its confidence threshold is
  provisional and must be selected on development data, never official test.

## Jetson setup record (capture before any benchmark)

Record in a new result JSON or environment file:

- Jetson model, RAM size, Ubuntu/L4T/JetPack version.
- TensorRT, CUDA, cuDNN, Python, PyTorch, and Ultralytics versions.
- Power mode (`nvpmodel`), clocks (`jetson_clocks` status), cooling, input
  power supply, and whether other GPU processes are running.
- Git commit, model/checkpoint SHA256, calibration-manifest SHA256, and engine
  SHA256.

Use JetPack-provided TensorRT on Jetson. Do not install an unbounded pip
TensorRT package or upgrade Torch/Ultralytics during the benchmark campaign.
The server exporter currently enforces TensorRT major version 10; if JetPack
has another major version, stop and record it before changing any exporter.

## Required Jetson experiments

1. Recreate the exact uniform and low-luminance calibration sets from their
   manifests and `train/images`, or verify their selected source-image lists.
2. Rebuild FP16, INT8-uniform, and INT8-low-luminance engines directly on
   Jetson at image size 640 and static batch 1.
3. Evaluate each rebuilt engine on the same eight official CCTSDB splits.
   Commit JSON/provenance, not engine binaries or images.
4. Benchmark each engine using `scripts/benchmark_tensorrt_engine.py` on
   preloaded development images with a fixed warm-up and sample count.
5. Add a Jetson power logger using `tegrastats` or `jtop`; report sampling
   method, mean/peak power, duration, and energy per image. The current server
   benchmark has only before/after power snapshots and is not an energy result.
6. Select the deployment confidence threshold on development data only, save
   it in the deployment contract, then freeze it before final official-test
   reporting.

## Research experiments still needed

- A 50/50 mixed calibration set (uniform + low-luminance) for YOLO11n.
- At least one INT8 calibration comparison on YOLOv8n to test architecture
  generality. YOLO26n remains an FP32 architecture baseline unless resources
  permit broader coverage.
- Qualitative error analysis for night and rain. Rain has highly imbalanced
  classes, so its per-class AP needs an explicit caveat.
- Prefer a clean CUDA 12.1/TensorRT 10 environment for final server
  reproducibility: the current server runs correctly, but its pip freeze still
  contains CUDA 13 packages left from the failed TensorRT 11 attempt.

## Publishing checklist

- Report mean/p50/p95 latency, throughput, memory, power, and energy with the
  exact target configuration.
- Present mAP50 and mAP50-95 for full/night as primary results; weather
  subsets are diagnostics with sample-size/class-distribution caveats.
- Keep the paper's main claim narrow: calibration policy changes domain-wise
  INT8 robustness; it is not a new detector architecture claim.
