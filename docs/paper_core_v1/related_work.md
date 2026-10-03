# Related work and scope boundary

Source check performed 2026-10-04. This is a focused map, not an exhaustive
survey. A publisher/CVF/arXiv source is linked for each item. “Verified” below
means the primary landing page and bibliographic/title information were
retrieved or matched to the existing repository source record; it does not
imply that every result in the cited work was independently reproduced.

| Work | Primary source / status | Relevant overlap | Difference and boundary for this paper |
|---|---|---|---|
| Karimov, Imani & Kazakov, “Quantization Robustness to Input Degradations for Object Detection” (arXiv:2508.19600, v3, 2026) | [arXiv primary record](https://arxiv.org/abs/2508.19600); title/authors/abstract retrieved. The record describes YOLO scale/precision comparisons and degradation-aware calibration. Peer-reviewed publication status was not established in this check. | Closest conceptual overlap: object-detector PTQ, multiple YOLO scales, degraded input robustness and alternative calibration distributions. Their abstract reports that degradation-aware calibration did not improve broadly/consistently, with model/condition exceptions. | Our locked evidence is CCTSDB traffic-sign detection and explicitly compares build variation plus selective head precision; it does not claim general degradation robustness or introduce a calibration algorithm. The two-model confirmation is pending and cannot yet support a cross-family conclusion. |
| Yifu Ding, Weilun Feng, Chuyan Chen, Jinyang Guo & Xianglong Liu, “Reg-PTQ: Regression-specialized Post-training Quantization for Fully Quantized Object Detector” (CVPR 2024) | [CVF Open Access primary paper](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html); landing page and author metadata retrieved (HTTP 200). DOI was not exposed in the page metadata inspected; use CVF citation/BibTeX at final bibliography check. | Directly establishes that regression branches in detectors can require quantization-specific attention; therefore localization sensitivity by itself is not novel. | Reg-PTQ proposes a quantization method for fully quantized detectors. This project’s current precision-head result is an empirical, architecture-specific TensorRT precision-constraint observation, not a competing PTQ algorithm and not a claim of W8A8 throughout the engine. |
| “Benchmarking the Reliability of Post-training Quantization: a Particular Focus on Worst-case Performance” (arXiv:2303.13003) | [arXiv primary record](https://arxiv.org/abs/2303.13003); title/abstract retrieved. Venue status is left unspecified here because this check did not verify the final proceedings record. | Prior work treats PTQ reliability and adverse-case performance as first-class outcomes. It cautions against reporting only a favorable mean. | We report observed engine/build ranges and size-stratified endpoints for a fixed task/checkpoint, but do not claim a general reliability framework or population-level worst-case guarantee. |
| Kim et al., “Inlier-Centric Post-Training Quantization for Object Detection Models” (InlierQ, arXiv:2602.03472, 2026) | [arXiv primary record](https://arxiv.org/abs/2602.03472); title/authors/abstract retrieved. Treat as a preprint unless a peer-reviewed venue is independently confirmed. | Recent detection PTQ research targets task-relevant activation information and label-free calibration, rather than assuming generic visual diversity is sufficient. | InlierQ is a method contribution using gradient-aware saliency/mixture modeling. Our work does not reproduce or compare against its algorithm; it motivates keeping proposed calibration-policy claims modest. |
| Bolya et al., “TIDE: A General Toolbox for Identifying Object Detection Errors” (ECCV 2020) | [arXiv primary record](https://arxiv.org/abs/2008.08115) retrieved; the project’s prior review identifies the ECCV 2020 publication. Verify the proceedings citation in the final bibliography before submission. | Provides a framework for separating detection error types and analyzing predictions, not only aggregate mAP. | We may use error decomposition as a diagnostic or cite its rationale, but neither the decomposition itself nor size-binned AP is presented as a new algorithm. Any TIDE-derived result must be separately rerun/reported; it is not inferred from current AP tables. |
| Jahanifar et al., “AgriJetsonBench: External-Power-Referenced TensorRT Benchmarking of Agricultural Vision Models on Jetson Edge Platforms” (arXiv:2608.00927, 2026) | [arXiv primary record](https://arxiv.org/abs/2608.00927); title/authors/abstract retrieved. Treat as a 2026 preprint absent a verified proceedings record. | Demonstrates current interest in explicit TensorRT timing boundaries, board-input energy and reproducibility for deployed vision workloads. | Different task/domain and hardware. Our YOLO11n latency artifact is a single shared RTX 8000 timing diagnostic without external energy measurement; it is not comparable as an energy or Jetson deployment result. |

## Positioning that the evidence can support

The defensible current contribution is an empirical measurement package: it
separates one YOLO11n head-precision discovery from repeated-build variation,
keeps COCO/XML and Ultralytics endpoints distinct, and records a bounded
cross-model confirmation protocol. It is not a new quantization algorithm,
not a proof that one branch is a causal mechanism, and not yet evidence that
the YOLO11n observation transfers to YOLOv8n or YOLO26n. The v2 study is
operator-pending.

The closest prior art makes broad “first to quantize traffic-sign detectors,”
“first regression-aware detector PTQ,” or “first robustness-aware calibration”
claims unsafe without a substantially broader search and stronger novelty
argument. More defensible novelty, if supported after confirmation, would be the
carefully controlled cross-architecture characterization of *when* precision
constraints alter task AP and build variability for fixed traffic-sign
checkpoints. Do not claim that this is the first such study until systematic
database searches and coauthor review are complete.

## Reference records to finalize

1. Karimov, T., Imani, H., and Kazakov, A. *Quantization Robustness to Input
   Degradations for Object Detection*. arXiv:2508.19600, version 3 (2026-05-01).
   https://arxiv.org/abs/2508.19600
2. Ding, Y., Feng, W., Chen, C., Guo, J., and Liu, X. *Reg-PTQ:
   Regression-specialized Post-training Quantization for Fully Quantized
   Object Detector*. CVPR 2024. Use the CVF page's final BibTeX in the
   bibliography; DOI was not exposed in the inspected metadata.
3. *Benchmarking the Reliability of Post-training Quantization: a Particular
   Focus on Worst-case Performance*. arXiv:2303.13003. Confirm final venue and
   DOI, if any, before final submission.
4. Kim, M. et al. *Inlier-Centric Post-Training Quantization for Object
   Detection Models*. arXiv:2602.03472 (2026). Preprint status as checked.
5. Bolya, D. et al. *TIDE: A General Toolbox for Identifying Object Detection
   Errors*. ECCV 2020; arXiv:2008.08115. Confirm proceedings DOI/pages.
6. Jahanifar, H. et al. *AgriJetsonBench: External-Power-Referenced TensorRT
   Benchmarking of Agricultural Vision Models on Jetson Edge Platforms*.
   arXiv:2608.00927 (2026). Preprint status as checked.

The source pages for Springer and MDPI journal records were not both
retrievable in this session (Springer returned a client challenge; MDPI
returned HTTP 403). Journal statements in the readiness checklist therefore
retain their source and verification caveats rather than treating network
failure as confirmation.
