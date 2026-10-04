# Related work and scope boundary

Source check performed 2026-10-04. This is a focused map, not an exhaustive
survey. A publisher/CVF/arXiv or institutional-repository source is linked
for each item. “Verified” below
means the primary landing page and bibliographic/title information were
retrieved or matched to the existing repository source record; it does not
imply that every result in the cited work was independently reproduced. This
pass re-read arXiv API metadata/abstracts and the CVF paper record. The DOI and
page range for Reg-PTQ were cross-checked against DOI-registry metadata.

| Work | Primary source / status | Relevant overlap | Difference and boundary for this paper |
|---|---|---|---|
| Karimov, Imani & Kazakov, “Quantization Robustness to Input Degradations for Object Detection” (arXiv:2508.19600, v3, 2026) | [arXiv primary record](https://arxiv.org/abs/2508.19600); title/authors/abstract retrieved. The record describes YOLO scale/precision comparisons and degradation-aware calibration. Peer-reviewed publication status was not established in this check. | Closest conceptual overlap: object-detector PTQ, multiple YOLO scales, degraded input robustness and alternative calibration distributions. Their abstract reports that degradation-aware calibration did not improve broadly/consistently, with model/condition exceptions. | Our locked evidence is CCTSDB traffic-sign detection and explicitly compares build variation plus selective head precision; it does not claim general degradation robustness or introduce a calibration algorithm. The two-model confirmation is pending and cannot yet support a cross-family conclusion. |
| Yifu Ding, Weilun Feng, Chuyan Chen, Jinyang Guo & Xianglong Liu, “Reg-PTQ: Regression-specialized Post-training Quantization for Fully Quantized Object Detector” (CVPR 2024) | [CVF Open Access primary paper](https://openaccess.thecvf.com/content/CVPR2024/html/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.html); page returned HTTP 200. DOI-registry metadata matches CVPR 2024, pp. 16174–16184, doi:10.1109/CVPR52733.2024.01531. | Directly establishes that regression branches in detectors can require quantization-specific attention; therefore localization sensitivity by itself is not novel. | Reg-PTQ proposes a quantization method for fully quantized detectors. This project’s completed precision-head evidence is an empirical YOLO11n TensorRT precision-constraint observation; cross-model confirmation remains pending. It is not a competing PTQ algorithm or a claim of W8A8 throughout the engine. |
| “Benchmarking the Reliability of Post-training Quantization: a Particular Focus on Worst-case Performance” (arXiv:2303.13003) | [arXiv primary record](https://arxiv.org/abs/2303.13003); title/abstract retrieved. Venue status is left unspecified here because this check did not verify the final proceedings record. | Prior work treats PTQ reliability and adverse-case performance as first-class outcomes. It cautions against reporting only a favorable mean. | We report observed engine/build ranges and size-stratified endpoints for a fixed task/checkpoint, but do not claim a general reliability framework or population-level worst-case guarantee. |
| Kim et al., “Inlier-Centric Post-Training Quantization for Object Detection Models” (InlierQ, arXiv:2602.03472, 2026) | [arXiv primary record](https://arxiv.org/abs/2602.03472); title/authors/abstract retrieved (v1, 2026-02-03). No peer-reviewed venue was confirmed in this source check. | The abstract describes task-irrelevant activation anomalies, gradient-aware volume saliency, inlier/anomaly separation and label-free calibration. | InlierQ is a method contribution using gradient-aware saliency and an EM-fitted score distribution. Our work does not reproduce or compare against its algorithm; it motivates keeping proposed calibration-policy claims modest. |
| Bolya et al., “TIDE: A General Toolbox for Identifying Object Detection Errors” (ECCV 2020) | [arXiv primary record](https://arxiv.org/abs/2008.08115) retrieved. DOI-registry metadata returns *Computer Vision – ECCV 2020*, pp. 558–573, DOI 10.1007/978-3-030-58580-8_33. The Springer chapter URL redirected through a cookie interstitial in this check. | Provides a framework for separating detection error types and analyzing predictions, not only aggregate mAP. | We may use error decomposition as a diagnostic or cite its rationale, but neither the decomposition itself nor size-binned AP is presented as a new algorithm. Any TIDE-derived result must be separately rerun/reported; it is not inferred from current AP tables. |
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
   Object Detector*. CVPR 2024, pp. 16174–16184,
   doi:10.1109/CVPR52733.2024.01531. The CVF page is the preferred final
   bibliographic source; DOI and pages also match DOI-registry metadata.
3. *Benchmarking the Reliability of Post-training Quantization: a Particular
   Focus on Worst-case Performance*. arXiv:2303.13003. Confirm final venue and
   DOI, if any, before final submission.
4. Kim, M. et al. *Inlier-Centric Post-Training Quantization for Object
   Detection Models*. arXiv:2602.03472 (2026). Preprint status as checked.
5. Bolya, D. et al. *TIDE: A General Toolbox for Identifying Object Detection
   Errors*. ECCV 2020, pp. 558–573, doi:10.1007/978-3-030-58580-8_33;
   arXiv:2008.08115. DOI-registry metadata checked; Springer full-page
   rendering was not independently inspected because of its cookie redirect.
6. Jahanifar, H. et al. *AgriJetsonBench: External-Power-Referenced TensorRT
   Benchmarking of Agricultural Vision Models on Jetson Edge Platforms*.
   arXiv:2608.00927 (2026). Preprint status as checked.
7. Zhang, J. et al. *CCTSDB 2021: A More Comprehensive Traffic Sign Detection
   Benchmark*. *Human-centric Computing and Information Sciences*, vol. 12,
   2022, doi:10.22967/HCIS.2022.12.023. Institutional repository metadata
   was retrieved from https://centaur.reading.ac.uk/106129/; also see the
   official release repository https://github.com/csust7zhangjm/CCTSDB2021.
8. Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context*. *Computer
   Vision – ECCV 2014*, pp. 740–755, 2014,
   doi:10.1007/978-3-319-10602-1_48. DOI-registry metadata checked.

Live source check on this pass: arXiv API and CVF returned HTTP 200; the
Springer scope endpoint returned HTTP 200 with a `cookies_not_supported`
error redirect, so page content was not validated; MDPI journal endpoints
returned HTTP 403. The Journal of Imaging JIF/quartile/rank figures in the
readiness note remain a prior-source report, not independently confirmed in
this pass. Treat both journals' current ranking system/year/category and
Journal of Imaging APC details as unresolved until the publisher or the
institutional library verifies them.

The CCTSDB2021 benchmark paper and release repository were identified through
the University of Reading repository record and the project’s existing
dataset-source record. The main manuscript now cites the benchmark paper;
confirm the exact archive/version, release terms and any required attribution
against the specific XML archive used. The COCO reference is for the
matching/evaluation convention; custom CCTSDB size bins are not presented as
native COCO thresholds.
