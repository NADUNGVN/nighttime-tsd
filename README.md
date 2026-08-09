# Domain-aware INT8 traffic-sign detection on edge

The active research line is a locked Image and Vision Computing (IVC) study:
it measures how a train-only luminance-stratified PTQ calibration policy
changes domain-wise INT8 traffic-sign detection across 15 YOLO models and five
edge execution configurations. The study uses CCTSDB2021 only; TT100K, MTSD,
and CURE-TSD are out of scope for this manuscript.

The repository contains a clean CCTSDB benchmark pipeline and controlled
intake scaffolding for TT100K, MTSD, and CURE-TSD. Historical models, results,
datasets, and scripts were intentionally removed because they were either
unreproducible or used an incompatible class mapping.

Start with [the immutable CCTSDB protocol](docs/PROTOCOL.md) and the
[locked IVC study](docs/IVC_STUDY_V1.md). The official CCTSDB2021 release
contains 16,356 training images and 1,500 positive official-test images; use
the dataset under its GPL-3.0 terms and cite its benchmark paper.
