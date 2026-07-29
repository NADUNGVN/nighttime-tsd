# Domain-aware INT8 traffic-sign detection on edge

The active research line measures how TensorRT precision and PTQ calibration
change lightweight traffic-sign detection across CCTSDB2021 weather/light
domains, then measures latency and energy on Jetson AGX Orin.

The repository contains one clean CCTSDB pipeline. Historical models, results,
datasets, and scripts were intentionally removed because they were either
unreproducible or used an incompatible class mapping.

Start with [the protocol](docs/PROTOCOL.md). The official CCTSDB2021 release
contains 16,356 training images and 1,500 positive official-test images; use
the dataset under its GPL-3.0 terms and cite its benchmark paper.
