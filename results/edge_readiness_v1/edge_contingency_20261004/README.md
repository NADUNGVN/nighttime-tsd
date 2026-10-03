# Bounded read-only alternative-device inventory

Collected `2026-10-03T18:19:39Z` (2026-10-04 Asia/Saigon) under E2L1-031: one bounded, host-key-verified SSH inventory for each approved online alias `pi5`, `rubik` and `nano`. Each transport exited 0 and each fixed collector recorded 27/27 command markers. This establishes current reachability and observed installed state only; it does not establish model compatibility, idle conditions, performance or study authorization.

The remote queries read model/OS/CPU, memory/storage, installed runtime/package versions, a short process-name/CPU/memory snapshot, available thermal/power telemetry sources and current power/clock readouts where available. There was no network scan, retry, installation, accelerator/model load, build, inference, benchmark, process control or configuration change. Raw captures remain local and outside Git. Public artifacts are sanitized allowlisted summaries; hostname and SSH route are omitted, serial/MAC/IP identifiers are redacted, and per-command raw-output hashes are retained.

| Target | Exact observed model | OS / Python | Backend and libraries observed | Available RAM / root free | Readiness note |
| --- | --- | --- | --- | --- | --- |
| E1 | Raspberry Pi 5 Model B Rev 1.1; Hailo-8 identified | Debian 13; Python 3.13.5 | HailoRT CLI 4.23.0, `hailo-all` 5.1.1; NumPy 2.2.4; OpenCV 4.10.0 | 7.3 GiB / 2.1 GiB | Hailo backend only; root 93% used; no power measurement boundary established. |
| E4 | Thundercomm RUBIK Pi 3 | Ubuntu 24.04.4; Python 3.12.3 | QAIRT libs/DSP/tools 2.46.0; NumPy 1.26.4; OpenCV 4.6.0; no TensorRT/CUDA packages observed | 6.3 GiB / 26 GiB | Distinct Qualcomm backend; CLI/model compatibility and energy boundary unverified. |
| E5 | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super | Ubuntu 22.04.5, L4T R36.5.2; Python 3.10.12 | CUDA 12.6.68, driver 540.5.0, TensorRT 10.3.0.30, `trtexec` v100300; NumPy 1.21.5; OpenCV import missing | 5.3 GiB / 792 GiB | Closest TensorRT candidate, but current concrete CUDA owner is CUDA-11.4-specific and no real E5 adapter/engine validation exists. |

Current E5 mode was reported as `MAXN_SUPER`; this was observed, not set. Current thermal reads (CPU/GPU about 51.2°C on E5) and the single `ps` sample are not controlled idle or benchmark evidence. E1 exposed `vcgencmd`; E4 exposed `sensors` and powercap/IIO paths; E5 exposed `tegrastats` and IIO. Source presence alone does not establish calibrated watts/joules or a shared measurement boundary.

## Sanitized artifacts

The `inventory_sanitized_v2.json` files below are authoritative. The sanitizer retains each command's status, sanitized output, output byte count and SHA-256 of the original command output. Raw captures and route details are not included.

| File | SHA-256 |
| --- | --- |
| `E1/inventory_sanitized_v2.json` | `8f367f65106d31359c914cae1c9f88e1194690c1bf95d0fa728eaf93aa5f7b6b` |
| `E4/inventory_sanitized_v2.json` | `a2aa9d1cbf83ab2b13493da13d38d7345e11529c3b40ec125a527f1b89c63ad3` |
| `E5/inventory_sanitized_v2.json` | `e56a570f38e97ee8ce313b2dad5155da9d3a441943b88ed7057df550002a45c1` |

Collector source SHA-256: `ff756bae1c3a03189c47ee93fcd50942d7fcc5f653acca7b2db4c862d5dc5ba5`. Sanitizer source SHA-256: `3b8d5a15b299f940f7267e00b12b9feabc91c775dffd44e8ff166fa4cd116e1f`. Collector/sanitizer tests: [`tests/test_edge_readiness.py`](../../../tests/test_edge_readiness.py), 13/13 pass in Python 3.10.

## Recommendation

Keep E2 as the primary target. E5 is a possible separately reviewed TensorRT path, not a replacement: its API profile exists in the adapter, but its concrete CUDA 12.6 library path, OpenCV preprocessing dependency and E5-specific provider/cleanup tests remain open. E1/Hailo and E4/QAIRT are distinct backends and remain readiness-only. No model calls or device-side modifications are authorized by this inventory.
