# Precision-head calibration order CPU audit

No TensorRT build, model forward, calibration cache operation, or CUDA allocation was run.
The audit verifies the actual pinned Ultralytics dataset/dataloader path over every accepted image in U42/U43/U44.
Local label files have CRLF/LF byte differences from the Linux source inventory; all normalize to the accepted LF hashes, and labels are not passed as calibration tensors.

| Selection | Expected | Yielded | Validated | IDs match | Materialization |
|---|---:|---:|---:|---|---|
| U42 | 1024 | 1024 | 1024 | True | isolated_temporary_materialization_from_hash_verified_local_source |
| U43 | 1024 | 1024 | 1024 | True | isolated_temporary_materialization_from_hash_verified_local_source |
| U44 | 1024 | 1024 | 1024 | True | isolated_temporary_materialization_from_hash_verified_local_source |

Ultralytics source fingerprints: `{"build_dataloader": "3526a3fbca169de6bdfde5e6128a5be128d16135725458475728a3d145a65164", "build_yolo_dataset": "cc30939a9a049a201481a43b84f1b41eea32f0e68bfe57aba3bbada5c9433e49", "check_det_dataset": "180a0628c120d35a930fb2ba7681c6bc646658ab7555e66357aa611c53d04ee2", "exporter_calibration_method": "c3083ef89201276d5ea73e217846a0ddcaeed92b4909f75a2545c4ef2016b44c"}`
