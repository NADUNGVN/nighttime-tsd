#!/usr/bin/env python3
"""Pre-observation comparator for the YOLO11n E2 correctness smoke.

The comparator is deliberately dependency-free. It compares the native raw
``output0`` tensor before decode/NMS and records compute mode separately from
the TensorRT binding/storage dtype.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

OUTPUT_SHAPE = (1, 7, 8400)
OUTPUT_ELEMENTS = OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2]
CLASS_ORDER = ("prohibitory", "mandatory", "warning")


@dataclass(frozen=True)
class DomainTolerance:
    absolute: float
    relative: float


@dataclass(frozen=True)
class ComparisonPolicy:
    source_compute_mode: str = "fp32_reference"
    target_compute_mode: str = "fp16"
    source_dtype: str = "float32"
    target_binding_dtype: str = "float32"
    box_tolerance: DomainTolerance = DomainTolerance(absolute=5e-3, relative=1e-2)
    score_tolerance: DomainTolerance = DomainTolerance(absolute=2e-3, relative=1e-2)
    class_order: tuple[str, ...] = CLASS_ORDER
    coordinate_convention: str = "xywh in pixels of the 640x640 letterboxed input; channel order x,y,w,h"
    mismatch_limit: int = 20

    def as_dict(self) -> dict[str, object]:
        return {
            "source_compute_mode": self.source_compute_mode,
            "target_compute_mode": self.target_compute_mode,
            "source_dtype": self.source_dtype,
            "target_binding_dtype": self.target_binding_dtype,
            "tolerance_equation": "abs(reference-target) <= absolute + relative*abs(reference)",
            "box_tolerance": {"absolute": self.box_tolerance.absolute, "relative": self.box_tolerance.relative},
            "score_tolerance": {"absolute": self.score_tolerance.absolute, "relative": self.score_tolerance.relative},
            "class_order": list(self.class_order),
            "coordinate_convention": self.coordinate_convention,
            "comparison_stage": "native output0 before decode/NMS",
        }


@dataclass(frozen=True)
class Mismatch:
    flat_index: int
    channel: int
    channel_kind: str
    reference: float | None
    target: float | None
    absolute_error: float | None
    relative_error: float | None
    allowed_error: float | None
    reason: str | None = None


@dataclass
class ComparisonResult:
    passed: bool
    compared_elements: int
    mismatch_count: int
    nonfinite_reference: int
    nonfinite_target: int
    max_absolute_error: float | None
    max_relative_error: float | None
    domain_summary: dict[str, dict[str, object]]
    mismatches: list[Mismatch] = field(default_factory=list)
    policy: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "pass" if self.passed else "fail",
            "passed": self.passed,
            "compared_elements": self.compared_elements,
            "mismatch_count": self.mismatch_count,
            "nonfinite_reference": self.nonfinite_reference,
            "nonfinite_target": self.nonfinite_target,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "domain_summary": self.domain_summary,
            "mismatches": [m.__dict__ for m in self.mismatches],
            "policy": self.policy,
        }


def _validate_policy(policy: ComparisonPolicy) -> None:
    if policy.class_order != CLASS_ORDER:
        raise ValueError("CLASS_ORDER_MISMATCH: expected pinned three-class order")
    if policy.mismatch_limit < 0:
        raise ValueError("MISMATCH_LIMIT_INVALID")
    for name, tolerance in (("box", policy.box_tolerance), ("score", policy.score_tolerance)):
        if not math.isfinite(tolerance.absolute) or not math.isfinite(tolerance.relative) or tolerance.absolute < 0 or tolerance.relative < 0:
            raise ValueError(f"{name.upper()}_TOLERANCE_INVALID")


def compare_output0(reference: Sequence[float], target: Sequence[float], policy: ComparisonPolicy | None = None) -> ComparisonResult:
    """Compare two flattened native output tensors with fixed pre-observation policy."""
    policy = policy or ComparisonPolicy()
    _validate_policy(policy)
    if len(reference) != OUTPUT_ELEMENTS or len(target) != OUTPUT_ELEMENTS:
        raise ValueError(f"OUTPUT_SHAPE_MISMATCH: expected {OUTPUT_SHAPE} ({OUTPUT_ELEMENTS} elements)")

    nonfinite_reference = sum(not math.isfinite(float(value)) for value in reference)
    nonfinite_target = sum(not math.isfinite(float(value)) for value in target)
    domain_summary: dict[str, dict[str, object]] = {
        "boxes": {"elements": 4 * OUTPUT_SHAPE[2], "mismatches": 0, "nonfinite": 0, "max_absolute_error": 0.0, "max_relative_error": 0.0},
        "scores": {"elements": 3 * OUTPUT_SHAPE[2], "mismatches": 0, "nonfinite": 0, "max_absolute_error": 0.0, "max_relative_error": 0.0},
    }
    mismatches: list[Mismatch] = []
    mismatch_count = 0
    max_absolute_error: float | None = 0.0
    max_relative_error: float | None = 0.0
    for flat_index, (reference_value, target_value) in enumerate(zip(reference, target)):
        reference_float = float(reference_value)
        target_float = float(target_value)
        channel = flat_index // OUTPUT_SHAPE[2]
        kind = "boxes" if channel < 4 else "scores"
        tolerance = policy.box_tolerance if kind == "boxes" else policy.score_tolerance
        if not (math.isfinite(reference_float) and math.isfinite(target_float)):
            absolute_error = None
            relative_error = None
            allowed_error = None
            reason = "+".join(name for name, value in (("NONFINITE_REFERENCE", reference_float), ("NONFINITE_TARGET", target_float)) if not math.isfinite(value))
            mismatch = True
        else:
            absolute_error = abs(reference_float - target_float)
            relative_error = absolute_error / max(abs(reference_float), 1e-12)
            allowed_error = tolerance.absolute + tolerance.relative * abs(reference_float)
            reason = None
            mismatch = absolute_error > allowed_error
        domain = domain_summary[kind]
        if absolute_error is None:
            domain["nonfinite"] = int(domain["nonfinite"]) + 1
        else:
            max_absolute_error = max(float(max_absolute_error), absolute_error)
            max_relative_error = max(float(max_relative_error), relative_error or 0.0)
            domain["max_absolute_error"] = max(float(domain["max_absolute_error"]), absolute_error)
            domain["max_relative_error"] = max(float(domain["max_relative_error"]), relative_error or 0.0)
        if mismatch:
            mismatch_count += 1
            domain["mismatches"] = int(domain["mismatches"]) + 1
            if len(mismatches) < policy.mismatch_limit:
                mismatches.append(Mismatch(flat_index, channel, kind, reference_float if math.isfinite(reference_float) else None, target_float if math.isfinite(target_float) else None, absolute_error, relative_error, allowed_error, reason))

    passed = nonfinite_reference == 0 and nonfinite_target == 0 and mismatch_count == 0
    return ComparisonResult(
        passed=passed,
        compared_elements=OUTPUT_ELEMENTS,
        mismatch_count=mismatch_count,
        nonfinite_reference=nonfinite_reference,
        nonfinite_target=nonfinite_target,
        max_absolute_error=max_absolute_error,
        max_relative_error=max_relative_error,
        domain_summary=domain_summary,
        mismatches=mismatches,
        policy=policy.as_dict(),
    )
