#!/usr/bin/env python3
"""Short-lived CUDA environment probe for the inference-repeat parent."""
from __future__ import annotations

import json


def main():
    try:
        from uniform_build_repeat import environment
        report = {"schema_version": 1, "status": "ok", "environment": environment()}
        print(json.dumps(report, allow_nan=False, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as error:
        # Keep the transport parseable even when the runtime probe fails. The
        # parent still rejects nonzero exit and never starts a capture.
        print(json.dumps({"schema_version": 1, "status": "error",
                          "error_type": type(error).__name__, "error": str(error)},
                         allow_nan=False, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
