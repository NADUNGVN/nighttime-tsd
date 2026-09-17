"""Python 3.8-compatible stable errors for the edge boundary."""
from __future__ import annotations

from typing import Any, Dict, Optional


class AdapterError(ValueError):
    """Stable machine-readable boundary error."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__("{}: {}".format(code, message))
        self.code = code
        self.message = message
        self.details = details or {}

    def as_dict(self) -> Dict[str, Any]:
        return {"status": "error", "error": {"code": self.code, "message": self.message, "details": self.details}}
