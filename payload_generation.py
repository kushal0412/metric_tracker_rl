"""Shared method registry and submission preview helpers."""

from __future__ import annotations

from typing import Any

try:
    from .analysis_tools import (
        SharedAnalysisToolkit,
        available_analysis_methods,
        preview_submission_rows,
        submission_row_key,
    )
    from .models import MetricSubmissionRow, SubmissionPreview
    from .server.data_generator import available_synthetic_generator_methods
except ImportError:
    from analysis_tools import (
        SharedAnalysisToolkit,
        available_analysis_methods,
        preview_submission_rows,
        submission_row_key,
    )
    from models import MetricSubmissionRow, SubmissionPreview
    from server.data_generator import available_synthetic_generator_methods


def available_payload_generation_methods():
    """Backward-compatible alias for the shared analysis method list."""
    return available_analysis_methods()


def preview_submission(
    rows: list[MetricSubmissionRow] | list[dict[str, Any]],
) -> SubmissionPreview:
    """Validate a submission without using hidden labels."""
    return preview_submission_rows(rows)


__all__ = [
    "SharedAnalysisToolkit",
    "available_analysis_methods",
    "available_payload_generation_methods",
    "available_synthetic_generator_methods",
    "preview_submission",
    "submission_row_key",
]
