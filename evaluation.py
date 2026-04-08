"""Deterministic grading for the metric tracker RL environment."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .analysis_tools import preview_submission_rows, submission_row_key
    from .models import MetricSubmissionRow, RewardBreakdown, SubmissionIssue, SubmissionPreview
except ImportError:
    from analysis_tools import preview_submission_rows, submission_row_key
    from models import MetricSubmissionRow, RewardBreakdown, SubmissionIssue, SubmissionPreview

SCORE_EPSILON = 0.000001


@dataclass(frozen=True)
class EvaluationConfig:
    """Tunable parameters for deterministic scoring."""

    value_tolerance: float = 0.06
    delta_tolerance: float = 0.06
    precision_weight: float = 0.30
    recall_weight: float = 0.30
    anomaly_type_weight: float = 0.12
    detection_method_weight: float = 0.10
    value_weight: float = 0.12
    severity_weight: float = 0.06
    extra_row_penalty: float = 0.03
    duplicate_row_penalty: float = 0.04
    invalid_row_penalty: float = 0.05
    exploit_row_multiplier: float = 3.0
    exploit_penalty: float = 0.15


@dataclass
class EvaluationResult:
    """Complete scoring result."""

    preview: SubmissionPreview
    issues: list[SubmissionIssue]
    reward_breakdown: RewardBreakdown
    matched_rows: int
    is_perfect: bool


def _bounded_total_score(score: float) -> float:
    """Clamp evaluator scores to the open interval (0, 1)."""
    rounded_score = round(score, 6)
    return min(1.0 - SCORE_EPSILON, max(SCORE_EPSILON, rounded_score))


def evaluate_submission(
    submitted_rows: list[dict] | list[MetricSubmissionRow],
    expected_rows: list[MetricSubmissionRow],
    config: EvaluationConfig | None = None,
    *,
    include_debug_expected: bool = False,
) -> EvaluationResult:
    """Grade one submission against deterministic expectations."""
    cfg = config or EvaluationConfig()
    preview = preview_submission_rows(submitted_rows)
    expected_map = {submission_row_key(row): row for row in expected_rows}
    submitted_map = {submission_row_key(row): row for row in preview.normalized_rows}

    issues = list(preview.issues)
    matched_keys = [key for key in submitted_map if key in expected_map]
    extra_keys = [key for key in submitted_map if key not in expected_map]
    missing_keys = [key for key in expected_map if key not in submitted_map]

    anomaly_type_hits = 0
    detection_method_hits = 0
    value_hits = 0.0
    severity_hits = 0

    for key in matched_keys:
        submitted = submitted_map[key]
        expected = expected_map[key]
        field_issues = _field_issues(submitted, expected, cfg, include_debug_expected)
        issues.extend(field_issues)
        if submitted.anomaly_type == expected.anomaly_type:
            anomaly_type_hits += 1
        if submitted.detection_method == expected.detection_method:
            detection_method_hits += 1
        value_hits += _value_match_score(submitted, expected, cfg)
        if submitted.severity == expected.severity:
            severity_hits += 1

    for key in extra_keys:
        submitted = submitted_map[key]
        issues.append(
            SubmissionIssue(
                row_key=key,
                issue_type="extra_row",
                message="Row is not expected for this episode.",
                submitted_row=submitted.model_dump(),
                expected_row=None,
            )
        )

    for key in missing_keys:
        expected = expected_map[key]
        issues.append(
            SubmissionIssue(
                row_key=key,
                issue_type="missing_row",
                message="Expected anomaly row is missing from the submission.",
                submitted_row=None,
                expected_row=expected.model_dump() if include_debug_expected else None,
            )
        )

    valid_submitted = len(preview.normalized_rows)
    matched_count = len(matched_keys)
    expected_count = len(expected_rows)
    precision = matched_count / valid_submitted if valid_submitted else 0.0
    recall = matched_count / expected_count if expected_count else 1.0
    denominator = max(matched_count, 1)
    anomaly_type_accuracy = anomaly_type_hits / denominator if matched_count else 0.0
    detection_method_accuracy = detection_method_hits / denominator if matched_count else 0.0
    value_accuracy = value_hits / denominator if matched_count else 0.0
    severity_accuracy = severity_hits / denominator if matched_count else 0.0

    extra_penalty = min(0.5, len(extra_keys) * cfg.extra_row_penalty)
    duplicate_penalty = min(0.4, preview.duplicate_rows * cfg.duplicate_row_penalty)
    invalid_penalty = min(0.4, preview.invalid_rows * cfg.invalid_row_penalty)
    exploit_penalty = 0.0
    exploit_limit = max(6, int(expected_count * cfg.exploit_row_multiplier))
    if valid_submitted > exploit_limit:
        exploit_penalty = cfg.exploit_penalty

    total_score = (
        precision * cfg.precision_weight
        + recall * cfg.recall_weight
        + anomaly_type_accuracy * cfg.anomaly_type_weight
        + detection_method_accuracy * cfg.detection_method_weight
        + value_accuracy * cfg.value_weight
        + severity_accuracy * cfg.severity_weight
        - extra_penalty
        - duplicate_penalty
        - invalid_penalty
        - exploit_penalty
    )
    total_score = _bounded_total_score(total_score)

    breakdown = RewardBreakdown(
        precision=round(precision, 6),
        recall=round(recall, 6),
        anomaly_type_accuracy=round(anomaly_type_accuracy, 6),
        detection_method_accuracy=round(detection_method_accuracy, 6),
        value_accuracy=round(value_accuracy, 6),
        severity_accuracy=round(severity_accuracy, 6),
        extra_row_penalty=round(extra_penalty, 6),
        duplicate_penalty=round(duplicate_penalty, 6),
        invalid_row_penalty=round(invalid_penalty, 6),
        exploit_penalty=round(exploit_penalty, 6),
        total_score=total_score,
        matched_rows=matched_count,
        expected_rows=expected_count,
        submitted_rows=len(submitted_rows),
        valid_submitted_rows=valid_submitted,
        extra_rows=len(extra_keys),
        duplicate_rows=preview.duplicate_rows,
        invalid_rows=preview.invalid_rows,
        missing_rows=len(missing_keys),
    )
    is_perfect = total_score >= 0.999999 and not issues
    return EvaluationResult(
        preview=preview,
        issues=issues,
        reward_breakdown=breakdown,
        matched_rows=matched_count,
        is_perfect=is_perfect,
    )


def _field_issues(
    submitted: MetricSubmissionRow,
    expected: MetricSubmissionRow,
    cfg: EvaluationConfig,
    include_debug_expected: bool,
) -> list[SubmissionIssue]:
    issues: list[SubmissionIssue] = []
    row_key = submission_row_key(expected)
    expected_dump = expected.model_dump() if include_debug_expected else None
    if submitted.anomaly_type != expected.anomaly_type:
        issues.append(
            SubmissionIssue(
                row_key=row_key,
                issue_type="wrong_anomaly_type",
                message=f"Expected anomaly_type={expected.anomaly_type}.",
                submitted_row=submitted.model_dump(),
                expected_row=expected_dump,
            )
        )
    if submitted.detection_method != expected.detection_method:
        issues.append(
            SubmissionIssue(
                row_key=row_key,
                issue_type="wrong_detection_method",
                message=f"Expected detection_method={expected.detection_method}.",
                submitted_row=submitted.model_dump(),
                expected_row=expected_dump,
            )
        )
    if not _close(submitted.baseline_value, expected.baseline_value, cfg.value_tolerance):
        issues.append(
            SubmissionIssue(
                row_key=row_key,
                issue_type="wrong_baseline_value",
                message="Baseline value is outside tolerance.",
                submitted_row=submitted.model_dump(),
                expected_row=expected_dump,
            )
        )
    if not _close(submitted.observed_value, expected.observed_value, cfg.value_tolerance):
        issues.append(
            SubmissionIssue(
                row_key=row_key,
                issue_type="wrong_observed_value",
                message="Observed value is outside tolerance.",
                submitted_row=submitted.model_dump(),
                expected_row=expected_dump,
            )
        )
    if not _close(submitted.delta_value, expected.delta_value, cfg.delta_tolerance):
        issues.append(
            SubmissionIssue(
                row_key=row_key,
                issue_type="wrong_delta_value",
                message="Delta value is outside tolerance.",
                submitted_row=submitted.model_dump(),
                expected_row=expected_dump,
            )
        )
    if submitted.severity != expected.severity:
        issues.append(
            SubmissionIssue(
                row_key=row_key,
                issue_type="wrong_severity",
                message=f"Expected severity={expected.severity}.",
                submitted_row=submitted.model_dump(),
                expected_row=expected_dump,
            )
        )
    return issues


def _value_match_score(
    submitted: MetricSubmissionRow,
    expected: MetricSubmissionRow,
    cfg: EvaluationConfig,
) -> float:
    checks = [
        _close(submitted.baseline_value, expected.baseline_value, cfg.value_tolerance),
        _close(submitted.observed_value, expected.observed_value, cfg.value_tolerance),
        _close(submitted.delta_value, expected.delta_value, cfg.delta_tolerance),
    ]
    return sum(1.0 for ok in checks if ok) / len(checks)


def _close(submitted: float, expected: float, tolerance: float) -> bool:
    allowed = max(tolerance, abs(expected) * tolerance)
    return abs(submitted - expected) <= allowed
