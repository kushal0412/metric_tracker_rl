"""Data models for the metric tracker RL environment."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


class MetricRecord(BaseModel):
    """Hourly or daily aggregate metrics for the app funnel."""

    date: str = Field(..., description="ISO date in YYYY-MM-DD format.")
    hour: int | None = Field(
        default=None,
        description="Hour bucket in 24h format. Null for daily aggregates.",
    )
    app_opens: int = Field(default=0, description="Count of app_open events.")
    menu_opens: int = Field(default=0, description="Count of menu_open events.")
    product_added_to_cart: int = Field(
        default=0,
        description="Count of product_added_to_cart events.",
    )
    orders_placed: int = Field(default=0, description="Count of order_placed events.")
    payment_successful: int = Field(
        default=0,
        description="Count of payment_successful events.",
    )


class ConversionMetricDefinition(BaseModel):
    """Definition for a conversion metric that the agent can cite."""

    name: str = Field(..., description="Stable conversion metric identifier.")
    numerator: str = Field(..., description="Numerator event.")
    denominator: str = Field(..., description="Denominator event.")
    description: str = Field(..., description="Human-readable formula.")


class MethodSpec(BaseModel):
    """Description of a shared safe analysis method."""

    name: str = Field(..., description="Method name.")
    description: str = Field(..., description="What the method does.")
    parameters: list[str] = Field(
        default_factory=list,
        description="Ordered parameter names for the method.",
    )


class MetricSubmissionRow(BaseModel):
    """Submitted anomaly row."""

    date: str = Field(..., description="ISO date in YYYY-MM-DD format.")
    entity_type: str = Field(
        ...,
        description=(
            "Stable entity family such as conversion_rate, event_count, funnel_step, "
            "hourly_mix, or data_quality."
        ),
    )
    entity_name: str = Field(..., description="Stable entity identifier.")
    anomaly_type: str = Field(..., description="Stable anomaly type identifier.")
    detection_method: str = Field(..., description="Shared analysis method used.")
    baseline_value: float = Field(..., description="Reference baseline value.")
    observed_value: float = Field(..., description="Observed anomalous value.")
    delta_value: float = Field(..., description="Observed minus baseline.")
    severity: Literal["low", "medium", "high", "critical"] = Field(
        ...,
        description="Severity label.",
    )


class PayloadGeneratorMethod(BaseModel):
    """A declarative payload generation method."""

    method_name: str = Field(
        ...,
        description="Generator method name, for example get_median_filter_rows.",
    )
    metric_name: str | None = Field(
        default=None,
        description="Single count metric or conversion metric name. Optional.",
    )
    metric_names: list[str] = Field(
        default_factory=list,
        description="Optional list of metrics to run. Empty means all metrics.",
    )
    threshold_multiplier: float = Field(
        ...,
        description="Multiplier applied to the metric std-from-median value.",
    )


class SyntheticAnomalyGenerator(BaseModel):
    """A declarative reset-time synthetic anomaly generator."""

    method_name: str = Field(
        default="metric_stddev_shift",
        description="Synthetic generator method name.",
    )
    metric_name: str | None = Field(
        default=None,
        description="Single count metric or conversion metric name. Optional.",
    )
    metric_names: list[str] = Field(
        default_factory=list,
        description="Optional list of metrics to generate on. Empty means use metric_name.",
    )
    date: str | None = Field(
        default=None,
        description="Single ISO date to inject on. Optional.",
    )
    dates: list[str] = Field(
        default_factory=list,
        description="Optional list of ISO dates to inject on.",
    )
    stddev_factor: float = Field(
        default=2.0,
        description="Multiplier applied to std_dev_from_median when creating the target value.",
    )
    direction: Literal["up", "down", "auto"] = Field(
        default="auto",
        description="Whether to shift the metric upward or downward.",
    )


class SyntheticGeneratorApplication(BaseModel):
    """Resolved synthetic generator application used for the active episode."""

    method_name: str = Field(..., description="Synthetic generator method used.")
    date: str = Field(..., description="ISO date the generator was applied to.")
    metric_name: str = Field(..., description="Metric name used by the generator.")
    metric_type: Literal["event_count", "conversion_rate"] = Field(
        ...,
        description="Resolved metric family.",
    )
    direction: Literal["up", "down"] = Field(..., description="Resolved direction.")
    anomaly_type: str = Field(..., description="Expected anomaly type generated.")
    detection_method: str = Field(..., description="Shared analysis method that should detect it.")
    baseline_value: float = Field(..., description="Median baseline used during generation.")
    pre_applied_value: float = Field(..., description="Metric value before generation.")
    std_dev_from_median: float = Field(..., description="Std-from-median used during generation.")
    stddev_factor: float = Field(..., description="Configured stddev factor.")
    threshold_value: float = Field(..., description="stddev_factor * std_dev_from_median.")
    target_value: float = Field(..., description="Requested target value before rebalancing.")
    actual_value: float = Field(..., description="Observed value after generation.")
    formula: str = Field(..., description="Human-readable formula used for generation.")


class SubmissionIssue(BaseModel):
    """Feedback about a submitted row or missing expectation."""

    row_key: str = Field(..., description="Stable key in date|entity_type|entity_name form.")
    issue_type: str = Field(..., description="Issue class.")
    message: str = Field(..., description="Human-readable explanation.")
    submitted_row: dict[str, Any] | None = Field(
        default=None,
        description="Submitted row fragment when relevant.",
    )
    expected_row: dict[str, Any] | None = Field(
        default=None,
        description="Expected row fragment when debug is enabled.",
    )


class RewardBreakdown(BaseModel):
    """Deterministic grading components."""

    precision: float = 0.0
    recall: float = 0.0
    anomaly_type_accuracy: float = 0.0
    detection_method_accuracy: float = 0.0
    value_accuracy: float = 0.0
    severity_accuracy: float = 0.0
    extra_row_penalty: float = 0.0
    duplicate_penalty: float = 0.0
    invalid_row_penalty: float = 0.0
    exploit_penalty: float = 0.0
    total_score: float = 0.0
    matched_rows: int = 0
    expected_rows: int = 0
    submitted_rows: int = 0
    valid_submitted_rows: int = 0
    extra_rows: int = 0
    duplicate_rows: int = 0
    invalid_rows: int = 0
    missing_rows: int = 0


class SubmissionPreview(BaseModel):
    """Safe preview of a candidate submission before grading."""

    valid_rows: int = 0
    invalid_rows: int = 0
    duplicate_rows: int = 0
    unique_keys: int = 0
    issues: list[SubmissionIssue] = Field(default_factory=list)
    normalized_rows: list[MetricSubmissionRow] = Field(default_factory=list)


class BenchmarkTaskSpec(BaseModel):
    """Public metadata for a benchmark task."""

    task_id: str = Field(..., description="Stable benchmark task identifier.")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ...,
        description="Canonical task difficulty.",
    )
    instruction: str = Field(..., description="Task instruction shown to the agent.")
    objective: str = Field(..., description="Concrete success objective.")
    scenario_family: str = Field(..., description="Scenario family used to generate the task episode.")
    anomaly_density: str = Field(..., description="Relative anomaly density for the task episode.")
    anomaly_count: int = Field(..., description="Number of anomalous rows expected for the task.")
    grader_name: str = Field(..., description="Programmatic grader used for the task.")


class MetricTrackerRlAction(Action):
    """Submitted anomaly payload for the current episode."""

    classifications: list[MetricSubmissionRow] = Field(
        default_factory=list,
        description="Submitted anomaly rows for the dataset.",
    )
    analysis_method: str | None = Field(
        default=None,
        description="Optional shared analysis method to call instead of grading a submission.",
    )
    analysis_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the selected analysis method.",
    )
    payload_generators: list[PayloadGeneratorMethod] = Field(
        default_factory=list,
        description="Declarative payload generation methods to run inside the environment.",
    )


class MetricTrackerRlObservation(Observation):
    """Observation containing the dataset and analysis surface."""

    task_id: str = Field(
        default="",
        description="Stable identifier for the active benchmark task.",
    )
    status: str = Field(
        default="ready",
        description="Episode status: ready, in_progress, evaluated, or completed.",
    )
    message: str = Field(default="", description="Human-readable environment feedback.")
    instruction: str = Field(
        default="",
        description="Task presented to the model for the current episode.",
    )
    conversion_metric_definitions: list[ConversionMetricDefinition] = Field(
        default_factory=list,
        description="Conversion formulas the model may cite.",
    )
    available_synthetic_generator_methods: list[MethodSpec] = Field(
        default_factory=list,
        description="Reset-time synthetic generator methods available for seeded data creation.",
    )
    applied_synthetic_generators: list[SyntheticGeneratorApplication] = Field(
        default_factory=list,
        description="Resolved synthetic generator applications used for the active episode.",
    )
    available_methods: list[MethodSpec] = Field(
        default_factory=list,
        description="Safe shared analysis methods available to agents and humans.",
    )
    available_tasks: list[BenchmarkTaskSpec] = Field(
        default_factory=list,
        description="Catalog of benchmark tasks available in this environment.",
    )
    daily_metrics: list[MetricRecord] = Field(
        default_factory=list,
        description="Deprecated raw daily data field. Kept empty in standard mode.",
    )
    hourly_metrics: list[MetricRecord] = Field(
        default_factory=list,
        description="Deprecated raw hourly data field. Kept empty in standard mode.",
    )
    analysis_result: dict[str, Any] | None = Field(
        default=None,
        description="Result of the latest analysis-method call.",
    )
    generated_rows: list[MetricSubmissionRow] = Field(
        default_factory=list,
        description="Rows generated from payload generator methods, if used.",
    )
    submitted_rows: list[MetricSubmissionRow] = Field(
        default_factory=list,
        description="Most recent submitted anomaly rows.",
    )
    submission_preview: SubmissionPreview | None = Field(
        default=None,
        description="Safe preview information for the latest submitted payload.",
    )
    submission_issues: list[SubmissionIssue] = Field(
        default_factory=list,
        description="Feedback for the latest submitted payload.",
    )
    reward_breakdown: RewardBreakdown | None = Field(
        default=None,
        description="Deterministic reward components for the latest step.",
    )
    expected_row_count: int = Field(
        default=0,
        description="Number of expected anomaly rows in the current episode.",
    )
    correct_row_count: int = Field(
        default=0,
        description="Number of matched anomaly rows in the latest step.",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Episode configuration visible in standard mode.",
    )
    debug: dict[str, Any] | None = Field(
        default=None,
        description="Developer-only debug payload. Hidden in standard mode.",
    )
