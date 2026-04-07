"""Shared safe analysis methods for agents and the manual UI."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any

try:
    from .models import (
        ConversionMetricDefinition,
        MethodSpec,
        MetricRecord,
        MetricSubmissionRow,
        PayloadGeneratorMethod,
        SubmissionIssue,
        SubmissionPreview,
    )
except ImportError:
    from models import (
        ConversionMetricDefinition,
        MethodSpec,
        MetricRecord,
        MetricSubmissionRow,
        PayloadGeneratorMethod,
        SubmissionIssue,
        SubmissionPreview,
    )


FUNNEL_STEPS: tuple[tuple[str, str], ...] = (
    ("menu_opens", "app_opens"),
    ("product_added_to_cart", "menu_opens"),
    ("orders_placed", "product_added_to_cart"),
    ("payment_successful", "orders_placed"),
)

COUNT_METRICS: tuple[str, ...] = (
    "app_opens",
    "menu_opens",
    "product_added_to_cart",
    "orders_placed",
    "payment_successful",
)

DEFAULT_METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        name="task_overview",
        description="Return compact task context, config, entity catalog, and payload schema.",
    ),
    MethodSpec(name="list_dates", description="List all dates in the dataset."),
    MethodSpec(
        name="list_entities",
        description="List count, rate, funnel, hourly mix, and data quality entities.",
    ),
    MethodSpec(
        name="rows_for_date",
        description="Return daily counts and derived rates for one date.",
        parameters=["date"],
    ),
    MethodSpec(
        name="hourly_rows_for_date",
        description="Return hourly rows and traffic-share summaries for one date.",
        parameters=["date"],
    ),
    MethodSpec(
        name="compare_rate_to_median",
        description="Compare one conversion rate against its daily median baseline.",
        parameters=["date", "entity_name"],
    ),
    MethodSpec(
        name="compare_count_to_median",
        description="Compare one event count against its daily median baseline.",
        parameters=["date", "entity_name"],
    ),
    MethodSpec(
        name="detect_funnel_break",
        description="Inspect funnel-step rates and monotonicity for a date.",
        parameters=["date"],
    ),
    MethodSpec(
        name="check_impossible_counts",
        description="Find impossible daily or hourly count relationships for a date.",
        parameters=["date"],
    ),
    MethodSpec(
        name="list_suspicious_dates",
        description="Rank dates by anomaly suspicion using shared heuristics.",
        parameters=["limit"],
    ),
    MethodSpec(
        name="preview_submission",
        description="Validate candidate payload rows without revealing ground truth.",
        parameters=["rows"],
    ),
    MethodSpec(
        name="show_raw_data",
        description="Return a head() style view of daily aggregate rows with count and rate metrics.",
        parameters=["limit"],
    ),
    MethodSpec(
        name="get_metric_median",
        description="Return the median for a count metric or conversion metric.",
        parameters=["metric_name"],
    ),
    MethodSpec(
        name="get_metric_std_dev_from_median",
        description="Return sqrt(mean((value - median)^2)) for a metric.",
        parameters=["metric_name"],
    ),
    MethodSpec(
        name="get_rows_with_abs_diff_from_median_gt",
        description="Return all dates where abs(value - median) is greater than a threshold.",
        parameters=["metric_name", "threshold"],
    ),
    MethodSpec(
        name="get_median_filter_rows",
        description="Build payload rows where abs(value - median) > threshold_multiplier * std_from_median.",
        parameters=["metric_name", "threshold_multiplier"],
    ),
    MethodSpec(
        name="get_rate_drop_from_median_rows",
        description="Build conversion-rate payload rows where median-filtered values drop below baseline.",
        parameters=["metric_name", "threshold_multiplier"],
    ),
    MethodSpec(
        name="get_rate_spike_from_median_rows",
        description="Build conversion-rate payload rows where median-filtered values spike above baseline.",
        parameters=["metric_name", "threshold_multiplier"],
    ),
    MethodSpec(
        name="get_absolute_drop_in_event_count_rows",
        description="Build event-count payload rows where median-filtered values drop below baseline.",
        parameters=["metric_name", "threshold_multiplier"],
    ),
    MethodSpec(
        name="get_absolute_spike_in_event_count_rows",
        description="Build event-count payload rows where median-filtered values spike above baseline.",
        parameters=["metric_name", "threshold_multiplier"],
    ),
    MethodSpec(
        name="get_funnel_break_rows",
        description="Build payload rows for funnel-step breaks by scanning dates for large funnel-rate drops.",
        parameters=["threshold_multiplier"],
    ),
    MethodSpec(
        name="get_hourly_traffic_mix_shift_rows",
        description="Build payload rows for dates with unusual app_open daytime-share shifts.",
        parameters=["threshold_multiplier"],
    ),
    MethodSpec(
        name="get_instrumentation_data_quality_issue_rows",
        description="Build payload rows for dates with impossible count relationships or instrumentation issues.",
        parameters=["threshold_multiplier"],
    ),
    MethodSpec(
        name="payload_generator",
        description="Run a list of payload generation methods and merge the generated rows.",
        parameters=["generator_methods"],
    ),
)


def available_analysis_methods() -> list[MethodSpec]:
    """Return the shared safe method surface."""
    return list(DEFAULT_METHOD_SPECS)


@dataclass
class AnalysisContext:
    """Structured input for the shared method implementation."""

    daily_metrics: list[MetricRecord]
    hourly_metrics: list[MetricRecord]
    conversion_definitions: list[ConversionMetricDefinition]
    instruction: str = ""
    config: dict[str, Any] | None = None


class SharedAnalysisToolkit:
    """Shared method implementation for agents and the manual UI."""

    def __init__(self, context: AnalysisContext) -> None:
        self._context = context
        self._daily_by_date = {row.date: row for row in context.daily_metrics}
        self._hourly_by_date: dict[str, list[MetricRecord]] = {}
        for row in context.hourly_metrics:
            self._hourly_by_date.setdefault(row.date, []).append(row)
        for rows in self._hourly_by_date.values():
            rows.sort(key=lambda item: item.hour if item.hour is not None else -1)
        self._dates = sorted(self._daily_by_date)
        self._conversion_map = {item.name: item for item in context.conversion_definitions}

    def task_overview(self) -> dict[str, Any]:
        """Return a compact overview of the task and available entities."""
        return {
            "instruction": self._context.instruction,
            "config": self._context.config or {},
            "date_count": len(self._dates),
            "dates": self._dates,
            "threshold_search_space": {
                "rate_delta_pct_points": [3.0, 4.5, 6.0, 8.0],
                "count_delta_pct": [10.0, 15.0, 22.0, 30.0],
                "funnel_delta_pct_points": [3.5, 5.0, 7.0, 10.0],
                "hourly_mix_delta_pct": [8.0, 12.0, 18.0, 25.0],
            },
            "payload_schema": [
                "date",
                "entity_type",
                "entity_name",
                "anomaly_type",
                "detection_method",
                "baseline_value",
                "observed_value",
                "delta_value",
                "severity",
            ],
            "available_methods": [item.model_dump() for item in available_analysis_methods()],
            "entities": self.list_entities()["entities"],
        }

    def list_dates(self) -> dict[str, Any]:
        return {"dates": self._dates}

    def list_entities(self) -> dict[str, Any]:
        conversions = [
            {
                "entity_type": "conversion_rate",
                "entity_name": item.name,
                "formula": item.description,
            }
            for item in self._context.conversion_definitions
        ]
        counts = [
            {
                "entity_type": "event_count",
                "entity_name": metric_name,
            }
            for metric_name in COUNT_METRICS
        ]
        funnels = [
            {
                "entity_type": "funnel_step",
                "entity_name": f"{numerator}_from_{denominator}",
            }
            for numerator, denominator in FUNNEL_STEPS
        ]
        quality = [
            {
                "entity_type": "data_quality",
                "entity_name": f"{numerator}_lte_{denominator}",
            }
            for numerator, denominator in FUNNEL_STEPS
        ]
        hourly = [
            {
                "entity_type": "hourly_mix",
                "entity_name": "app_opens:daytime_share",
            }
        ]
        return {"entities": conversions + counts + funnels + quality + hourly}

    def rows_for_date(self, date: str) -> dict[str, Any]:
        row = self._daily_by_date.get(date)
        if row is None:
            return {"found": False, "date": date, "error": "Date not found."}
        derived_rates = self._conversion_rates(row)
        return {
            "found": True,
            "date": date,
            "daily_metrics": row.model_dump(),
            "derived_rates": derived_rates,
        }

    def hourly_rows_for_date(self, date: str) -> dict[str, Any]:
        rows = self._hourly_by_date.get(date, [])
        if not rows:
            return {"found": False, "date": date, "error": "Date not found."}
        total = sum(item.app_opens for item in rows) or 1
        daytime_hours = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
        daytime_share = round(
            sum(item.app_opens for item in rows if item.hour in daytime_hours) / total,
            4,
        )
        return {
            "found": True,
            "date": date,
            "summary": {
                "daytime_share": daytime_share,
                "top_hours": sorted(
                    (
                        {
                            "hour": item.hour,
                            "app_opens": item.app_opens,
                            "share": round(item.app_opens / total, 4),
                        }
                        for item in rows
                    ),
                    key=lambda item: item["app_opens"],
                    reverse=True,
                )[:5],
            },
            "rows": [item.model_dump() for item in rows],
        }

    def compare_rate_to_median(self, date: str, entity_name: str) -> dict[str, Any]:
        record = self._daily_by_date.get(date)
        definition = self._conversion_map.get(entity_name)
        if record is None or definition is None:
            return {
                "found": False,
                "date": date,
                "entity_name": entity_name,
                "error": "Date or conversion entity not found.",
            }
        series = [self._rate_for_record(item, definition) for item in self._context.daily_metrics]
        baseline = round(median(series), 4)
        observed = round(self._rate_for_record(record, definition), 4)
        delta = round(observed - baseline, 4)
        anomaly_type = "normal"
        if delta <= -self._rate_threshold():
            anomaly_type = "rate_drop_from_median"
        elif delta >= self._rate_threshold():
            anomaly_type = "rate_spike_from_median"
        return {
            "found": True,
            "date": date,
            "entity_type": "conversion_rate",
            "entity_name": entity_name,
            "detection_method": "compare_rate_to_median",
            "baseline_value": baseline,
            "observed_value": observed,
            "delta_value": delta,
            "anomaly_type": anomaly_type,
            "severity": self._severity(abs(delta), medium=4.0, high=8.0, critical=12.0),
        }

    def compare_count_to_median(self, date: str, entity_name: str) -> dict[str, Any]:
        record = self._daily_by_date.get(date)
        if record is None or entity_name not in COUNT_METRICS:
            return {
                "found": False,
                "date": date,
                "entity_name": entity_name,
                "error": "Date or count entity not found.",
            }
        series = [float(getattr(item, entity_name)) for item in self._context.daily_metrics]
        baseline = round(median(series), 4)
        observed = round(float(getattr(record, entity_name)), 4)
        delta = round(observed - baseline, 4)
        threshold = max(50.0, baseline * self._count_threshold_fraction())
        anomaly_type = "normal"
        if delta <= -threshold:
            anomaly_type = "absolute_drop_in_event_count"
        elif delta >= threshold:
            anomaly_type = "absolute_spike_in_event_count"
        return {
            "found": True,
            "date": date,
            "entity_type": "event_count",
            "entity_name": entity_name,
            "detection_method": "compare_count_to_median",
            "baseline_value": baseline,
            "observed_value": observed,
            "delta_value": delta,
            "anomaly_type": anomaly_type,
            "severity": self._severity(
                abs(delta) / max(baseline, 1.0) * 100.0,
                medium=12.0,
                high=22.0,
                critical=35.0,
            ),
        }

    def detect_funnel_break(self, date: str) -> dict[str, Any]:
        record = self._daily_by_date.get(date)
        if record is None:
            return {"found": False, "date": date, "error": "Date not found."}
        candidates: list[dict[str, Any]] = []
        for numerator, denominator in FUNNEL_STEPS:
            entity_name = f"{numerator}_from_{denominator}"
            baseline_series = [
                self._ratio(getattr(item, numerator), getattr(item, denominator)) * 100.0
                for item in self._context.daily_metrics
            ]
            baseline = round(median(baseline_series), 4)
            observed = round(
                self._ratio(getattr(record, numerator), getattr(record, denominator)) * 100.0,
                4,
            )
            delta = round(observed - baseline, 4)
            issue = {
                "entity_type": "funnel_step",
                "entity_name": entity_name,
                "detection_method": "detect_funnel_break",
                "baseline_value": baseline,
                "observed_value": observed,
                "delta_value": delta,
                "monotonicity_broken": getattr(record, numerator) > getattr(record, denominator),
                "severity": self._severity(abs(delta), medium=5.0, high=10.0, critical=15.0),
            }
            if issue["monotonicity_broken"] or delta <= -self._funnel_threshold():
                issue["anomaly_type"] = "funnel_break"
                candidates.append(issue)
        return {"found": True, "date": date, "candidates": candidates}

    def check_impossible_counts(self, date: str) -> dict[str, Any]:
        daily = self._daily_by_date.get(date)
        hourly_rows = self._hourly_by_date.get(date, [])
        if daily is None:
            return {"found": False, "date": date, "error": "Date not found."}
        issues = []
        issues.extend(self._impossible_issues(daily, scope="daily"))
        for row in hourly_rows:
            issues.extend(self._impossible_issues(row, scope=f"hour_{row.hour:02d}"))
        total_excess = round(sum(item["excess_value"] for item in issues), 4)
        return {
            "found": True,
            "date": date,
            "issue_count": len(issues),
            "total_excess": total_excess,
            "issues": issues,
        }

    def list_suspicious_dates(self, limit: int = 10) -> dict[str, Any]:
        ranked = []
        hourly_baseline = self._median_daytime_share()
        for date in self._dates:
            rate_signal = 0.0
            for definition in self._context.conversion_definitions:
                comparison = self.compare_rate_to_median(date, definition.name)
                rate_signal = max(rate_signal, abs(comparison["delta_value"]))
            count_signal = 0.0
            for metric_name in COUNT_METRICS:
                comparison = self.compare_count_to_median(date, metric_name)
                baseline = max(comparison["baseline_value"], 1.0)
                count_signal = max(
                    count_signal,
                    abs(comparison["delta_value"]) / baseline * 100.0,
                )
            funnel_candidates = self.detect_funnel_break(date)["candidates"]
            impossible = self.check_impossible_counts(date)
            hourly_share = self.hourly_rows_for_date(date)["summary"]["daytime_share"]
            hourly_signal = abs(hourly_share - hourly_baseline) * 100.0
            suspicion_score = round(
                rate_signal + count_signal + hourly_signal + impossible["total_excess"] * 0.05
                + len(funnel_candidates) * 6.0,
                4,
            )
            ranked.append(
                {
                    "date": date,
                    "suspicion_score": suspicion_score,
                    "max_rate_delta": round(rate_signal, 4),
                    "max_count_delta_pct": round(count_signal, 4),
                    "hourly_mix_delta_pct": round(hourly_signal, 4),
                    "funnel_candidate_count": len(funnel_candidates),
                    "impossible_issue_count": impossible["issue_count"],
                }
            )
        ranked.sort(key=lambda item: (item["suspicion_score"], item["date"]), reverse=True)
        return {"dates": ranked[: max(limit, 1)]}

    def preview_submission(self, rows: list[dict[str, Any]] | list[MetricSubmissionRow]) -> dict[str, Any]:
        preview = preview_submission_rows(rows)
        return preview.model_dump()

    def show_raw_data(self, limit: int = 5) -> dict[str, Any]:
        rows = []
        for record in self._context.daily_metrics[: max(limit, 1)]:
            row = record.model_dump()
            row.update(self._conversion_rates(record))
            rows.append(row)
        return {
            "row_count": len(self._context.daily_metrics),
            "returned_rows": len(rows),
            "rows": rows,
        }

    def get_metric_median(self, metric_name: str) -> dict[str, Any]:
        descriptor = self._metric_descriptor(metric_name)
        values = descriptor["values"]
        metric_median = round(median(values), 4) if values else 0.0
        return {
            "metric_name": metric_name,
            "metric_type": descriptor["metric_type"],
            "median_value": metric_median,
            "sample_size": len(values),
        }

    def get_metric_median_multi(
        self,
        metric_name: str | None = None,
        metric_names: list[str] | None = None,
    ) -> dict[str, Any]:
        resolved_metrics = self._resolve_metric_names(metric_name=metric_name, metric_names=metric_names)
        results = [self.get_metric_median(name) for name in resolved_metrics]
        return {
            "metric_name": metric_name,
            "metric_names": resolved_metrics,
            "results": results,
        }

    def get_metric_std_dev_from_median(self, metric_name: str) -> dict[str, Any]:
        descriptor = self._metric_descriptor(metric_name)
        values = descriptor["values"]
        metric_median = median(values) if values else 0.0
        std_from_median = math.sqrt(
            sum((value - metric_median) ** 2 for value in values) / len(values)
        ) if values else 0.0
        return {
            "metric_name": metric_name,
            "metric_type": descriptor["metric_type"],
            "median_value": round(metric_median, 4),
            "std_dev_from_median": round(std_from_median, 4),
            "sample_size": len(values),
        }

    def get_metric_std_dev_from_median_multi(
        self,
        metric_name: str | None = None,
        metric_names: list[str] | None = None,
    ) -> dict[str, Any]:
        resolved_metrics = self._resolve_metric_names(metric_name=metric_name, metric_names=metric_names)
        results = [self.get_metric_std_dev_from_median(name) for name in resolved_metrics]
        return {
            "metric_name": metric_name,
            "metric_names": resolved_metrics,
            "results": results,
        }

    def get_rows_with_abs_diff_from_median_gt(self, metric_name: str, threshold: float) -> dict[str, Any]:
        descriptor = self._metric_descriptor(metric_name)
        metric_median = median(descriptor["values"]) if descriptor["values"] else 0.0
        matches = []
        for date_key, value in descriptor["per_date_values"].items():
            abs_diff = abs(value - metric_median)
            if abs_diff <= threshold:
                continue
            row = {
                "date": date_key,
                "metric_name": metric_name,
                "metric_type": descriptor["metric_type"],
                "median_value": round(metric_median, 4),
                "observed_value": round(value, 4),
                "abs_diff": round(abs_diff, 4),
            }
            suggested = self._build_submission_row_for_metric(
                metric_name=metric_name,
                date=date_key,
                baseline_value=float(metric_median),
                observed_value=float(value),
            )
            if suggested is not None:
                row["suggested_payload_row"] = suggested.model_dump()
            matches.append(row)
        return {
            "metric_name": metric_name,
            "threshold": threshold,
            "match_count": len(matches),
            "rows": matches,
        }

    def get_rows_with_abs_diff_from_median_gt_multi(
        self,
        metric_name: str | None = None,
        metric_names: list[str] | None = None,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        resolved_metrics = self._resolve_metric_names(metric_name=metric_name, metric_names=metric_names)
        results = [
            self.get_rows_with_abs_diff_from_median_gt(name, threshold)
            for name in resolved_metrics
        ]
        return {
            "metric_name": metric_name,
            "metric_names": resolved_metrics,
            "threshold": threshold,
            "results": results,
        }

    def get_median_filter_rows(self, metric_name: str, threshold_multiplier: float) -> dict[str, Any]:
        return self.get_median_filter_rows_multi(
            metric_name=metric_name,
            metric_names=[],
            threshold_multiplier=threshold_multiplier,
        )

    def get_median_filter_rows_multi(
        self,
        metric_name: str | None = None,
        metric_names: list[str] | None = None,
        threshold_multiplier: float = 2.0,
    ) -> dict[str, Any]:
        resolved_metrics = self._resolve_metric_names(metric_name=metric_name, metric_names=metric_names)
        details = []
        generated: dict[str, dict[str, Any]] = {}
        total_matches = 0
        for resolved_metric in resolved_metrics:
            stats = self.get_metric_std_dev_from_median(resolved_metric)
            threshold = stats["std_dev_from_median"] * threshold_multiplier
            rows_result = self.get_rows_with_abs_diff_from_median_gt(resolved_metric, threshold)
            payload_rows = [
                row["suggested_payload_row"]
                for row in rows_result["rows"]
                if row.get("suggested_payload_row")
            ]
            total_matches += rows_result["match_count"]
            for row in payload_rows:
                submission_row = MetricSubmissionRow(**row)
                generated[submission_row_key(submission_row)] = submission_row.model_dump()
            details.append(
                {
                    "metric_name": resolved_metric,
                    "threshold": round(threshold, 4),
                    "match_count": rows_result["match_count"],
                    "rows": rows_result["rows"],
                    "generated_rows": payload_rows,
                }
            )
        return {
            "method_name": "get_median_filter_rows",
            "metric_name": metric_name,
            "metric_names": resolved_metrics,
            "threshold_multiplier": threshold_multiplier,
            "match_count": total_matches,
            "generated_rows": list(generated.values()),
            "details": details,
        }

    def get_rate_drop_from_median_rows(
        self,
        metric_name: str | None = None,
        metric_names: list[str] | None = None,
        threshold_multiplier: float = 2.0,
    ) -> dict[str, Any]:
        return self._metric_family_filter_rows(
            method_name="get_rate_drop_from_median_rows",
            metric_name=metric_name,
            metric_names=metric_names,
            threshold_multiplier=threshold_multiplier,
            metric_type="conversion_rate",
            allowed_anomaly_types={"rate_drop_from_median"},
        )

    def get_rate_spike_from_median_rows(
        self,
        metric_name: str | None = None,
        metric_names: list[str] | None = None,
        threshold_multiplier: float = 2.0,
    ) -> dict[str, Any]:
        return self._metric_family_filter_rows(
            method_name="get_rate_spike_from_median_rows",
            metric_name=metric_name,
            metric_names=metric_names,
            threshold_multiplier=threshold_multiplier,
            metric_type="conversion_rate",
            allowed_anomaly_types={"rate_spike_from_median"},
        )

    def get_absolute_drop_in_event_count_rows(
        self,
        metric_name: str | None = None,
        metric_names: list[str] | None = None,
        threshold_multiplier: float = 2.0,
    ) -> dict[str, Any]:
        return self._metric_family_filter_rows(
            method_name="get_absolute_drop_in_event_count_rows",
            metric_name=metric_name,
            metric_names=metric_names,
            threshold_multiplier=threshold_multiplier,
            metric_type="event_count",
            allowed_anomaly_types={"absolute_drop_in_event_count"},
        )

    def get_absolute_spike_in_event_count_rows(
        self,
        metric_name: str | None = None,
        metric_names: list[str] | None = None,
        threshold_multiplier: float = 2.0,
    ) -> dict[str, Any]:
        return self._metric_family_filter_rows(
            method_name="get_absolute_spike_in_event_count_rows",
            metric_name=metric_name,
            metric_names=metric_names,
            threshold_multiplier=threshold_multiplier,
            metric_type="event_count",
            allowed_anomaly_types={"absolute_spike_in_event_count"},
        )

    def get_funnel_break_rows(self, threshold_multiplier: float = 2.0) -> dict[str, Any]:
        details = []
        generated: dict[str, dict[str, Any]] = {}
        total_matches = 0
        for numerator, denominator in FUNNEL_STEPS:
            entity_name = f"{numerator}_from_{denominator}"
            per_date_values = {
                date_key: round(
                    self._ratio(getattr(record, numerator), getattr(record, denominator)) * 100.0,
                    4,
                )
                for date_key, record in self._daily_by_date.items()
            }
            values = list(per_date_values.values())
            baseline = median(values) if values else 0.0
            std_from_median = math.sqrt(
                sum((value - baseline) ** 2 for value in values) / len(values)
            ) if values else 0.0
            threshold = max(std_from_median * float(threshold_multiplier), self._funnel_threshold())
            rows = []
            generated_rows = []
            for date_key, observed_value in per_date_values.items():
                delta_value = round(observed_value - baseline, 4)
                if delta_value > -threshold:
                    continue
                row = {
                    "date": date_key,
                    "entity_type": "funnel_step",
                    "entity_name": entity_name,
                    "anomaly_type": "funnel_break",
                    "detection_method": "detect_funnel_break",
                    "baseline_value": round(baseline, 4),
                    "observed_value": round(observed_value, 4),
                    "delta_value": delta_value,
                    "severity": self._severity(abs(delta_value), medium=5.0, high=10.0, critical=15.0),
                }
                total_matches += 1
                rows.append(row)
                submission_row = MetricSubmissionRow(**row)
                generated[submission_row_key(submission_row)] = submission_row.model_dump()
                generated_rows.append(submission_row.model_dump())
            details.append(
                {
                    "entity_name": entity_name,
                    "threshold": round(threshold, 4),
                    "match_count": len(rows),
                    "rows": rows,
                    "generated_rows": generated_rows,
                }
            )
        return {
            "method_name": "get_funnel_break_rows",
            "threshold_multiplier": threshold_multiplier,
            "match_count": total_matches,
            "generated_rows": list(generated.values()),
            "details": details,
        }

    def get_hourly_traffic_mix_shift_rows(self, threshold_multiplier: float = 2.0) -> dict[str, Any]:
        per_date_values = {}
        for date_key in self._dates:
            summary = self.hourly_rows_for_date(date_key).get("summary", {})
            per_date_values[date_key] = float(summary.get("daytime_share", 0.0))
        values = list(per_date_values.values())
        baseline = median(values) if values else 0.0
        std_from_median = math.sqrt(
            sum((value - baseline) ** 2 for value in values) / len(values)
        ) if values else 0.0
        threshold = std_from_median * float(threshold_multiplier)
        rows = []
        generated_rows = []
        for date_key, observed_value in per_date_values.items():
            delta_value = round(observed_value - baseline, 4)
            if abs(delta_value) <= threshold:
                continue
            row = {
                "date": date_key,
                "entity_type": "hourly_mix",
                "entity_name": "app_opens:daytime_share",
                "anomaly_type": "hourly_traffic_mix_shift",
                "detection_method": "hourly_rows_for_date",
                "baseline_value": round(baseline, 4),
                "observed_value": round(observed_value, 4),
                "delta_value": delta_value,
                "severity": self._severity(abs(delta_value) * 100.0, medium=10.0, high=18.0, critical=25.0),
            }
            rows.append(row)
            generated_rows.append(row)
        return {
            "method_name": "get_hourly_traffic_mix_shift_rows",
            "threshold_multiplier": threshold_multiplier,
            "match_count": len(rows),
            "generated_rows": generated_rows,
            "details": [
                {
                    "entity_name": "app_opens:daytime_share",
                    "threshold": round(threshold, 4),
                    "match_count": len(rows),
                    "rows": rows,
                    "generated_rows": generated_rows,
                }
            ],
        }

    def get_instrumentation_data_quality_issue_rows(
        self,
        threshold_multiplier: float = 2.0,
    ) -> dict[str, Any]:
        per_date_totals = {
            date_key: float(self.check_impossible_counts(date_key).get("total_excess", 0.0))
            for date_key in self._dates
        }
        values = list(per_date_totals.values())
        baseline = median(values) if values else 0.0
        std_from_median = math.sqrt(
            sum((value - baseline) ** 2 for value in values) / len(values)
        ) if values else 0.0
        threshold = std_from_median * float(threshold_multiplier)
        generated: dict[str, dict[str, Any]] = {}
        details = []
        total_matches = 0
        for numerator, denominator in FUNNEL_STEPS:
            entity_name = f"{numerator}_lte_{denominator}"
            rows = []
            generated_rows = []
            for date_key in self._dates:
                result = self.check_impossible_counts(date_key)
                issue_names = {item["entity_name"] for item in result.get("issues", [])}
                observed_value = float(result.get("total_excess", 0.0))
                if entity_name not in issue_names or observed_value <= threshold:
                    continue
                row = {
                    "date": date_key,
                    "entity_type": "data_quality",
                    "entity_name": entity_name,
                    "anomaly_type": "instrumentation_data_quality_issue",
                    "detection_method": "check_impossible_counts",
                    "baseline_value": round(baseline, 4),
                    "observed_value": round(observed_value, 4),
                    "delta_value": round(observed_value - baseline, 4),
                    "severity": self._severity(observed_value, medium=20.0, high=60.0, critical=120.0),
                }
                total_matches += 1
                rows.append(row)
                submission_row = MetricSubmissionRow(**row)
                generated[submission_row_key(submission_row)] = submission_row.model_dump()
                generated_rows.append(submission_row.model_dump())
            details.append(
                {
                    "entity_name": entity_name,
                    "threshold": round(threshold, 4),
                    "match_count": len(rows),
                    "rows": rows,
                    "generated_rows": generated_rows,
                }
            )
        return {
            "method_name": "get_instrumentation_data_quality_issue_rows",
            "threshold_multiplier": threshold_multiplier,
            "match_count": total_matches,
            "generated_rows": list(generated.values()),
            "details": details,
        }

    def payload_generator(
        self,
        generator_methods: list[dict[str, Any]] | list[PayloadGeneratorMethod],
    ) -> dict[str, Any]:
        methods = [
            item if isinstance(item, PayloadGeneratorMethod) else PayloadGeneratorMethod(**item)
            for item in generator_methods
        ]
        generated: dict[str, MetricSubmissionRow] = {}
        details = []
        for method in methods:
            result = self._run_payload_generator_method(method)
            if "error" in result:
                details.append(result)
                continue
            for row in result["generated_rows"]:
                submission_row = MetricSubmissionRow(**row)
                generated[submission_row_key(submission_row)] = submission_row
            details.append(result)
        return {
            "generator_methods": [item.model_dump() for item in methods],
            "generated_row_count": len(generated),
            "generated_rows": [row.model_dump() for row in generated.values()],
            "details": details,
        }

    def _run_payload_generator_method(self, method: PayloadGeneratorMethod) -> dict[str, Any]:
        if method.method_name == "get_median_filter_rows":
            return self.get_median_filter_rows(
                metric_name=method.metric_name,
                threshold_multiplier=method.threshold_multiplier,
            ) if not method.metric_names else self.get_median_filter_rows_multi(
                metric_name=method.metric_name,
                metric_names=method.metric_names,
                threshold_multiplier=method.threshold_multiplier,
            )
        if method.method_name == "get_rate_drop_from_median_rows":
            return self.get_rate_drop_from_median_rows(
                metric_name=method.metric_name,
                metric_names=method.metric_names,
                threshold_multiplier=method.threshold_multiplier,
            )
        if method.method_name == "get_rate_spike_from_median_rows":
            return self.get_rate_spike_from_median_rows(
                metric_name=method.metric_name,
                metric_names=method.metric_names,
                threshold_multiplier=method.threshold_multiplier,
            )
        if method.method_name == "get_absolute_drop_in_event_count_rows":
            return self.get_absolute_drop_in_event_count_rows(
                metric_name=method.metric_name,
                metric_names=method.metric_names,
                threshold_multiplier=method.threshold_multiplier,
            )
        if method.method_name == "get_absolute_spike_in_event_count_rows":
            return self.get_absolute_spike_in_event_count_rows(
                metric_name=method.metric_name,
                metric_names=method.metric_names,
                threshold_multiplier=method.threshold_multiplier,
            )
        if method.method_name == "get_funnel_break_rows":
            return self.get_funnel_break_rows(threshold_multiplier=method.threshold_multiplier)
        if method.method_name == "get_hourly_traffic_mix_shift_rows":
            return self.get_hourly_traffic_mix_shift_rows(threshold_multiplier=method.threshold_multiplier)
        if method.method_name == "get_instrumentation_data_quality_issue_rows":
            return self.get_instrumentation_data_quality_issue_rows(threshold_multiplier=method.threshold_multiplier)
        return {
            "method": method.model_dump(),
            "error": "Unsupported payload generator method.",
        }

    def build_row_from_analysis(self, analysis_result: dict[str, Any]) -> dict[str, Any] | None:
        """Extract a payload row when an analysis result directly maps to one."""
        required_fields = {
            "date",
            "entity_type",
            "entity_name",
            "anomaly_type",
            "detection_method",
            "baseline_value",
            "observed_value",
            "delta_value",
            "severity",
        }
        if required_fields.issubset(analysis_result) and analysis_result.get("anomaly_type") != "normal":
            return {field: analysis_result[field] for field in required_fields}
        return None

    def _conversion_rates(self, record: MetricRecord) -> dict[str, float]:
        return {
            item.name: round(self._rate_for_record(record, item), 4)
            for item in self._context.conversion_definitions
        }

    def _metric_descriptor(self, metric_name: str) -> dict[str, Any]:
        if metric_name in COUNT_METRICS:
            values = [float(getattr(item, metric_name)) for item in self._context.daily_metrics]
            per_date_values = {
                item.date: float(getattr(item, metric_name))
                for item in self._context.daily_metrics
            }
            return {
                "metric_type": "event_count",
                "values": values,
                "per_date_values": per_date_values,
            }
        definition = self._conversion_map.get(metric_name)
        if definition is None:
            raise ValueError(f"Unknown metric_name: {metric_name}")
        values = [self._rate_for_record(item, definition) for item in self._context.daily_metrics]
        per_date_values = {
            item.date: self._rate_for_record(item, definition)
            for item in self._context.daily_metrics
        }
        return {
            "metric_type": "conversion_rate",
            "values": values,
            "per_date_values": per_date_values,
        }

    def _resolve_metric_names(
        self,
        *,
        metric_name: str | None,
        metric_names: list[str] | None,
    ) -> list[str]:
        names = [item for item in (metric_names or []) if item]
        if metric_name:
            names.append(metric_name)
        if not names:
            names = list(COUNT_METRICS) + list(self._conversion_map.keys())
        deduped = []
        seen = set()
        for item in names:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _resolve_metric_names_for_type(
        self,
        *,
        metric_name: str | None,
        metric_names: list[str] | None,
        metric_type: str,
    ) -> list[str]:
        resolved = self._resolve_metric_names(metric_name=metric_name, metric_names=metric_names)
        return [
            item
            for item in resolved
            if self._metric_descriptor(item)["metric_type"] == metric_type
        ]

    def _metric_family_filter_rows(
        self,
        *,
        method_name: str,
        metric_name: str | None,
        metric_names: list[str] | None,
        threshold_multiplier: float,
        metric_type: str,
        allowed_anomaly_types: set[str],
    ) -> dict[str, Any]:
        resolved_metrics = self._resolve_metric_names_for_type(
            metric_name=metric_name,
            metric_names=metric_names,
            metric_type=metric_type,
        )
        raw_result = self.get_median_filter_rows_multi(
            metric_name=None,
            metric_names=resolved_metrics,
            threshold_multiplier=threshold_multiplier,
        )
        generated: dict[str, dict[str, Any]] = {}
        details = []
        total_matches = 0
        for detail in raw_result["details"]:
            filtered_rows = []
            filtered_generated = []
            for row in detail["rows"]:
                suggested = row.get("suggested_payload_row")
                if not suggested or suggested.get("anomaly_type") not in allowed_anomaly_types:
                    continue
                filtered_rows.append(row)
                submission_row = MetricSubmissionRow(**suggested)
                generated[submission_row_key(submission_row)] = submission_row.model_dump()
                filtered_generated.append(submission_row.model_dump())
            total_matches += len(filtered_rows)
            details.append(
                {
                    **detail,
                    "match_count": len(filtered_rows),
                    "rows": filtered_rows,
                    "generated_rows": filtered_generated,
                }
            )
        return {
            "method_name": method_name,
            "metric_name": metric_name,
            "metric_names": resolved_metrics,
            "threshold_multiplier": threshold_multiplier,
            "match_count": total_matches,
            "generated_rows": list(generated.values()),
            "details": details,
        }

    def _build_submission_row_for_metric(
        self,
        *,
        metric_name: str,
        date: str,
        baseline_value: float,
        observed_value: float,
    ) -> MetricSubmissionRow | None:
        delta_value = round(observed_value - baseline_value, 4)
        if metric_name in COUNT_METRICS:
            threshold = max(50.0, baseline_value * self._count_threshold_fraction())
            if abs(delta_value) <= threshold:
                return None
            anomaly_type = (
                "absolute_spike_in_event_count"
                if delta_value > 0
                else "absolute_drop_in_event_count"
            )
            return MetricSubmissionRow(
                date=date,
                entity_type="event_count",
                entity_name=metric_name,
                anomaly_type=anomaly_type,
                detection_method="compare_count_to_median",
                baseline_value=round(baseline_value, 4),
                observed_value=round(observed_value, 4),
                delta_value=delta_value,
                severity=self._severity(
                    abs(delta_value) / max(baseline_value, 1.0) * 100.0,
                    medium=12.0,
                    high=22.0,
                    critical=35.0,
                ),
            )
        threshold = self._rate_threshold()
        if abs(delta_value) <= threshold:
            return None
        anomaly_type = "rate_spike_from_median" if delta_value > 0 else "rate_drop_from_median"
        return MetricSubmissionRow(
            date=date,
            entity_type="conversion_rate",
            entity_name=metric_name,
            anomaly_type=anomaly_type,
            detection_method="compare_rate_to_median",
            baseline_value=round(baseline_value, 4),
            observed_value=round(observed_value, 4),
            delta_value=delta_value,
            severity=self._severity(abs(delta_value), medium=4.0, high=8.0, critical=12.0),
        )

    def _impossible_issues(self, row: MetricRecord, scope: str) -> list[dict[str, Any]]:
        issues = []
        for numerator, denominator in FUNNEL_STEPS:
            numerator_value = getattr(row, numerator)
            denominator_value = getattr(row, denominator)
            if numerator_value > denominator_value:
                issues.append(
                    {
                        "scope": scope,
                        "entity_name": f"{numerator}_lte_{denominator}",
                        "numerator": numerator_value,
                        "denominator": denominator_value,
                        "excess_value": round(float(numerator_value - denominator_value), 4),
                    }
                )
        return issues

    def _median_daytime_share(self) -> float:
        shares = []
        for date in self._dates:
            hourly_data = self.hourly_rows_for_date(date)
            shares.append(hourly_data["summary"]["daytime_share"])
        return round(median(shares), 4) if shares else 0.0

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def _rate_for_record(
        self,
        record: MetricRecord,
        definition: ConversionMetricDefinition,
    ) -> float:
        return self._ratio(
            getattr(record, definition.numerator),
            getattr(record, definition.denominator),
        ) * 100.0

    def _rate_threshold(self) -> float:
        difficulty = (self._context.config or {}).get("difficulty", "medium")
        return {"easy": 6.0, "medium": 4.5, "hard": 3.0}.get(difficulty, 4.5)

    def _funnel_threshold(self) -> float:
        difficulty = (self._context.config or {}).get("difficulty", "medium")
        return {"easy": 7.0, "medium": 5.0, "hard": 3.5}.get(difficulty, 5.0)

    def _count_threshold_fraction(self) -> float:
        difficulty = (self._context.config or {}).get("difficulty", "medium")
        return {"easy": 0.22, "medium": 0.15, "hard": 0.10}.get(difficulty, 0.15)

    @staticmethod
    def _severity(value: float, *, medium: float, high: float, critical: float) -> str:
        if value >= critical:
            return "critical"
        if value >= high:
            return "high"
        if value >= medium:
            return "medium"
        return "low"


def preview_submission_rows(
    rows: list[dict[str, Any]] | list[MetricSubmissionRow],
) -> SubmissionPreview:
    """Validate submission rows without using ground truth."""
    normalized_rows: list[MetricSubmissionRow] = []
    issues: list[SubmissionIssue] = []
    seen: set[str] = set()
    duplicate_rows = 0
    invalid_rows = 0

    for index, row in enumerate(rows):
        try:
            normalized = row if isinstance(row, MetricSubmissionRow) else MetricSubmissionRow(**row)
        except Exception as exc:
            invalid_rows += 1
            issues.append(
                SubmissionIssue(
                    row_key=f"row_{index}",
                    issue_type="invalid_row",
                    message=f"Row could not be parsed: {exc}",
                    submitted_row=row if isinstance(row, dict) else None,
                )
            )
            continue

        row_key = submission_row_key(normalized)
        if row_key in seen:
            duplicate_rows += 1
            issues.append(
                SubmissionIssue(
                    row_key=row_key,
                    issue_type="duplicate_row",
                    message="Duplicate date/entity row detected.",
                    submitted_row=normalized.model_dump(),
                )
            )
            continue

        seen.add(row_key)
        normalized_rows.append(normalized)

    return SubmissionPreview(
        valid_rows=len(normalized_rows),
        invalid_rows=invalid_rows,
        duplicate_rows=duplicate_rows,
        unique_keys=len(seen),
        issues=issues,
        normalized_rows=normalized_rows,
    )


def submission_row_key(row: MetricSubmissionRow) -> str:
    """Stable row key for matching submissions and expectations."""
    return f"{row.date}|{row.entity_type}|{row.entity_name}"
