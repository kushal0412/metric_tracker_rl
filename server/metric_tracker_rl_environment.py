"""Metric tracking RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..analysis_tools import AnalysisContext, SharedAnalysisToolkit, available_analysis_methods
    from ..evaluation import EvaluationConfig
    from ..models import (
        MetricTrackerRlAction,
        MetricTrackerRlObservation,
        MetricSubmissionRow,
        SyntheticAnomalyGenerator,
    )
    from ..tasks import DEFAULT_TASK_ID, available_task_specs, get_task_spec
    from .data_generator import (
        EpisodeConfig,
        EpisodeData,
        MetricDataGenerator,
        available_synthetic_generator_methods,
    )
except ImportError:
    from analysis_tools import AnalysisContext, SharedAnalysisToolkit, available_analysis_methods
    from models import (
        MetricTrackerRlAction,
        MetricTrackerRlObservation,
        MetricSubmissionRow,
        SyntheticAnomalyGenerator,
    )
    from tasks import DEFAULT_TASK_ID, available_task_specs, get_task_spec
    from server.data_generator import (
        EpisodeConfig,
        EpisodeData,
        MetricDataGenerator,
        available_synthetic_generator_methods,
    )
    from evaluation import EvaluationConfig


@dataclass(frozen=True)
class RewardConfig:
    """Compatibility wrapper around the evaluator configuration."""

    evaluation: EvaluationConfig = EvaluationConfig()


class MetricTrackerRlEnvironment(Environment):
    """Iterative multi-anomaly benchmark with safe analysis methods."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        generator: MetricDataGenerator | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        initial_task = get_task_spec(DEFAULT_TASK_ID)
        self._generator = generator or MetricDataGenerator()
        self._reward_config = reward_config or RewardConfig()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: EpisodeData | None = None
        self._completed = False
        self._debug_mode = False
        self._active_task = initial_task
        self._next_task_id = initial_task.task_id
        self._next_reset_config = initial_task.build_episode_config()
        self._last_analysis_result: dict | None = None
        self._expose_applied_generators = False

    def configure_next_reset(
        self,
        *,
        task_id: str | None = None,
        seed: int | None = None,
        scenario_family: str | None = None,
        difficulty: str | None = None,
        anomaly_density: str | None = None,
        anomaly_count: int | None = None,
        anomalies: list[dict] | list[SyntheticAnomalyGenerator] | None = None,
    ) -> None:
        """Update the configuration used for the next reset."""
        base_task = get_task_spec(task_id or self._next_task_id)
        base_config = base_task.build_episode_config() if task_id else self._next_reset_config
        anomaly_generators = tuple(
            item if isinstance(item, SyntheticAnomalyGenerator) else SyntheticAnomalyGenerator(**item)
            for item in (anomalies or [])
        )
        self._next_task_id = base_task.task_id
        self._next_reset_config = EpisodeConfig(
            seed=base_config.seed if seed is None else seed,
            scenario_family=base_config.scenario_family if scenario_family is None else scenario_family,
            difficulty=base_config.difficulty if difficulty is None else difficulty,
            anomaly_density=base_config.anomaly_density if anomaly_density is None else anomaly_density,
            anomaly_count=base_config.anomaly_count if anomaly_count is None else anomaly_count,
            anomaly_generators=anomaly_generators or base_config.anomaly_generators,
        ).normalized()

    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug-only environment views."""
        self._debug_mode = bool(enabled)

    def export_debug_snapshot(self) -> dict:
        """Return a developer-only debug snapshot for the active episode."""
        if not self._debug_mode:
            raise RuntimeError("Debug mode is disabled.")
        if self._episode is None:
            return {}
        return {
            "config": self._episode.config.__dict__,
            "expected_payload": [row.model_dump() for row in self._episode.expected_rows],
            "anomaly_schedule": self._episode.anomaly_schedule,
            "applied_synthetic_generators": [
                row.model_dump() for row in self._episode.applied_synthetic_generators
            ],
        }

    def reset(
        self,
        task_id: str | None = None,
        seed: int | None = None,
        scenario_family: str | None = None,
        difficulty: str | None = None,
        anomaly_density: str | None = None,
        anomaly_count: int | None = None,
        anomalies: list[dict] | list[SyntheticAnomalyGenerator] | None = None,
    ) -> MetricTrackerRlObservation:
        """Generate a fresh dataset and hidden target payload."""
        if any(value is not None for value in (task_id, seed, scenario_family, difficulty, anomaly_density, anomaly_count)) or anomalies is not None:
            self.configure_next_reset(
                task_id=task_id,
                seed=seed,
                scenario_family=scenario_family,
                difficulty=difficulty,
                anomaly_density=anomaly_density,
                anomaly_count=anomaly_count,
                anomalies=anomalies,
            )
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._active_task = get_task_spec(self._next_task_id)
        self._episode = self._generator.generate_episode(self._next_reset_config)
        self._completed = False
        self._last_analysis_result = None
        self._expose_applied_generators = anomalies is not None
        return self._build_observation(
            status="ready",
            message=self._active_task.objective,
            reward=0.0,
            done=False,
        )

    def step(self, action: MetricTrackerRlAction) -> MetricTrackerRlObservation:  # type: ignore[override]
        """Evaluate a submitted payload and return deterministic feedback."""
        if self._episode is None:
            return self.reset()
        if self._completed:
            return self._build_observation(
                status="completed",
                message="Dataset already solved. Call reset() to create a new dataset.",
                reward=1.0,
                done=True,
                submitted_rows=action.classifications,
            )

        if action.analysis_method:
            self._state.step_count += 1
            analysis_result = self._run_analysis(action.analysis_method, action.analysis_args)
            self._last_analysis_result = analysis_result
            return self._build_observation(
                status="analyzed",
                message=f"Ran analysis method `{action.analysis_method}`.",
                reward=0.0,
                done=False,
                analysis_result=analysis_result,
            )

        submitted_rows = action.classifications
        generated_rows: list[MetricSubmissionRow] = []
        if action.payload_generators:
            generator_result = self._run_analysis(
                "payload_generator",
                {"generator_methods": [item.model_dump() for item in action.payload_generators]},
            )
            self._last_analysis_result = generator_result
            generated_rows = [
                MetricSubmissionRow(**row)
                for row in generator_result["result"]["generated_rows"]
            ]
            submitted_rows = generated_rows

        self._state.step_count += 1
        result = self._active_task.grade_submission(
            submitted_rows,
            self._episode.expected_rows,
            config=self._reward_config.evaluation,
            include_debug_expected=self._debug_mode,
        )
        self._completed = result.is_perfect
        reward = result.reward_breakdown.total_score
        message = self._submission_message(result)
        return self._build_observation(
            status="evaluated" if result.is_perfect else "in_progress",
            message=message,
            reward=reward,
            done=result.is_perfect,
            submitted_rows=result.preview.normalized_rows,
            reward_breakdown=result.reward_breakdown,
            submission_preview=result.preview,
            issues=result.issues,
            correct_row_count=result.matched_rows,
            analysis_result=self._last_analysis_result,
            generated_rows=generated_rows,
        )

    @property
    def state(self) -> State:
        """Return current episode state."""
        return self._state

    def _build_observation(
        self,
        *,
        status: str,
        message: str,
        reward: float,
        done: bool,
        submitted_rows=None,
        reward_breakdown=None,
        submission_preview=None,
        issues=None,
        correct_row_count: int = 0,
        analysis_result=None,
        generated_rows=None,
    ) -> MetricTrackerRlObservation:
        assert self._episode is not None
        metadata = {
            "step": self._state.step_count,
            "current_state": self.state.model_dump(),
            "task_id": self._active_task.task_id,
            "objective": self._active_task.objective,
            "grader_name": self._active_task.grader_name,
            "seed": self._episode.config.seed,
            "scenario_family": self._episode.config.scenario_family,
            "difficulty": self._episode.config.difficulty,
            "anomaly_density": self._episode.config.anomaly_density,
            "anomaly_count": self._episode.config.anomaly_count,
        }
        return MetricTrackerRlObservation(
            task_id=self._active_task.task_id,
            status=status,
            message=message,
            instruction=self._active_task.instruction,
            conversion_metric_definitions=list(self._generator.config.conversion_definitions),
            available_synthetic_generator_methods=available_synthetic_generator_methods(),
            applied_synthetic_generators=(
                self._episode.applied_synthetic_generators
                if self._debug_mode or self._expose_applied_generators
                else []
            ),
            available_methods=available_analysis_methods(),
            available_tasks=available_task_specs(),
            daily_metrics=[],
            hourly_metrics=[],
            analysis_result=analysis_result,
            generated_rows=generated_rows or [],
            submitted_rows=submitted_rows or [],
            submission_preview=submission_preview,
            submission_issues=issues or [],
            reward_breakdown=reward_breakdown,
            expected_row_count=len(self._episode.expected_rows),
            correct_row_count=correct_row_count,
            reward=reward,
            done=done,
            config=metadata,
            debug=(
                {
                    "task_id": self._active_task.task_id,
                    "expected_payload": [row.model_dump() for row in self._episode.expected_rows],
                    "anomaly_schedule": self._episode.anomaly_schedule,
                    "reward_breakdown": reward_breakdown.model_dump() if reward_breakdown else None,
                    "issues": [item.model_dump() for item in (issues or [])],
                }
                if self._debug_mode
                else None
            ),
        )

    def _run_analysis(self, method_name: str, arguments: dict) -> dict:
        toolkit = SharedAnalysisToolkit(
            AnalysisContext(
                daily_metrics=self._episode.daily_metrics,
                hourly_metrics=self._episode.hourly_metrics,
                conversion_definitions=list(self._generator.config.conversion_definitions),
                instruction=self._active_task.instruction,
                config={
                    "task_id": self._active_task.task_id,
                    "objective": self._active_task.objective,
                    "grader_name": self._active_task.grader_name,
                    **self._episode.config.__dict__,
                },
            )
        )
        if method_name == "task_overview":
            result = toolkit.task_overview()
        elif method_name == "list_dates":
            result = toolkit.list_dates()
        elif method_name == "list_entities":
            result = toolkit.list_entities()
        elif method_name == "rows_for_date":
            result = toolkit.rows_for_date(arguments["date"])
        elif method_name == "hourly_rows_for_date":
            result = toolkit.hourly_rows_for_date(arguments["date"])
        elif method_name == "compare_rate_to_median":
            result = toolkit.compare_rate_to_median(arguments["date"], arguments["entity_name"])
        elif method_name == "compare_count_to_median":
            result = toolkit.compare_count_to_median(arguments["date"], arguments["entity_name"])
        elif method_name == "detect_funnel_break":
            result = toolkit.detect_funnel_break(arguments["date"])
        elif method_name == "check_impossible_counts":
            result = toolkit.check_impossible_counts(arguments["date"])
        elif method_name == "list_suspicious_dates":
            result = toolkit.list_suspicious_dates(limit=arguments.get("limit", 10))
        elif method_name == "preview_submission":
            result = toolkit.preview_submission(arguments.get("rows", []))
        elif method_name == "show_raw_data":
            result = toolkit.show_raw_data(limit=arguments.get("limit", 5))
        elif method_name == "get_metric_median":
            result = toolkit.get_metric_median_multi(
                metric_name=arguments.get("metric_name"),
                metric_names=arguments.get("metric_names", []),
            )
        elif method_name == "get_metric_std_dev_from_median":
            result = toolkit.get_metric_std_dev_from_median_multi(
                metric_name=arguments.get("metric_name"),
                metric_names=arguments.get("metric_names", []),
            )
        elif method_name == "get_rows_with_abs_diff_from_median_gt":
            result = toolkit.get_rows_with_abs_diff_from_median_gt_multi(
                metric_name=arguments.get("metric_name"),
                metric_names=arguments.get("metric_names", []),
                threshold=float(arguments["threshold"]),
            )
        elif method_name == "get_median_filter_rows":
            result = toolkit.get_median_filter_rows_multi(
                metric_name=arguments.get("metric_name"),
                metric_names=arguments.get("metric_names", []),
                threshold_multiplier=float(arguments["threshold_multiplier"]),
            )
        elif method_name == "get_rate_drop_from_median_rows":
            result = toolkit.get_rate_drop_from_median_rows(
                metric_name=arguments.get("metric_name"),
                metric_names=arguments.get("metric_names", []),
                threshold_multiplier=float(arguments["threshold_multiplier"]),
            )
        elif method_name == "get_rate_spike_from_median_rows":
            result = toolkit.get_rate_spike_from_median_rows(
                metric_name=arguments.get("metric_name"),
                metric_names=arguments.get("metric_names", []),
                threshold_multiplier=float(arguments["threshold_multiplier"]),
            )
        elif method_name == "get_absolute_drop_in_event_count_rows":
            result = toolkit.get_absolute_drop_in_event_count_rows(
                metric_name=arguments.get("metric_name"),
                metric_names=arguments.get("metric_names", []),
                threshold_multiplier=float(arguments["threshold_multiplier"]),
            )
        elif method_name == "get_absolute_spike_in_event_count_rows":
            result = toolkit.get_absolute_spike_in_event_count_rows(
                metric_name=arguments.get("metric_name"),
                metric_names=arguments.get("metric_names", []),
                threshold_multiplier=float(arguments["threshold_multiplier"]),
            )
        elif method_name == "get_funnel_break_rows":
            result = toolkit.get_funnel_break_rows(
                threshold_multiplier=float(arguments["threshold_multiplier"]),
            )
        elif method_name == "get_hourly_traffic_mix_shift_rows":
            result = toolkit.get_hourly_traffic_mix_shift_rows(
                threshold_multiplier=float(arguments["threshold_multiplier"]),
            )
        elif method_name == "get_instrumentation_data_quality_issue_rows":
            result = toolkit.get_instrumentation_data_quality_issue_rows(
                threshold_multiplier=float(arguments["threshold_multiplier"]),
            )
        elif method_name == "payload_generator":
            result = toolkit.payload_generator(arguments.get("generator_methods", []))
        else:
            raise ValueError(f"Unsupported analysis method: {method_name}")

        return {
            "method": method_name,
            "arguments": arguments,
            "result": result,
        }

    @staticmethod
    def _submission_message(result) -> str:
        if result.is_perfect:
            return "Submission is fully correct."
        extra_issues = [issue for issue in result.issues if issue.issue_type == "extra_row"]
        missing_count = result.reward_breakdown.missing_rows
        if not extra_issues and missing_count > 0:
            return (
                "All submitted rows are anomalies, but a few are missing. "
                f"Missing value count: {missing_count}."
            )
        if extra_issues:
            first = extra_issues[0]
            return f"Specific row is not an anomaly: {first.row_key}."
        return (
            f"Matched {result.reward_breakdown.matched_rows}/"
            f"{result.reward_breakdown.expected_rows} expected rows. Review the feedback."
        )
