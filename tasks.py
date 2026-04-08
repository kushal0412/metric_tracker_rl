"""Named benchmark tasks and deterministic task graders."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from .evaluation import EvaluationConfig, EvaluationResult, evaluate_submission
    from .models import BenchmarkTaskSpec, MetricSubmissionRow
    from .server.data_generator import EpisodeConfig
except ImportError:
    from evaluation import EvaluationConfig, EvaluationResult, evaluate_submission
    from models import BenchmarkTaskSpec, MetricSubmissionRow
    from server.data_generator import EpisodeConfig


DEFAULT_GRADER_NAME = "deterministic_exact_match"


@dataclass(frozen=True)
class TaskSpec:
    """A concrete benchmark task that an agent can solve and be graded on."""

    task_id: str
    difficulty: str
    instruction: str
    objective: str
    seed: int
    scenario_family: str
    anomaly_density: str
    anomaly_count: int
    grader_name: str = DEFAULT_GRADER_NAME
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)

    def build_episode_config(self) -> EpisodeConfig:
        """Return the canonical episode configuration for this task."""
        return EpisodeConfig(
            seed=self.seed,
            scenario_family=self.scenario_family,
            difficulty=self.difficulty,
            anomaly_density=self.anomaly_density,
            anomaly_count=self.anomaly_count,
        ).normalized()

    def grade_submission(
        self,
        submitted_rows: list[dict] | list[MetricSubmissionRow],
        expected_rows: list[MetricSubmissionRow],
        *,
        config: EvaluationConfig | None = None,
        include_debug_expected: bool = False,
    ) -> EvaluationResult:
        """Grade one candidate submission for this task."""
        return evaluate_submission(
            submitted_rows,
            expected_rows,
            config=config or self.evaluation_config,
            include_debug_expected=include_debug_expected,
        )

    def to_model(self) -> BenchmarkTaskSpec:
        """Return a typed summary safe to expose in observations."""
        return BenchmarkTaskSpec(
            task_id=self.task_id,
            difficulty=self.difficulty,
            instruction=self.instruction,
            objective=self.objective,
            scenario_family=self.scenario_family,
            anomaly_density=self.anomaly_density,
            anomaly_count=self.anomaly_count,
            grader_name=self.grader_name,
        )


TASKS: dict[str, TaskSpec] = {
    "easy_single_spike": TaskSpec(
        task_id="easy_single_spike",
        difficulty="easy",
        instruction=(
            "Investigate the seeded funnel dataset and submit every anomalous row. "
            "Use the shared analysis methods before submitting."
        ),
        objective=(
            "Find all anomalies and submit every correctly populated anomaly row."
        ),
        seed=11,
        scenario_family="rate_spike_from_median",
        anomaly_density="low",
        anomaly_count=2,
    ),
    "medium_mixed_pair": TaskSpec(
        task_id="medium_mixed_pair",
        difficulty="medium",
        instruction=(
            "Investigate the seeded funnel dataset and submit every anomalous row. "
            "Expect both event-count and conversion-rate reasoning."
        ),
        objective=(
            "Find the full set of medium-difficulty anomalies without submitting extras."
        ),
        seed=23,
        scenario_family="mixed",
        anomaly_density="medium",
        anomaly_count=3,
    ),
    "hard_mixed_multi": TaskSpec(
        task_id="hard_mixed_multi",
        difficulty="hard",
        instruction=(
            "Investigate the seeded funnel dataset and submit every anomalous row. "
            "Some anomalies are subtle, so use the analysis methods carefully and avoid over-submitting."
        ),
        objective=(
            "Recover the complete set of hard mixed anomalies while preserving precision."
        ),
        seed=37,
        scenario_family="mixed",
        anomaly_density="high",
        anomaly_count=4,
    ),
}

DEFAULT_TASK_ORDER: tuple[str, ...] = (
    "easy_single_spike",
    "medium_mixed_pair",
    "hard_mixed_multi",
)
DEFAULT_TASK_ID = DEFAULT_TASK_ORDER[0]


def get_task_spec(task_id: str) -> TaskSpec:
    """Return the task spec for a known task id."""
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported task_id: {task_id}") from exc


def available_task_specs() -> list[BenchmarkTaskSpec]:
    """Return typed summaries for all named benchmark tasks."""
    return [TASKS[task_id].to_model() for task_id in DEFAULT_TASK_ORDER]
