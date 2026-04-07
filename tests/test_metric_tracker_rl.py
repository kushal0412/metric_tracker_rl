from __future__ import annotations

from metric_tracker_rl.analysis_tools import AnalysisContext, SharedAnalysisToolkit
from metric_tracker_rl.evaluation import evaluate_submission
from metric_tracker_rl.models import MetricSubmissionRow
from metric_tracker_rl.server.data_generator import ALL_SCENARIO_FAMILIES, EpisodeConfig, MetricDataGenerator
from metric_tracker_rl.server.metric_tracker_rl_environment import MetricTrackerRlEnvironment
from metric_tracker_rl import MetricTrackerRlAction
from metric_tracker_rl.models import PayloadGeneratorMethod
from metric_tracker_rl.tasks import DEFAULT_TASK_ORDER, TASKS, get_task_spec


def _toolkit_for(seed: int = 11, scenario_family: str = "mixed") -> tuple[SharedAnalysisToolkit, list[MetricSubmissionRow]]:
    generator = MetricDataGenerator()
    episode = generator.generate_episode(
        EpisodeConfig(
            seed=seed,
            scenario_family=scenario_family,
            difficulty="medium",
            anomaly_density="medium",
            anomaly_count=5,
        )
    )
    toolkit = SharedAnalysisToolkit(
        AnalysisContext(
            daily_metrics=episode.daily_metrics,
            hourly_metrics=episode.hourly_metrics,
            conversion_definitions=list(generator.config.conversion_definitions),
            config=episode.config.__dict__,
        )
    )
    return toolkit, episode.expected_rows


def test_seed_reproducibility():
    generator = MetricDataGenerator()
    config = EpisodeConfig(seed=17, scenario_family="mixed", difficulty="hard", anomaly_density="high")
    first = generator.generate_episode(config)
    second = generator.generate_episode(config)

    assert [row.model_dump() for row in first.daily_metrics] == [row.model_dump() for row in second.daily_metrics]
    assert [row.model_dump() for row in first.hourly_metrics] == [row.model_dump() for row in second.hourly_metrics]
    assert [row.model_dump() for row in first.expected_rows] == [row.model_dump() for row in second.expected_rows]


def test_anomaly_variety():
    generator = MetricDataGenerator()
    family_results = {}
    for family in ALL_SCENARIO_FAMILIES[1:]:
        episode = generator.generate_episode(
            EpisodeConfig(
                seed=7,
                scenario_family=family,
                difficulty="medium",
                anomaly_density="medium",
                anomaly_count=5,
            )
        )
        family_results[family] = {row.anomaly_type for row in episode.expected_rows}

    assert family_results["rate_drop_from_median"] == {"rate_drop_from_median"}
    assert family_results["rate_spike_from_median"] == {"rate_spike_from_median"}
    assert family_results["absolute_drop_in_event_count"] == {"absolute_drop_in_event_count"}
    assert family_results["absolute_spike_in_event_count"] == {"absolute_spike_in_event_count"}
    assert family_results["funnel_break"] == {"funnel_break"}
    assert family_results["hourly_traffic_mix_shift"] == {"hourly_traffic_mix_shift"}
    assert family_results["instrumentation_data_quality_issue"] == {"instrumentation_data_quality_issue"}

    mixed = generator.generate_episode(
        EpisodeConfig(
            seed=7,
            scenario_family="mixed",
            difficulty="medium",
            anomaly_density="medium",
            anomaly_count=5,
        )
    )
    assert len(mixed.expected_rows) == 5
    assert {row.anomaly_type for row in mixed.expected_rows}.issubset(
        {
            "rate_drop_from_median",
            "rate_spike_from_median",
            "absolute_drop_in_event_count",
            "absolute_spike_in_event_count",
        }
    )
    assert len({row.anomaly_type for row in mixed.expected_rows}) >= 2


def test_evaluator_scores_perfect_submission():
    _, expected_rows = _toolkit_for()
    result = evaluate_submission(expected_rows, expected_rows)

    assert result.is_perfect is True
    assert result.reward_breakdown.total_score == 1.0
    assert result.reward_breakdown.extra_rows == 0
    assert result.reward_breakdown.duplicate_rows == 0
    assert result.reward_breakdown.invalid_rows == 0


def test_named_task_registry_covers_easy_medium_hard():
    assert DEFAULT_TASK_ORDER == (
        "easy_single_spike",
        "medium_mixed_pair",
        "hard_mixed_multi",
    )
    assert len(TASKS) == 3
    assert {TASKS[task_id].difficulty for task_id in DEFAULT_TASK_ORDER} == {"easy", "medium", "hard"}
    assert all(TASKS[task_id].grader_name for task_id in DEFAULT_TASK_ORDER)


def test_task_grader_scores_perfect_submission():
    generator = MetricDataGenerator()
    task = get_task_spec("medium_mixed_pair")
    episode = generator.generate_episode(task.build_episode_config())

    result = task.grade_submission(episode.expected_rows, episode.expected_rows)

    assert result.is_perfect is True
    assert result.reward_breakdown.total_score == 1.0


def test_duplicate_and_extra_rows_are_penalized():
    _, expected_rows = _toolkit_for()
    extra_row = MetricSubmissionRow(
        date=expected_rows[0].date,
        entity_type="event_count",
        entity_name="nonexistent_metric",
        anomaly_type="absolute_spike_in_event_count",
        detection_method="compare_count_to_median",
        baseline_value=100.0,
        observed_value=120.0,
        delta_value=20.0,
        severity="low",
    )
    submitted = [expected_rows[0], expected_rows[0], extra_row]
    result = evaluate_submission(submitted, expected_rows)

    assert result.is_perfect is False
    assert result.reward_breakdown.duplicate_rows == 1
    assert result.reward_breakdown.extra_rows == 1
    assert result.reward_breakdown.total_score < 1.0


def test_shared_methods_behave_consistently():
    toolkit, expected_rows = _toolkit_for(seed=3, scenario_family="mixed")
    overview = toolkit.task_overview()
    suspicious = toolkit.list_suspicious_dates(limit=5)
    first_row = expected_rows[0]

    assert overview["payload_schema"][0] == "date"
    method_names = {item["name"] for item in overview["available_methods"]}
    assert "show_raw_data" in method_names
    assert "get_median_filter_rows" in method_names
    assert "get_funnel_break_rows" in method_names
    assert "get_hourly_traffic_mix_shift_rows" in method_names
    assert "get_instrumentation_data_quality_issue_rows" in method_names
    assert "payload_generator" in method_names
    assert len(suspicious["dates"]) == 5

    if first_row.detection_method == "compare_rate_to_median":
        result = toolkit.compare_rate_to_median(first_row.date, first_row.entity_name)
        assert result["anomaly_type"] == first_row.anomaly_type
    elif first_row.detection_method == "compare_count_to_median":
        result = toolkit.compare_count_to_median(first_row.date, first_row.entity_name)
        assert result["anomaly_type"] == first_row.anomaly_type
    elif first_row.detection_method == "detect_funnel_break":
        result = toolkit.detect_funnel_break(first_row.date)
        assert any(item["entity_name"] == first_row.entity_name for item in result["candidates"])
    elif first_row.detection_method == "check_impossible_counts":
        result = toolkit.check_impossible_counts(first_row.date)
        assert result["issue_count"] > 0
    else:
        result = toolkit.hourly_rows_for_date(first_row.date)
        assert result["found"] is True

    raw = toolkit.show_raw_data(limit=3)
    assert raw["returned_rows"] == 3
    median_stats = toolkit.get_metric_median("app_open_to_order_placed")
    std_stats = toolkit.get_metric_std_dev_from_median("app_open_to_order_placed")
    assert median_stats["sample_size"] > 0
    assert std_stats["std_dev_from_median"] >= 0


def test_debug_mode_is_gated():
    env = MetricTrackerRlEnvironment()
    observation = env.reset()

    assert observation.debug is None
    assert observation.daily_metrics == []
    assert observation.hourly_metrics == []

    try:
        env.export_debug_snapshot()
    except RuntimeError as exc:
        assert "Debug mode is disabled" in str(exc)
    else:
        raise AssertionError("Expected debug snapshot to be gated.")

    env.set_debug_mode(True)
    debug_observation = env.reset()
    snapshot = env.export_debug_snapshot()

    assert debug_observation.debug is not None
    assert "expected_payload" in snapshot
    assert "applied_synthetic_generators" in snapshot


def test_reset_exposes_synthetic_generator_metadata():
    env = MetricTrackerRlEnvironment()
    observation = env.reset()

    assert observation.task_id == "easy_single_spike"
    assert len(observation.available_tasks) == 3
    assert observation.available_synthetic_generator_methods
    assert observation.available_synthetic_generator_methods[0].name == "metric_stddev_shift"
    assert observation.applied_synthetic_generators == []


def test_named_task_reset_updates_instruction_and_config():
    env = MetricTrackerRlEnvironment()
    observation = env.reset(task_id="hard_mixed_multi")

    assert observation.task_id == "hard_mixed_multi"
    assert observation.config["task_id"] == "hard_mixed_multi"
    assert observation.config["grader_name"] == "deterministic_exact_match"
    assert observation.config["difficulty"] == "hard"
    assert observation.instruction == get_task_spec("hard_mixed_multi").instruction


def test_custom_reset_anomalies_support_specific_dates_and_stddev_factor():
    env = MetricTrackerRlEnvironment()
    observation = env.reset(
        seed=21,
        scenario_family="mixed",
        anomaly_count=2,
        anomalies=[
            {
                "method_name": "metric_stddev_shift",
                "metric_name": "orders_placed",
                "date": "2026-03-20",
                "stddev_factor": 2.5,
                "direction": "down",
            },
            {
                "method_name": "metric_stddev_shift",
                "metric_name": "app_open_to_order_placed",
                "date": "2026-03-25",
                "stddev_factor": 2.0,
                "direction": "up",
            },
        ],
    )

    applied = {item.date: item for item in observation.applied_synthetic_generators}
    assert "2026-03-20" in applied
    assert "2026-03-25" in applied
    assert applied["2026-03-20"].metric_name == "orders_placed"
    assert applied["2026-03-20"].stddev_factor == 2.5
    assert applied["2026-03-20"].threshold_value == round(
        applied["2026-03-20"].std_dev_from_median * 2.5,
        4,
    )
    assert applied["2026-03-25"].metric_type == "conversion_rate"


def test_analysis_methods_run_through_step_api():
    env = MetricTrackerRlEnvironment()
    env.reset()
    analyzed = env.step(
        MetricTrackerRlAction(
            analysis_method="list_suspicious_dates",
            analysis_args={"limit": 3},
        )
    )

    assert analyzed.analysis_result is not None
    assert analyzed.analysis_result["method"] == "list_suspicious_dates"
    assert len(analyzed.analysis_result["result"]["dates"]) == 3


def test_payload_generator_method_creates_rows():
    toolkit, _ = _toolkit_for(seed=5, scenario_family="mixed")
    result = toolkit.get_median_filter_rows("app_open_to_order_placed", 2.0)
    assert result["details"][0]["threshold"] >= 0
    assert isinstance(result["generated_rows"], list)


def test_payload_generator_method_without_metric_runs_all_metrics():
    toolkit, _ = _toolkit_for(seed=5, scenario_family="mixed")
    result = toolkit.get_median_filter_rows_multi(metric_name=None, metric_names=[], threshold_multiplier=2.0)
    assert "app_opens" in result["metric_names"]
    assert "app_open_to_order_placed" in result["metric_names"]
    assert isinstance(result["generated_rows"], list)


def test_family_specific_generator_methods_create_matching_anomaly_types():
    cases = [
        ("rate_drop_from_median", "get_rate_drop_from_median_rows", 1.5),
        ("rate_spike_from_median", "get_rate_spike_from_median_rows", 1.5),
        ("absolute_drop_in_event_count", "get_absolute_drop_in_event_count_rows", 1.5),
        ("absolute_spike_in_event_count", "get_absolute_spike_in_event_count_rows", 1.5),
        ("funnel_break", "get_funnel_break_rows", 1.0),
        ("hourly_traffic_mix_shift", "get_hourly_traffic_mix_shift_rows", 1.0),
        ("instrumentation_data_quality_issue", "get_instrumentation_data_quality_issue_rows", 1.0),
    ]

    for family, method_name, threshold_multiplier in cases:
        toolkit, _ = _toolkit_for(seed=7, scenario_family=family)
        method = getattr(toolkit, method_name)
        if "rate_" in method_name or "event_count" in method_name:
            result = method(metric_name=None, metric_names=[], threshold_multiplier=threshold_multiplier)
        else:
            result = method(threshold_multiplier=threshold_multiplier)
        assert result["generated_rows"], method_name
        assert {row["anomaly_type"] for row in result["generated_rows"]} == {family}


def test_metric_summary_methods_without_metric_run_all_metrics():
    toolkit, _ = _toolkit_for(seed=5, scenario_family="mixed")
    medians = toolkit.get_metric_median_multi(metric_name=None, metric_names=[])
    stds = toolkit.get_metric_std_dev_from_median_multi(metric_name=None, metric_names=[])
    diffs = toolkit.get_rows_with_abs_diff_from_median_gt_multi(
        metric_name=None,
        metric_names=[],
        threshold=1.0,
    )
    assert "app_opens" in medians["metric_names"]
    assert "app_open_to_order_placed" in stds["metric_names"]
    assert len(medians["results"]) == len(medians["metric_names"])
    assert len(stds["results"]) == len(stds["metric_names"])
    assert len(diffs["results"]) == len(diffs["metric_names"])


def test_generator_submission_path_runs():
    env = MetricTrackerRlEnvironment()
    env.reset()
    result = env.step(
        MetricTrackerRlAction(
            payload_generators=[
                PayloadGeneratorMethod(
                    method_name="get_median_filter_rows",
                    metric_name="app_open_to_order_placed",
                    threshold_multiplier=2.0,
                )
            ]
        )
    )
    assert result.generated_rows is not None
    assert result.status in {"evaluated", "in_progress", "completed"}


def test_generator_submission_path_supports_family_specific_methods():
    env = MetricTrackerRlEnvironment()
    env.reset(task_id="hard_mixed_multi", scenario_family="funnel_break")
    result = env.step(
        MetricTrackerRlAction(
            payload_generators=[
                PayloadGeneratorMethod(
                    method_name="get_funnel_break_rows",
                    threshold_multiplier=1.0,
                )
            ]
        )
    )
    assert result.analysis_result is not None
    assert result.analysis_result["result"]["generated_rows"] is not None
