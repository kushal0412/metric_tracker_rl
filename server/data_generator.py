"""Synthetic multi-anomaly data generator for the metric tracker RL environment."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import median

try:
    from ..analysis_tools import COUNT_METRICS, FUNNEL_STEPS, SharedAnalysisToolkit, AnalysisContext
    from ..models import (
        ConversionMetricDefinition,
        MethodSpec,
        MetricRecord,
        MetricSubmissionRow,
        SyntheticAnomalyGenerator,
        SyntheticGeneratorApplication,
    )
except ImportError:
    from analysis_tools import COUNT_METRICS, FUNNEL_STEPS, SharedAnalysisToolkit, AnalysisContext
    from models import (
        ConversionMetricDefinition,
        MethodSpec,
        MetricRecord,
        MetricSubmissionRow,
        SyntheticAnomalyGenerator,
        SyntheticGeneratorApplication,
    )


ALL_SCENARIO_FAMILIES: tuple[str, ...] = (
    "mixed",
    "rate_drop_from_median",
    "rate_spike_from_median",
    "absolute_drop_in_event_count",
    "absolute_spike_in_event_count",
    "funnel_break",
    "hourly_traffic_mix_shift",
    "instrumentation_data_quality_issue",
)

SYNTHETIC_GENERATOR_METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        name="metric_stddev_shift",
        description=(
            "Inject a count or conversion anomaly on specific dates by setting the metric to "
            "median +/- stddev_factor * std_dev_from_median."
        ),
        parameters=["metric_name", "metric_names", "date", "dates", "stddev_factor", "direction"],
    ),
)


def available_synthetic_generator_methods() -> list[MethodSpec]:
    """Return supported reset-time synthetic generator methods."""
    return list(SYNTHETIC_GENERATOR_METHOD_SPECS)


@dataclass(frozen=True)
class GeneratorConfig:
    """Configurable parameters for synthetic metric generation."""

    conversion_definitions: tuple[ConversionMetricDefinition, ...] = (
        ConversionMetricDefinition(
            name="app_open_to_menu_open",
            numerator="menu_opens",
            denominator="app_opens",
            description="menu_opens / app_opens * 100",
        ),
        ConversionMetricDefinition(
            name="menu_open_to_product_added_to_cart",
            numerator="product_added_to_cart",
            denominator="menu_opens",
            description="product_added_to_cart / menu_opens * 100",
        ),
        ConversionMetricDefinition(
            name="product_added_to_cart_to_order_placed",
            numerator="orders_placed",
            denominator="product_added_to_cart",
            description="orders_placed / product_added_to_cart * 100",
        ),
        ConversionMetricDefinition(
            name="order_placed_to_payment_successful",
            numerator="payment_successful",
            denominator="orders_placed",
            description="payment_successful / orders_placed * 100",
        ),
        ConversionMetricDefinition(
            name="app_open_to_order_placed",
            numerator="orders_placed",
            denominator="app_opens",
            description="orders_placed / app_opens * 100",
        ),
        ConversionMetricDefinition(
            name="app_open_to_payment_successful",
            numerator="payment_successful",
            denominator="app_opens",
            description="payment_successful / app_opens * 100",
        ),
    )
    num_weeks: int = 4
    end_date_offset_days: int = 1
    base_daily_app_opens: int = 18000
    weekday_factors: tuple[float, ...] = (0.95, 1.0, 1.02, 1.04, 1.06, 1.12, 1.08)
    hourly_weights: tuple[float, ...] = (
        0.010,
        0.008,
        0.007,
        0.007,
        0.010,
        0.018,
        0.028,
        0.040,
        0.050,
        0.055,
        0.058,
        0.060,
        0.058,
        0.056,
        0.054,
        0.052,
        0.054,
        0.060,
        0.072,
        0.078,
        0.075,
        0.060,
        0.038,
        0.025,
    )
    baseline_rates: dict[str, float] = field(
        default_factory=lambda: {
            "menu_opens": 0.63,
            "product_added_to_cart": 0.29,
            "orders_placed": 0.44,
            "payment_successful": 0.91,
        }
    )

    @property
    def num_days(self) -> int:
        return self.num_weeks * 7


@dataclass(frozen=True)
class EpisodeConfig:
    """Per-episode configuration."""

    seed: int = 0
    scenario_family: str = "mixed"
    difficulty: str = "medium"
    anomaly_density: str = "medium"
    anomaly_count: int = 3
    anomaly_generators: tuple[SyntheticAnomalyGenerator, ...] = ()

    def normalized(self) -> "EpisodeConfig":
        family = self.scenario_family if self.scenario_family in ALL_SCENARIO_FAMILIES else "mixed"
        difficulty = self.difficulty if self.difficulty in {"easy", "medium", "hard"} else "medium"
        density = self.anomaly_density if self.anomaly_density in {"low", "medium", "high"} else "medium"
        return EpisodeConfig(
            seed=int(self.seed),
            scenario_family=family,
            difficulty=difficulty,
            anomaly_density=density,
            anomaly_count=max(1, int(self.anomaly_count or 3)),
            anomaly_generators=tuple(self.anomaly_generators or ()),
        )


@dataclass
class PlannedAnomaly:
    """Internal anomaly schedule item."""

    date: str
    anomaly_type: str
    entity_type: str
    entity_name: str
    detection_method: str
    details: dict[str, str]


@dataclass
class EpisodeData:
    """Synthetic dataset and ground truth used for one episode."""

    config: EpisodeConfig
    scenario_label: str
    daily_metrics: list[MetricRecord]
    hourly_metrics: list[MetricRecord]
    expected_rows: list[MetricSubmissionRow]
    anomaly_schedule: list[dict[str, str]]
    applied_synthetic_generators: list[SyntheticGeneratorApplication]


class MetricDataGenerator:
    """Reusable synthetic data generator used by the env and custom UI."""

    def __init__(self, config: GeneratorConfig | None = None, seed: int | None = None) -> None:
        self.config = config or GeneratorConfig()
        self._default_seed = int(seed or 0)

    def generate_episode(self, episode_config: EpisodeConfig | None = None) -> EpisodeData:
        """Generate one seeded episode."""
        config = (episode_config or EpisodeConfig(seed=self._default_seed)).normalized()
        rng = random.Random(config.seed)
        end_date = date.today() - timedelta(days=self.config.end_date_offset_days)
        start_date = end_date - timedelta(days=self.config.num_days - 1)
        base_hourly = self._generate_base_hourly_metrics(start_date, rng, config)
        applied_synthetic_generators: list[SyntheticGeneratorApplication] = []
        if self._use_synthetic_metric_generators(config):
            anomaly_plan, applied_synthetic_generators = self._apply_metric_generators(
                base_hourly,
                rng,
                config,
            )
        else:
            anomaly_plan = self._plan_anomalies(base_hourly, rng, config)
            self._apply_anomalies(base_hourly, anomaly_plan, rng, config)
        daily_metrics, hourly_metrics = self._materialize_metrics(base_hourly)
        if applied_synthetic_generators:
            self._refresh_applied_generator_actuals(
                applied_synthetic_generators,
                daily_metrics,
            )
        expected_rows = self._build_expected_rows(daily_metrics, hourly_metrics, anomaly_plan, config)
        anomaly_schedule = [
            {
                "date": item.date,
                "anomaly_type": item.anomaly_type,
                "entity_type": item.entity_type,
                "entity_name": item.entity_name,
                "detection_method": item.detection_method,
            }
            for item in anomaly_plan
        ]
        return EpisodeData(
            config=config,
            scenario_label=config.scenario_family,
            daily_metrics=daily_metrics,
            hourly_metrics=hourly_metrics,
            expected_rows=expected_rows,
            anomaly_schedule=anomaly_schedule,
            applied_synthetic_generators=applied_synthetic_generators,
        )

    def _use_synthetic_metric_generators(self, episode_config: EpisodeConfig) -> bool:
        if episode_config.anomaly_generators:
            return True
        return episode_config.scenario_family in {
            "mixed",
            "rate_drop_from_median",
            "rate_spike_from_median",
            "absolute_drop_in_event_count",
            "absolute_spike_in_event_count",
        }

    def _generate_base_hourly_metrics(
        self,
        start_date: date,
        rng: random.Random,
        episode_config: EpisodeConfig,
    ) -> dict[str, list[MetricRecord]]:
        hourly: dict[str, list[MetricRecord]] = {}
        difficulty_noise = {"easy": 0.015, "medium": 0.025, "hard": 0.035}[episode_config.difficulty]
        for day_index in range(self.config.num_days):
            current_date = start_date + timedelta(days=day_index)
            date_key = current_date.isoformat()
            weekday_factor = self.config.weekday_factors[current_date.weekday()]
            trend_factor = 1.0 + day_index * 0.0025
            noise_factor = 1.0 + rng.uniform(-0.02, 0.02)
            total_app_opens = round(
                self.config.base_daily_app_opens * weekday_factor * trend_factor * noise_factor
            )
            weights = self._hour_weights(current_date.weekday(), rng)
            hourly_app_opens = self._allocate_total(total_app_opens, weights, rng)
            day_rows: list[MetricRecord] = []
            for hour, app_opens in enumerate(hourly_app_opens):
                menu_rate = self._bounded(
                    self.config.baseline_rates["menu_opens"] * (1.0 + rng.uniform(-difficulty_noise, difficulty_noise)),
                    0.50,
                    0.80,
                )
                cart_rate = self._bounded(
                    self.config.baseline_rates["product_added_to_cart"]
                    * (1.0 + rng.uniform(-difficulty_noise * 1.2, difficulty_noise * 1.2)),
                    0.18,
                    0.42,
                )
                order_rate = self._bounded(
                    self.config.baseline_rates["orders_placed"]
                    * (1.0 + rng.uniform(-difficulty_noise * 1.2, difficulty_noise * 1.2)),
                    0.28,
                    0.62,
                )
                payment_rate = self._bounded(
                    self.config.baseline_rates["payment_successful"]
                    * (1.0 + rng.uniform(-difficulty_noise, difficulty_noise)),
                    0.76,
                    0.99,
                )
                menu_opens = round(app_opens * menu_rate)
                carts = round(menu_opens * cart_rate)
                orders = round(carts * order_rate)
                payments = round(orders * payment_rate)
                day_rows.append(
                    MetricRecord(
                        date=date_key,
                        hour=hour,
                        app_opens=app_opens,
                        menu_opens=menu_opens,
                        product_added_to_cart=carts,
                        orders_placed=orders,
                        payment_successful=payments,
                    )
                )
            hourly[date_key] = day_rows
        return hourly

    def _plan_anomalies(
        self,
        base_hourly: dict[str, list[MetricRecord]],
        rng: random.Random,
        episode_config: EpisodeConfig,
    ) -> list[PlannedAnomaly]:
        dates = sorted(base_hourly)
        candidate_dates = dates[3:-2] if len(dates) > 8 else dates
        family_pool = (
            list(ALL_SCENARIO_FAMILIES[1:])
            if episode_config.scenario_family == "mixed"
            else [episode_config.scenario_family]
        )
        target_count = max(
            1,
            int(
                episode_config.anomaly_count
                or {"low": 3, "medium": 5, "high": 7}[episode_config.anomaly_density]
            ),
        )
        plan: list[PlannedAnomaly] = []
        used_pairs: set[tuple[str, str, str]] = set()
        family_order = family_pool[:]
        rng.shuffle(family_order)
        family_index = 0

        while len(plan) < target_count:
            if family_index >= len(family_order):
                family_order = family_pool[:]
                rng.shuffle(family_order)
                family_index = 0
            anomaly_type = family_order[family_index]
            family_index += 1
            date_key = rng.choice(candidate_dates)
            entity_type, entity_name, detection_method, details = self._pick_entity_for_family(
                anomaly_type,
                rng,
            )
            dedupe_key = (date_key, entity_type, entity_name)
            if dedupe_key in used_pairs:
                continue
            used_pairs.add(dedupe_key)
            plan.append(
                PlannedAnomaly(
                    date=date_key,
                    anomaly_type=anomaly_type,
                    entity_type=entity_type,
                    entity_name=entity_name,
                    detection_method=detection_method,
                    details=details,
                )
            )
        plan.sort(key=lambda item: (item.date, item.entity_type, item.entity_name))
        return plan

    def _pick_entity_for_family(
        self,
        anomaly_type: str,
        rng: random.Random,
    ) -> tuple[str, str, str, dict[str, str]]:
        if anomaly_type in {"rate_drop_from_median", "rate_spike_from_median"}:
            definition = rng.choice(list(self.config.conversion_definitions))
            return (
                "conversion_rate",
                definition.name,
                "compare_rate_to_median",
                {"conversion_name": definition.name},
            )
        if anomaly_type in {"absolute_drop_in_event_count", "absolute_spike_in_event_count"}:
            metric_name = rng.choice(list(COUNT_METRICS))
            return (
                "event_count",
                metric_name,
                "compare_count_to_median",
                {"metric_name": metric_name},
            )
        if anomaly_type == "funnel_break":
            numerator, denominator = rng.choice(list(FUNNEL_STEPS))
            return (
                "funnel_step",
                f"{numerator}_from_{denominator}",
                "detect_funnel_break",
                {"numerator": numerator, "denominator": denominator},
            )
        if anomaly_type == "hourly_traffic_mix_shift":
            return (
                "hourly_mix",
                "app_opens:daytime_share",
                "hourly_rows_for_date",
                {},
            )
        numerator, denominator = rng.choice(list(FUNNEL_STEPS))
        return (
            "data_quality",
            f"{numerator}_lte_{denominator}",
            "check_impossible_counts",
            {"numerator": numerator, "denominator": denominator},
        )

    def _apply_anomalies(
        self,
        hourly: dict[str, list[MetricRecord]],
        plan: list[PlannedAnomaly],
        rng: random.Random,
        episode_config: EpisodeConfig,
    ) -> None:
        difficulty = episode_config.difficulty
        for item in plan:
            rows = hourly[item.date]
            if item.anomaly_type == "rate_drop_from_median":
                self._apply_rate_change(rows, item.details["conversion_name"], rng, difficulty, direction="down")
            elif item.anomaly_type == "rate_spike_from_median":
                self._apply_rate_change(rows, item.details["conversion_name"], rng, difficulty, direction="up")
            elif item.anomaly_type == "absolute_drop_in_event_count":
                self._apply_count_change(rows, item.details["metric_name"], rng, difficulty, direction="down")
            elif item.anomaly_type == "absolute_spike_in_event_count":
                self._apply_count_change(rows, item.details["metric_name"], rng, difficulty, direction="up")
            elif item.anomaly_type == "funnel_break":
                self._apply_funnel_break(rows, item.details["numerator"], item.details["denominator"], rng, difficulty)
            elif item.anomaly_type == "hourly_traffic_mix_shift":
                self._apply_hourly_mix_shift(rows, rng, difficulty)
            elif item.anomaly_type == "instrumentation_data_quality_issue":
                self._apply_data_quality_issue(rows, item.details["numerator"], item.details["denominator"], rng, difficulty)

    def _apply_rate_change(
        self,
        rows: list[MetricRecord],
        conversion_name: str,
        rng: random.Random,
        difficulty: str,
        *,
        direction: str,
    ) -> None:
        definition = next(item for item in self.config.conversion_definitions if item.name == conversion_name)
        multipliers = {
            "easy": (0.74, 1.32),
            "medium": (0.82, 1.22),
            "hard": (0.88, 1.15),
        }[difficulty]
        multiplier = multipliers[0] if direction == "down" else multipliers[1]
        for row in rows:
            denominator_value = getattr(row, definition.denominator)
            observed = round(denominator_value * multiplier * self._base_rate_from_metric(definition.numerator))
            setattr_value = min(max(observed, 0), denominator_value)
            self._set_metric_and_rebalance(row, definition.numerator, setattr_value)

    def _apply_count_change(
        self,
        rows: list[MetricRecord],
        metric_name: str,
        rng: random.Random,
        difficulty: str,
        *,
        direction: str,
    ) -> None:
        multipliers = {
            "easy": (0.58, 1.42),
            "medium": (0.72, 1.28),
            "hard": (0.82, 1.18),
        }[difficulty]
        multiplier = multipliers[0] if direction == "down" else multipliers[1]
        for row in rows:
            original = getattr(row, metric_name)
            updated = max(0, round(original * multiplier))
            self._set_metric_and_rebalance(row, metric_name, updated)

    def _apply_funnel_break(
        self,
        rows: list[MetricRecord],
        numerator: str,
        denominator: str,
        rng: random.Random,
        difficulty: str,
    ) -> None:
        if numerator == "menu_opens":
            return
        drop = {"easy": 0.45, "medium": 0.58, "hard": 0.7}[difficulty]
        for row in rows:
            denominator_value = getattr(row, denominator)
            broken_value = max(0, round(denominator_value * drop))
            self._set_metric_and_rebalance(row, numerator, broken_value)

    def _apply_hourly_mix_shift(
        self,
        rows: list[MetricRecord],
        rng: random.Random,
        difficulty: str,
    ) -> None:
        total = sum(row.app_opens for row in rows)
        if total <= 0:
            return
        shift = {"easy": 0.28, "medium": 0.20, "hard": 0.14}[difficulty]
        boosted_hours = {0, 1, 2, 3, 4, 21, 22, 23}
        weights = []
        for row in rows:
            base = row.app_opens / total
            if row.hour in boosted_hours:
                base *= 1.0 + shift
            elif 9 <= (row.hour or 0) <= 18:
                base *= max(0.2, 1.0 - shift)
            weights.append(base)
        normalized = [value / sum(weights) for value in weights]
        redistributed = self._allocate_total(total, normalized, rng)
        for row, app_opens in zip(rows, redistributed, strict=False):
            row.app_opens = app_opens
            menu_rate = self._ratio(row.menu_opens, max(row.app_opens, 1))
            row.menu_opens = min(row.app_opens, round(app_opens * menu_rate))
            cart_rate = self._ratio(row.product_added_to_cart, max(row.menu_opens, 1))
            row.product_added_to_cart = min(row.menu_opens, round(row.menu_opens * cart_rate))
            order_rate = self._ratio(row.orders_placed, max(row.product_added_to_cart, 1))
            row.orders_placed = min(row.product_added_to_cart, round(row.product_added_to_cart * order_rate))
            payment_rate = self._ratio(row.payment_successful, max(row.orders_placed, 1))
            row.payment_successful = min(row.orders_placed, round(row.orders_placed * payment_rate))

    def _apply_data_quality_issue(
        self,
        rows: list[MetricRecord],
        numerator: str,
        denominator: str,
        rng: random.Random,
        difficulty: str,
    ) -> None:
        affected_hours = {"easy": 5, "medium": 4, "hard": 3}[difficulty]
        for row in rng.sample(rows, k=min(affected_hours, len(rows))):
            denominator_value = getattr(row, denominator)
            violation = max(1, round(denominator_value * {"easy": 0.12, "medium": 0.08, "hard": 0.05}[difficulty]))
            setattr(row, numerator, denominator_value + violation)
            self._rebalance_downstream_from(row, numerator)

    def _apply_metric_generators(
        self,
        hourly: dict[str, list[MetricRecord]],
        rng: random.Random,
        episode_config: EpisodeConfig,
    ) -> tuple[list[PlannedAnomaly], list[SyntheticGeneratorApplication]]:
        generator_specs = self._resolve_metric_generators(hourly, rng, episode_config)
        if not generator_specs:
            return [], []

        daily_metrics, hourly_metrics = self._materialize_metrics(hourly)
        toolkit = SharedAnalysisToolkit(
            AnalysisContext(
                daily_metrics=daily_metrics,
                hourly_metrics=hourly_metrics,
                conversion_definitions=list(self.config.conversion_definitions),
                config=episode_config.__dict__,
            )
        )

        anomaly_plan: list[PlannedAnomaly] = []
        applications: list[SyntheticGeneratorApplication] = []
        seen_pairs: set[tuple[str, str]] = set()
        for spec in generator_specs:
            for date_key in self._resolve_generator_dates(spec, hourly, rng):
                for metric_name in self._resolve_generator_metrics(spec):
                    dedupe_key = (date_key, metric_name)
                    if dedupe_key in seen_pairs:
                        continue
                    seen_pairs.add(dedupe_key)
                    application = self._build_metric_generator_application(
                        toolkit=toolkit,
                        date_key=date_key,
                        metric_name=metric_name,
                        spec=spec,
                        rng=rng,
                    )
                    self._apply_metric_generator_application(hourly[date_key], application)
                    applications.append(application)
                    anomaly_plan.append(
                        PlannedAnomaly(
                            date=date_key,
                            anomaly_type=application.anomaly_type,
                            entity_type=application.metric_type,
                            entity_name=metric_name,
                            detection_method=application.detection_method,
                            details={"metric_name": metric_name},
                        )
                    )
        applications.sort(key=lambda item: (item.date, item.metric_name))
        anomaly_plan.sort(key=lambda item: (item.date, item.entity_type, item.entity_name))
        return anomaly_plan, applications

    def _resolve_metric_generators(
        self,
        hourly: dict[str, list[MetricRecord]],
        rng: random.Random,
        episode_config: EpisodeConfig,
    ) -> list[SyntheticAnomalyGenerator]:
        if episode_config.anomaly_generators:
            return list(episode_config.anomaly_generators)

        dates = sorted(hourly)
        candidate_dates = dates[3:-2] if len(dates) > 8 else dates
        metric_pool = self._metric_pool_for_family(episode_config.scenario_family)
        if not metric_pool:
            return []

        used_pairs: set[tuple[str, str]] = set()
        generated: list[SyntheticAnomalyGenerator] = []
        default_stddev = {"easy": 2.6, "medium": 2.2, "hard": 1.8}[episode_config.difficulty]
        while len(generated) < max(1, episode_config.anomaly_count):
            date_key = rng.choice(candidate_dates)
            metric_name = rng.choice(metric_pool)
            if (date_key, metric_name) in used_pairs:
                continue
            used_pairs.add((date_key, metric_name))
            generated.append(
                SyntheticAnomalyGenerator(
                    method_name="metric_stddev_shift",
                    metric_name=metric_name,
                    date=date_key,
                    stddev_factor=default_stddev,
                    direction=self._default_direction_for_family(episode_config.scenario_family, rng),
                )
            )
        return generated

    def _metric_pool_for_family(self, scenario_family: str) -> list[str]:
        conversion_metrics = [item.name for item in self.config.conversion_definitions]
        if scenario_family in {"rate_drop_from_median", "rate_spike_from_median"}:
            return conversion_metrics
        if scenario_family in {"absolute_drop_in_event_count", "absolute_spike_in_event_count"}:
            return list(COUNT_METRICS)
        if scenario_family == "mixed":
            return list(COUNT_METRICS) + conversion_metrics
        return []

    @staticmethod
    def _default_direction_for_family(scenario_family: str, rng: random.Random) -> str:
        if scenario_family in {"rate_drop_from_median", "absolute_drop_in_event_count"}:
            return "down"
        if scenario_family in {"rate_spike_from_median", "absolute_spike_in_event_count"}:
            return "up"
        return "down" if rng.random() < 0.5 else "up"

    def _resolve_generator_dates(
        self,
        spec: SyntheticAnomalyGenerator,
        hourly: dict[str, list[MetricRecord]],
        rng: random.Random,
    ) -> list[str]:
        dates = [item for item in spec.dates if item in hourly]
        if spec.date and spec.date in hourly:
            dates.append(spec.date)
        if not dates:
            dates = [rng.choice(sorted(hourly))]
        seen = set()
        deduped = []
        for item in dates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _resolve_generator_metrics(self, spec: SyntheticAnomalyGenerator) -> list[str]:
        metrics = [item for item in spec.metric_names if item]
        if spec.metric_name:
            metrics.append(spec.metric_name)
        if not metrics:
            metrics = list(COUNT_METRICS) + [item.name for item in self.config.conversion_definitions]
        seen = set()
        deduped = []
        for item in metrics:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _build_metric_generator_application(
        self,
        *,
        toolkit: SharedAnalysisToolkit,
        date_key: str,
        metric_name: str,
        spec: SyntheticAnomalyGenerator,
        rng: random.Random,
    ) -> SyntheticGeneratorApplication:
        stats = toolkit.get_metric_std_dev_from_median(metric_name)
        descriptor = toolkit._metric_descriptor(metric_name)
        baseline_value = float(stats["median_value"])
        std_dev_from_median = float(stats["std_dev_from_median"])
        pre_applied_value = float(descriptor["per_date_values"][date_key])
        direction = spec.direction if spec.direction != "auto" else ("down" if rng.random() < 0.5 else "up")
        sign = -1.0 if direction == "down" else 1.0
        threshold_value = round(std_dev_from_median * float(spec.stddev_factor), 4)
        metric_type = "event_count" if metric_name in COUNT_METRICS else "conversion_rate"
        if metric_type == "event_count":
            minimum_shift = max(50.0, baseline_value * toolkit._count_threshold_fraction()) * 1.05
            applied_shift = max(threshold_value, round(minimum_shift, 4))
            target_value = max(0.0, baseline_value + sign * applied_shift)
            anomaly_type = "absolute_spike_in_event_count" if sign > 0 else "absolute_drop_in_event_count"
            detection_method = "compare_count_to_median"
        else:
            applied_shift = max(threshold_value, round(toolkit._rate_threshold() * 1.05, 4))
            target_value = self._bounded(baseline_value + sign * applied_shift, 0.0, 100.0)
            anomaly_type = "rate_spike_from_median" if sign > 0 else "rate_drop_from_median"
            detection_method = "compare_rate_to_median"
        return SyntheticGeneratorApplication(
            method_name=spec.method_name,
            date=date_key,
            metric_name=metric_name,
            metric_type=metric_type,
            direction="up" if sign > 0 else "down",
            anomaly_type=anomaly_type,
            detection_method=detection_method,
            baseline_value=round(baseline_value, 4),
            pre_applied_value=round(pre_applied_value, 4),
            std_dev_from_median=round(std_dev_from_median, 4),
            stddev_factor=round(float(spec.stddev_factor), 4),
            threshold_value=threshold_value,
            target_value=round(target_value, 4),
            actual_value=round(target_value, 4),
            formula=(
                f"{metric_name} = median {'+' if sign > 0 else '-'} "
                "max(stddev_factor * std_dev_from_median, detector_threshold)"
            ),
        )

    def _apply_metric_generator_application(
        self,
        rows: list[MetricRecord],
        application: SyntheticGeneratorApplication,
    ) -> None:
        if application.metric_type == "event_count":
            self._apply_daily_count_target(
                rows,
                application.metric_name,
                int(round(application.target_value)),
            )
            return
        self._apply_daily_conversion_target(
            rows,
            application.metric_name,
            float(application.target_value),
        )

    def _apply_daily_count_target(
        self,
        rows: list[MetricRecord],
        metric_name: str,
        target_total: int,
    ) -> None:
        target_total = max(0, target_total)
        current_values = [max(0, getattr(row, metric_name)) for row in rows]
        current_total = sum(current_values)
        if current_total > 0:
            weights = [value / current_total for value in current_values]
        else:
            app_total = sum(max(0, row.app_opens) for row in rows) or len(rows)
            weights = [max(0, row.app_opens) / app_total for row in rows]
        allocated = self._allocate_total(target_total, weights, random.Random(target_total + len(rows)))
        for row, value in zip(rows, allocated, strict=False):
            self._set_metric_and_rebalance(row, metric_name, value)

    def _apply_daily_conversion_target(
        self,
        rows: list[MetricRecord],
        conversion_name: str,
        target_rate_pct: float,
    ) -> None:
        definition = next(item for item in self.config.conversion_definitions if item.name == conversion_name)
        bounded_rate = self._bounded(target_rate_pct / 100.0, 0.0, 1.0)
        for row in rows:
            denominator_value = getattr(row, definition.denominator)
            numerator_target = round(denominator_value * bounded_rate)
            self._set_metric_and_rebalance(row, definition.numerator, numerator_target)

    def _refresh_applied_generator_actuals(
        self,
        applications: list[SyntheticGeneratorApplication],
        daily_metrics: list[MetricRecord],
    ) -> None:
        by_date = {row.date: row for row in daily_metrics}
        conversion_map = {item.name: item for item in self.config.conversion_definitions}
        for application in applications:
            record = by_date.get(application.date)
            if record is None:
                continue
            if application.metric_type == "event_count":
                actual_value = float(getattr(record, application.metric_name))
            else:
                definition = conversion_map[application.metric_name]
                denominator = getattr(record, definition.denominator)
                actual_value = round(
                    (getattr(record, definition.numerator) / denominator * 100.0)
                    if denominator > 0
                    else 0.0,
                    4,
                )
            application.actual_value = round(actual_value, 4)

    def _build_expected_rows(
        self,
        daily_metrics: list[MetricRecord],
        hourly_metrics: list[MetricRecord],
        plan: list[PlannedAnomaly],
        episode_config: EpisodeConfig,
    ) -> list[MetricSubmissionRow]:
        toolkit = SharedAnalysisToolkit(
            AnalysisContext(
                daily_metrics=daily_metrics,
                hourly_metrics=hourly_metrics,
                conversion_definitions=list(self.config.conversion_definitions),
                config={
                    "seed": episode_config.seed,
                    "scenario_family": episode_config.scenario_family,
                    "difficulty": episode_config.difficulty,
                    "anomaly_density": episode_config.anomaly_density,
                    "anomaly_count": episode_config.anomaly_count,
                },
            )
        )
        rows: list[MetricSubmissionRow] = []
        for item in plan:
            if item.detection_method == "compare_rate_to_median":
                result = toolkit.compare_rate_to_median(item.date, item.entity_name)
            elif item.detection_method == "compare_count_to_median":
                result = toolkit.compare_count_to_median(item.date, item.entity_name)
            elif item.detection_method == "detect_funnel_break":
                candidates = toolkit.detect_funnel_break(item.date)["candidates"]
                result = next((row for row in candidates if row["entity_name"] == item.entity_name), None)
                if result is None:
                    numerator = item.details["numerator"]
                    denominator = item.details["denominator"]
                    daily_row = next(row for row in daily_metrics if row.date == item.date)
                    baseline_series = [
                        (getattr(row, numerator) / getattr(row, denominator) * 100.0)
                        if getattr(row, denominator) > 0
                        else 0.0
                        for row in daily_metrics
                    ]
                    baseline = round(median(baseline_series), 4)
                    observed = round(
                        (getattr(daily_row, numerator) / getattr(daily_row, denominator) * 100.0)
                        if getattr(daily_row, denominator) > 0
                        else 0.0,
                        4,
                    )
                    delta = round(observed - baseline, 4)
                    result = {
                        "entity_type": item.entity_type,
                        "entity_name": item.entity_name,
                        "baseline_value": baseline,
                        "observed_value": observed,
                        "delta_value": delta,
                        "severity": self._severity_from_ratio(abs(delta), 5.0, 10.0, 15.0),
                    }
            elif item.detection_method == "check_impossible_counts":
                impossible = toolkit.check_impossible_counts(item.date)
                result = {
                    "date": item.date,
                    "entity_type": item.entity_type,
                    "entity_name": item.entity_name,
                    "anomaly_type": item.anomaly_type,
                    "detection_method": item.detection_method,
                    "baseline_value": 0.0,
                    "observed_value": round(impossible["total_excess"], 4),
                    "delta_value": round(impossible["total_excess"], 4),
                    "severity": self._severity_from_ratio(impossible["total_excess"], 20.0, 60.0, 120.0),
                }
            else:
                observed_share = toolkit.hourly_rows_for_date(item.date)["summary"]["daytime_share"]
                baseline_share = toolkit._median_daytime_share()
                delta = round(observed_share - baseline_share, 4)
                result = {
                    "date": item.date,
                    "entity_type": item.entity_type,
                    "entity_name": item.entity_name,
                    "anomaly_type": item.anomaly_type,
                    "detection_method": item.detection_method,
                    "baseline_value": round(baseline_share, 4),
                    "observed_value": round(observed_share, 4),
                    "delta_value": delta,
                    "severity": self._severity_from_ratio(abs(delta) * 100.0, 10.0, 18.0, 25.0),
                }

            if not result:
                continue
            normalized = dict(result)
            normalized["date"] = item.date
            normalized["anomaly_type"] = item.anomaly_type
            normalized["detection_method"] = item.detection_method
            rows.append(MetricSubmissionRow(**normalized))
        deduped = {f"{row.date}|{row.entity_type}|{row.entity_name}": row for row in rows}
        return sorted(deduped.values(), key=lambda row: (row.date, row.entity_type, row.entity_name))

    def _materialize_metrics(
        self,
        base_hourly: dict[str, list[MetricRecord]],
    ) -> tuple[list[MetricRecord], list[MetricRecord]]:
        hourly_metrics = []
        daily_metrics = []
        for date_key in sorted(base_hourly):
            rows = base_hourly[date_key]
            hourly_metrics.extend(rows)
            daily_metrics.append(
                MetricRecord(
                    date=date_key,
                    hour=None,
                    app_opens=sum(item.app_opens for item in rows),
                    menu_opens=sum(item.menu_opens for item in rows),
                    product_added_to_cart=sum(item.product_added_to_cart for item in rows),
                    orders_placed=sum(item.orders_placed for item in rows),
                    payment_successful=sum(item.payment_successful for item in rows),
                )
            )
        return daily_metrics, hourly_metrics

    def _set_metric_and_rebalance(self, row: MetricRecord, metric_name: str, value: int) -> None:
        caps = {
            "app_opens": None,
            "menu_opens": row.app_opens,
            "product_added_to_cart": row.menu_opens,
            "orders_placed": row.product_added_to_cart,
            "payment_successful": row.orders_placed,
        }
        cap = caps.get(metric_name)
        bounded = max(0, value if cap is None else min(value, cap))
        setattr(row, metric_name, bounded)
        self._rebalance_downstream_from(row, metric_name)
        self._rebalance_upstream_to(row, metric_name)

    def _rebalance_downstream_from(self, row: MetricRecord, metric_name: str) -> None:
        order = list(COUNT_METRICS)
        start_index = order.index(metric_name)
        for index in range(start_index + 1, len(order)):
            parent_name = order[index - 1]
            current_name = order[index]
            parent_value = getattr(row, parent_name)
            current_value = min(getattr(row, current_name), parent_value)
            setattr(row, current_name, max(0, current_value))

    def _rebalance_upstream_to(self, row: MetricRecord, metric_name: str) -> None:
        order = list(COUNT_METRICS)
        start_index = order.index(metric_name)
        for index in range(start_index - 1, -1, -1):
            child_name = order[index + 1]
            current_name = order[index]
            child_value = getattr(row, child_name)
            current_value = max(getattr(row, current_name), child_value)
            setattr(row, current_name, current_value)

    def _base_rate_from_metric(self, metric_name: str) -> float:
        if metric_name == "menu_opens":
            return self.config.baseline_rates["menu_opens"]
        if metric_name == "product_added_to_cart":
            return self.config.baseline_rates["product_added_to_cart"]
        if metric_name == "orders_placed":
            return self.config.baseline_rates["orders_placed"]
        if metric_name == "payment_successful":
            return self.config.baseline_rates["payment_successful"]
        return 1.0

    def _hour_weights(self, weekday: int, rng: random.Random) -> list[float]:
        weekend_multiplier = 1.12 if weekday >= 5 else 1.0
        weights = [
            max(0.001, value * weekend_multiplier * (1.0 + rng.uniform(-0.08, 0.08)))
            for value in self.config.hourly_weights
        ]
        total = sum(weights)
        return [value / total for value in weights]

    @staticmethod
    def _allocate_total(total: int, weights: list[float], rng: random.Random) -> list[int]:
        raw = [total * weight for weight in weights]
        integers = [int(value) for value in raw]
        remainder = total - sum(integers)
        ranked = sorted(
            range(len(weights)),
            key=lambda index: (raw[index] - integers[index], rng.random()),
            reverse=True,
        )
        for index in ranked[:remainder]:
            integers[index] += 1
        return integers

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _bounded(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)

    @staticmethod
    def _severity_from_ratio(value: float, medium: float, high: float, critical: float) -> str:
        if value >= critical:
            return "critical"
        if value >= high:
            return "high"
        if value >= medium:
            return "medium"
        return "low"
