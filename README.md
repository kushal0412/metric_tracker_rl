---
title: Metric Tracker RL
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - analytics
  - anomaly-detection
---

# Metric Tracker RL

`metric_tracker_rl` is an OpenEnv benchmark for investigating synthetic product-funnel metrics and submitting a structured anomaly report. It is designed to run as a containerized Hugging Face Space and exposes the same environment through both an OpenEnv-compatible HTTP API and a Gradio debugger.

## Environment Description And Motivation

This environment models a common analytics workflow: a team notices a KPI shift, inspects daily and hourly aggregates, compares observed values to historical baselines, and decides which anomalies are real enough to report. The benchmark focuses on disciplined investigation rather than raw generation. Agents must use safe analysis tools, avoid over-submitting, and produce a precise anomaly payload that matches hidden seeded ground truth.

The motivation for the benchmark is to test whether an agent can:

- navigate a realistic tabular analytics task without direct oracle access
- combine count-based, rate-based, funnel, and hourly reasoning
- preserve precision when multiple anomaly families may be present
- translate evidence into a stable machine-graded submission format

Each reset creates a deterministic four-week synthetic dataset with daily and hourly funnel aggregates. Hidden anomaly labels are derived from the reset configuration, so tasks are reproducible and programmatically graded.

## Action Space

The environment accepts `MetricTrackerRlAction` with three fields:

- `classifications`: final anomaly rows to grade
- `analysis_method`: optional safe method name to call instead of grading
- `analysis_args`: arguments for the selected analysis method
- `payload_generators`: optional declarative generator methods that create submission rows inside the environment

Each `classifications` row must include:

- `date`: ISO date in `YYYY-MM-DD`
- `entity_type`: one of the stable families such as `conversion_rate`, `event_count`, `funnel_step`, `hourly_mix`, or `data_quality`
- `entity_name`: stable metric or entity identifier
- `anomaly_type`: anomaly family identifier
- `detection_method`: analysis method used to justify the row
- `baseline_value`: historical reference value
- `observed_value`: measured anomalous value
- `delta_value`: `observed_value - baseline_value`
- `severity`: one of `low`, `medium`, `high`, or `critical`

## Observation Space

The environment returns `MetricTrackerRlObservation`, which includes:

- task metadata: `task_id`, `instruction`, `status`, and visible episode config
- method surface: `available_methods` and `available_synthetic_generator_methods`
- task catalog: `available_tasks`
- metric definitions: `conversion_metric_definitions`
- latest tool output: `analysis_result`
- latest submission output: `generated_rows`, `submitted_rows`, `submission_preview`, `submission_issues`, and `reward_breakdown`
- progress counters: `expected_row_count` and `correct_row_count`

In standard benchmark mode, raw `daily_metrics`, raw `hourly_metrics`, and hidden debug payloads are not exposed directly. Agents are expected to inspect the data through the read-only shared analysis methods instead.

## Shared Analysis Surface

Humans in the Gradio debugger and agents in `inference.py` use the same read-only analysis surface:

- `task_overview`
- `list_dates`
- `list_entities`
- `rows_for_date`
- `hourly_rows_for_date`
- `compare_rate_to_median`
- `compare_count_to_median`
- `detect_funnel_break`
- `check_impossible_counts`
- `list_suspicious_dates`
- `preview_submission`
- payload-generator helpers such as `get_median_filter_rows`

This keeps the benchmark focused on investigation quality rather than privileged access.

## Tasks And Expected Difficulty

The benchmark ships with three named deterministic tasks:

1. `easy_single_spike`
   Expected difficulty: easy.
   One obvious event-count spike is present. A careful single-method investigation should usually be enough.
2. `medium_mixed_pair`
   Expected difficulty: medium.
   Three anomalies are present across mixed count and rate signals. Precision matters because over-submission is penalized.
3. `hard_mixed_multi`
   Expected difficulty: hard.
   Five anomalies are present with higher density and weaker signal separation. Agents need broader exploration and tighter filtering.

Supported anomaly families across resets:

- `rate_drop_from_median`
- `rate_spike_from_median`
- `absolute_drop_in_event_count`
- `absolute_spike_in_event_count`
- `funnel_break`
- `hourly_traffic_mix_shift`
- `instrumentation_data_quality_issue`

## Reward And Grading

Grading is deterministic and normalized to `[0, 1]`. The evaluator rewards:

- precision
- recall
- correct `anomaly_type`
- correct `detection_method`
- numeric accuracy for `baseline_value`, `observed_value`, and `delta_value` within tolerance
- correct `severity`

Penalties apply for:

- extra rows
- duplicate rows
- invalid rows
- exploit-style mass submission patterns

The observation exposes `submission_preview`, `submission_issues`, and `reward_breakdown` after a graded step.

## Baseline Scores

Reference scores below were measured locally with a deterministic scripted payload-generator baseline that submits:

- `easy_single_spike`: `get_absolute_spike_in_event_count_rows(threshold_multiplier=2.0)`
- `medium_mixed_pair`: `get_median_filter_rows(threshold_multiplier=2.0)`
- `hard_mixed_multi`: `get_median_filter_rows(threshold_multiplier=2.0)`

Measured normalized scores:

- `easy_single_spike`: `1.000000`
- `medium_mixed_pair`: `0.662500`
- `hard_mixed_multi`: `0.421818`
- average across named tasks: `0.694773`

These numbers are useful as a simple non-LLM reference point, not as a ceiling. A perfect submission still scores `1.0` on each task.

## Hugging Face Space Deployment

This repository is configured for a containerized Hugging Face Space:

- `README.md` frontmatter sets `sdk: docker`
- the Space is tagged with `openenv`
- [`openenv.yaml`](/Users/kushaljaisinghani/Documents/sample_envs/metric_tracker_rl/openenv.yaml) points to `server.app:app`
- [`Dockerfile`](/Users/kushaljaisinghani/Documents/sample_envs/metric_tracker_rl/Dockerfile) starts the OpenEnv HTTP server on port `8000`

## Setup

### Local Python Setup

```bash
cd metric_tracker_rl
uv sync
```

### Run The Environment Locally

```bash
cd metric_tracker_rl
uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run The Inference Baseline

Set credentials in [`.env.inference`](/Users/kushaljaisinghani/Documents/sample_envs/metric_tracker_rl/.env.inference), then run:

```bash
cd metric_tracker_rl
source .env.inference
uv run python inference.py
```

The inference baseline runs:

- `easy_single_spike`
- `medium_mixed_pair`
- `hard_mixed_multi`

It prints one score per task and an overall average benchmark score.

## Container Build And Run

Build the image:

```bash
cd metric_tracker_rl
docker build -t metric-tracker-rl .
```

Run the container:

```bash
docker run --rm -p 8000:8000 metric-tracker-rl
```

Once running, the Space-compatible server is available at `http://localhost:8000`.

## Validation

Useful checks:

```bash
cd metric_tracker_rl
openenv validate .
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Manual Debugging UI

The bundled Gradio UI exposes:

- named-task selection
- reset controls for `seed`, `scenario_family`, `difficulty`, and `anomaly_density`
- the same shared analysis methods used by the agent baseline
- payload preview and submission feedback
- charts for daily counts, rates, hourly metrics, and funnel shape

Debug mode can expose expected rows and anomaly schedules for development, but that view is intentionally gated and is not part of standard benchmark play.
