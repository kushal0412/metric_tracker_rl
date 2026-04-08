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

## How The Agent Should Choose Methods

The intended agent behavior is not "call every tool and submit everything." The benchmark rewards selecting the narrowest useful method for the anomaly family that the evidence supports.

Practical method-selection rules:

- start with `task_overview` to confirm the task shape, expected payload format, and visible config
- use broad discovery methods first when the anomaly family is unclear:
  - `get_median_filter_rows`
  - `list_suspicious_dates`
  - `rows_for_date`
- use targeted confirmation methods once a candidate anomaly is visible:
  - `compare_count_to_median` for event-count spikes or drops
  - `compare_rate_to_median` for conversion-rate shifts
  - `detect_funnel_break` for step-level funnel problems
  - `check_impossible_counts` for instrumentation or impossible-value issues
- use family-specific generator methods when the anomaly family is already clear:
  - `get_absolute_spike_in_event_count_rows`
  - `get_absolute_drop_in_event_count_rows`
  - `get_rate_spike_from_median_rows`
  - `get_rate_drop_from_median_rows`
  - `get_funnel_break_rows`
  - `get_hourly_traffic_mix_shift_rows`
  - `get_instrumentation_data_quality_issue_rows`
- prefer fewer high-confidence rows over broad over-submission because extra rows are penalized
- use `preview_submission` before final submission when manually building rows

In practice, a strong agent usually follows this pattern:

1. Identify which metric family is likely broken.
2. Confirm the exact date and entity with a comparison tool.
3. Generate the smallest plausible payload.
4. Submit only when the evidence is specific enough to justify the row.

## How Payload Generation Works In The Server

The server supports two final submission paths:

- direct row submission with `classifications`
- declarative server-side generation with `payload_generators`

The payload-generator path is usually simpler and more stable because the model chooses methods and thresholds, and the server constructs the final anomaly rows.

Simple flow:

```text
LLM
  -> choose analysis method from available_methods
  -> inspect evidence from analysis_result
  -> choose one or more payload generator methods
  -> submit payload_generators

Server
  -> run payload_generator inside the environment
  -> create normalized submission rows
  -> grade submitted_rows against hidden expected_rows
  -> return reward_breakdown, submission_issues, generated_rows
```

At the server level, the path is:

```text
MetricTrackerRlAction(payload_generators=[...])
  -> environment step
  -> _run_analysis("payload_generator", ...)
  -> generated_rows
  -> grade_submission(submitted_rows, expected_rows)
  -> observation.reward_breakdown + observation.submission_issues
```

This means the LLM is responsible for choosing the right generator method, but the server is responsible for turning that declarative request into actual payload rows and grading them.

## Example Decision Path

Suppose the agent suspects a conversion-rate drop but does not yet know which metric is responsible.

```text
1. task_overview()
2. get_median_filter_rows(metric_names=["app_open_to_order_placed", "app_open_to_payment_successful"], threshold_multiplier=2.0)
3. compare_rate_to_median(date="2026-03-19", entity_name="app_open_to_payment_successful")
4. payload_generator(generator_methods=[
     {
       "method_name": "get_rate_drop_from_median_rows",
       "metric_name": "app_open_to_payment_successful",
       "threshold_multiplier": 2.0
     }
   ])
5. submit_payload_generator(...)
```

What happens conceptually:

- the first broad method narrows the search space
- the comparison method confirms the metric/date pair with a baseline and observed value
- the generator submission asks the server to build the final row in the benchmark's required schema
- the grader scores the generated payload against the hidden expected anomalies

If the feedback reports extra rows or missing rows, the agent should refine the generator choice or threshold rather than blindly adding more methods.

## Tasks And Expected Difficulty

The benchmark ships with three named deterministic tasks:

1. `easy_single_spike`
   Expected difficulty: easy.
   Two rate-spike anomalies are present. A careful targeted investigation should usually be enough.
2. `medium_mixed_pair`
   Expected difficulty: medium.
   Three anomalies are present across mixed count and rate signals. Precision matters because over-submission is penalized.
3. `hard_mixed_multi`
   Expected difficulty: hard.
   Four anomalies are present with higher density and weaker signal separation. Agents need broader exploration and tighter filtering.

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
