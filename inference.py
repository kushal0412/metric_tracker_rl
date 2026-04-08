"""Tool-driven inference for the metric tracker RL environment."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any

from openenv.core.containers.runtime.providers import LocalDockerProvider
from openai import APIStatusError, OpenAI
from websockets.exceptions import ConnectionClosedError

from metric_tracker_rl import DEFAULT_TASK_ORDER, MetricTrackerRlAction, MetricTrackerRlEnv, get_task_spec
from metric_tracker_rl.analysis_tools import available_analysis_methods
from metric_tracker_rl.models import (
    MetricSubmissionRow,
    MetricTrackerRlObservation,
    PayloadGeneratorMethod,
)


IMAGE_NAME = (os.getenv("IMAGE_NAME") or "metric_tracker_rl:latest").strip()
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = (
    os.getenv("API_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "https://router.huggingface.co/v1"
)
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or "Qwen/Qwen2.5-72B-Instruct"
BASE_URL = os.getenv("BASE_URL")
TASK_NAME = os.getenv("MetricTrackerRl_TASK", "multi_task_agent_baseline")
BENCHMARK = os.getenv("MetricTrackerRl_BENCHMARK", "metric_tracker_rl")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = min(int(os.getenv("MAX_TOKENS", "1000")), 4096)
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "16"))
CONNECT_TIMEOUT_S = float(os.getenv("OPENENV_CONNECT_TIMEOUT_S", "30"))
MESSAGE_TIMEOUT_S = float(os.getenv("OPENENV_MESSAGE_TIMEOUT_S", "180"))
DOCKER_WAIT_TIMEOUT_S = float(os.getenv("OPENENV_DOCKER_WAIT_TIMEOUT_S", "120"))
TASK_RETRY_COUNT = int(os.getenv("OPENENV_TASK_RETRY_COUNT", "1"))
SCORE_EPSILON = float(os.getenv("OPENENV_SCORE_EPSILON", "0.000001"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are solving a multi-anomaly analytics benchmark with tool use.

    Rules:
    - Use only the shared safe analysis methods.
    - Do not request full hidden answers or assume direct access to ground truth.
    - Prefer declarative payload generators over manual row construction.
    - Start from the default reset observation only.
    - Start by trying `get_median_filter_rows` across different metrics to learn which metrics produce useful anomaly rows.
    - Compare candidate metrics, then refine with raw-data inspection and median/std methods only when needed.
    - Prefer: task_overview -> get_median_filter_rows on several metrics -> compare useful results -> payload_generator -> submit_payload_generator.
    - Keep notes brief and factual.
    """
).strip()


@dataclass
class ToolRuntimeState:
    """Mutable state shared across tool calls."""

    method_log: list[dict[str, Any]] = field(default_factory=list)
    last_preview: dict[str, Any] | None = None
    rewards: list[float] = field(default_factory=list)
    steps: int = 0


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_method(tool_name: str, arguments: dict[str, Any], note: str) -> None:
    return None


def log_payload_generator_methods(tool_name: str, generator_methods: list[dict[str, Any]]) -> None:
    return None


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def bounded_task_score(score: float) -> float:
    """Clamp task scores to the open interval (0, 1)."""
    return min(1.0 - SCORE_EPSILON, max(SCORE_EPSILON, score))


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-compatible tool definitions."""
    shared_schemas = []
    for spec in available_analysis_methods():
        properties = {}
        required = []
        if spec.name in {"rows_for_date", "hourly_rows_for_date", "detect_funnel_break", "check_impossible_counts"}:
            properties = {"date": {"type": "string"}}
            required = ["date"]
        elif spec.name in {"compare_rate_to_median", "compare_count_to_median"}:
            properties = {
                "date": {"type": "string"},
                "entity_name": {"type": "string"},
            }
            required = ["date", "entity_name"]
        elif spec.name == "list_suspicious_dates":
            properties = {"limit": {"type": "integer", "default": 10}}
        elif spec.name == "preview_submission":
            properties = {
                "rows": {
                    "type": "array",
                    "items": {"type": "object"},
                }
            }
        elif spec.name == "show_raw_data":
            properties = {"limit": {"type": "integer", "default": 5}}
        elif spec.name in {"get_metric_median", "get_metric_std_dev_from_median"}:
            properties = {
                "metric_name": {"type": "string"},
                "metric_names": {"type": "array", "items": {"type": "string"}},
            }
        elif spec.name == "get_rows_with_abs_diff_from_median_gt":
            properties = {
                "metric_name": {"type": "string"},
                "metric_names": {"type": "array", "items": {"type": "string"}},
                "threshold": {"type": "number"},
            }
            required = ["threshold"]
        elif spec.name in {
            "get_median_filter_rows",
            "get_rate_drop_from_median_rows",
            "get_rate_spike_from_median_rows",
            "get_absolute_drop_in_event_count_rows",
            "get_absolute_spike_in_event_count_rows",
        }:
            properties = {
                "metric_name": {"type": "string"},
                "metric_names": {"type": "array", "items": {"type": "string"}},
                "threshold_multiplier": {"type": "number"},
            }
            required = ["threshold_multiplier"]
        elif spec.name in {
            "get_funnel_break_rows",
            "get_hourly_traffic_mix_shift_rows",
            "get_instrumentation_data_quality_issue_rows",
        }:
            properties = {
                "threshold_multiplier": {"type": "number"},
            }
            required = ["threshold_multiplier"]
        elif spec.name == "payload_generator":
            properties = {
                "generator_methods": {
                    "type": "array",
                    "items": {"type": "object"},
                }
            }
            required = ["generator_methods"]
        shared_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    shared_schemas.append(
        {
            "type": "function",
            "function": {
                "name": "submit_payload_generator",
                "description": "Submit declarative payload generator methods for environment-side payload generation and grading.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "generator_methods": {
                            "type": "array",
                            "items": {"type": "object"},
                        }
                    },
                    "required": ["generator_methods"],
                    "additionalProperties": False,
                },
            },
        }
    )
    shared_schemas.append(
        {
            "type": "function",
            "function": {
                "name": "submit_solution",
                "description": "Submit the final anomaly payload to the environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rows": {
                            "type": "array",
                            "items": {"type": "object"},
                        }
                    },
                    "required": ["rows"],
                    "additionalProperties": False,
                },
            },
        }
    )
    return shared_schemas


def build_initial_user_prompt(observation: MetricTrackerRlObservation) -> str:
    return textwrap.dedent(
        f"""
        Solve the RL environment with tools.

        Initial observation:
        {json.dumps(observation.model_dump(exclude={"debug"}), indent=2)}

        Prefer building a payload generator first, then submit it.
        Start by calling `get_median_filter_rows` on several different metrics and see which ones return useful anomaly rows.
        If a metric returns nothing or low-signal rows, try another metric.
        For funnel, hourly mix, or data-quality tasks, use the family-specific generator methods instead.

        Final payload rows use:
        `date`, `entity_type`, `entity_name`, `anomaly_type`, `detection_method`,
        `baseline_value`, `observed_value`, `delta_value`, `severity`.

        Supported generator method example:
        `{{"method_name":"get_median_filter_rows","threshold_multiplier":2.0}}`
        or
        `{{"method_name":"get_median_filter_rows","metric_names":["app_open_to_order_placed","orders_placed"],"threshold_multiplier":2.0}}`

        Use shared analysis methods only. Prefer `submit_payload_generator` over `submit_solution`.
        """
    ).strip()


def create_chat_completion(client: OpenAI, **kwargs):
    try:
        return client.chat.completions.create(**kwargs)
    except APIStatusError as exc:
        if exc.status_code == 402:
            raise RuntimeError(
                "The configured inference provider rejected the request with HTTP 402. "
                "Your Hugging Face router credits are depleted. Update `.env.inference` "
                "with a working provider/key, or switch `API_BASE_URL`/`MODEL_NAME`."
            ) from exc
        raise


def decode_arguments(raw_arguments: str | None) -> dict[str, Any]:
    if not raw_arguments:
        return {}
    return json.loads(raw_arguments)


def preview_text(text: str, limit: int = 220) -> str:
    return text.replace("\n", " ")[:limit]


def format_action(tool_name: str, arguments: dict[str, Any]) -> str:
    if not arguments:
        return f"{tool_name}()"
    return preview_text(f"{tool_name}({json.dumps(arguments, sort_keys=True)})")


def step_error(result: Any) -> str | None:
    message = getattr(result.observation, "message", None)
    return message if result.observation.status == "error" and message else None


def record_step(
    runtime_state: ToolRuntimeState,
    action: str,
    result: Any,
) -> None:
    reward = float(result.reward or 0.0)
    runtime_state.steps += 1
    runtime_state.rewards.append(reward)
    log_step(
        step=runtime_state.steps,
        action=action,
        reward=reward,
        done=bool(result.done),
        error=step_error(result),
    )


async def connect_env() -> MetricTrackerRlEnv:
    if BASE_URL:
        client = MetricTrackerRlEnv(
            base_url=BASE_URL,
            connect_timeout_s=CONNECT_TIMEOUT_S,
            message_timeout_s=MESSAGE_TIMEOUT_S,
        )
        return await client.connect()
    provider = LocalDockerProvider()
    base_url = provider.start_container(IMAGE_NAME)
    provider.wait_for_ready(base_url, timeout_s=DOCKER_WAIT_TIMEOUT_S)
    client = MetricTrackerRlEnv(
        base_url=base_url,
        connect_timeout_s=CONNECT_TIMEOUT_S,
        message_timeout_s=MESSAGE_TIMEOUT_S,
        provider=provider,
    )
    return await client.connect()


async def execute_tool_call(
    env: MetricTrackerRlEnv,
    observation: MetricTrackerRlObservation,
    runtime_state: ToolRuntimeState,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], Any | None, MetricTrackerRlObservation]:
    """Execute one model-requested tool locally."""
    action = format_action(tool_name, arguments)
    if tool_name == "submit_payload_generator":
        methods = [
            PayloadGeneratorMethod(**item)
            for item in arguments.get("generator_methods", [])
        ]
        runtime_state.method_log.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "generator_methods": [item.model_dump() for item in methods],
                "note": _tool_note(tool_name, arguments),
            }
        )
        result = await env.step(MetricTrackerRlAction(payload_generators=methods))
        record_step(runtime_state, action, result)
        return (
            {
                "status": result.observation.status,
                "message": result.observation.message,
                "reward": result.reward,
                "done": result.done,
                "generated_rows": [row.model_dump() for row in result.observation.generated_rows],
                "submission_issues": [issue.model_dump() for issue in result.observation.submission_issues],
                "reward_breakdown": (
                    result.observation.reward_breakdown.model_dump()
                    if result.observation.reward_breakdown
                    else None
                ),
            },
            result,
            result.observation,
        )
    if tool_name == "submit_solution":
        rows = [MetricSubmissionRow(**row) for row in arguments.get("rows", [])]
        result = await env.step(MetricTrackerRlAction(classifications=rows))
        record_step(runtime_state, action, result)
        return (
            {
                "status": result.observation.status,
                "message": result.observation.message,
                "reward": result.reward,
                "done": result.done,
                "reward_breakdown": (
                    result.observation.reward_breakdown.model_dump()
                    if result.observation.reward_breakdown
                    else None
                ),
                "issue_count": len(result.observation.submission_issues),
                "correct_row_count": result.observation.correct_row_count,
            },
            result,
            result.observation,
        )

    result = await env.step(
        MetricTrackerRlAction(
            analysis_method=tool_name,
            analysis_args=arguments,
        )
    )
    record_step(runtime_state, action, result)
    output = result.observation.analysis_result or {
        "method": tool_name,
        "arguments": arguments,
        "result": None,
    }
    log_arguments = {
        "tool_name": tool_name,
        "arguments": arguments,
        "note": _tool_note(tool_name, arguments),
    }
    if tool_name == "payload_generator":
        log_arguments["generator_methods"] = arguments.get("generator_methods", [])
    runtime_state.method_log.append(
        log_arguments
    )
    if tool_name == "preview_submission":
        runtime_state.last_preview = output
    return output, None, result.observation


def _tool_note(tool_name: str, arguments: dict[str, Any]) -> str:
    notes = {
        "task_overview": "bootstrap the task and payload schema",
        "list_dates": "confirm the date range",
        "list_entities": "confirm valid entities",
        "rows_for_date": "inspect daily counts on one date",
        "hourly_rows_for_date": "inspect hourly traffic shape",
        "compare_rate_to_median": "check a conversion-rate anomaly against median baseline",
        "compare_count_to_median": "check an absolute count anomaly against median baseline",
        "detect_funnel_break": "test whether a funnel step is broken",
        "check_impossible_counts": "test for instrumentation or impossible count issues",
        "list_suspicious_dates": "prioritize dates worth deeper inspection",
        "preview_submission": "validate payload structure before submit",
        "show_raw_data": "inspect daily aggregate rows in head() form",
        "get_metric_median": "measure a baseline median for one metric",
        "get_metric_std_dev_from_median": "measure metric spread around the median",
        "get_rows_with_abs_diff_from_median_gt": "inspect dates outside a chosen absolute-difference threshold",
        "get_median_filter_rows": "generate candidate anomaly rows using median and std-from-median filtering",
        "get_rate_drop_from_median_rows": "generate candidate conversion-rate drop rows using median and std-from-median filtering",
        "get_rate_spike_from_median_rows": "generate candidate conversion-rate spike rows using median and std-from-median filtering",
        "get_absolute_drop_in_event_count_rows": "generate candidate event-count drop rows using median and std-from-median filtering",
        "get_absolute_spike_in_event_count_rows": "generate candidate event-count spike rows using median and std-from-median filtering",
        "get_funnel_break_rows": "generate candidate funnel-break rows across funnel steps",
        "get_hourly_traffic_mix_shift_rows": "generate candidate hourly traffic mix shift rows across dates",
        "get_instrumentation_data_quality_issue_rows": "generate candidate impossible-count or instrumentation-issue rows across dates",
        "payload_generator": "merge multiple generator methods into one candidate payload",
        "submit_payload_generator": "submit generator methods for environment-side generation and grading",
    }
    return notes.get(tool_name, f"run {tool_name} with {arguments}")


async def run_agent_loop(
    client: OpenAI,
    env: MetricTrackerRlEnv,
    observation: MetricTrackerRlObservation,
) -> tuple[Any, str, int, list[dict[str, Any]], ToolRuntimeState]:
    """Run a tool-calling loop until the env is solved or the round limit is hit."""
    runtime_state = ToolRuntimeState()
    current_observation = observation
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_user_prompt(current_observation)},
    ]
    last_result = None
    final_text = ""
    tool_rounds = 0

    for _ in range(MAX_TOOL_ROUNDS):
        completion = create_chat_completion(
            client,
            model=MODEL_NAME,
            messages=messages,
            tools=tool_schemas(),
            tool_choice="auto",
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        message = completion.choices[0].message
        assistant_payload: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            assistant_payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]
        messages.append(assistant_payload)

        if not message.tool_calls:
            final_text = (message.content or "").strip()
            break

        tool_rounds += 1
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = decode_arguments(tool_call.function.arguments)
            if tool_name != "submit_solution":
                log_method(tool_name, arguments, _tool_note(tool_name, arguments))
            if tool_name in {"payload_generator", "submit_payload_generator"}:
                log_payload_generator_methods(
                    tool_name,
                    arguments.get("generator_methods", []),
                )
            tool_output, maybe_result, current_observation = await execute_tool_call(
                env,
                current_observation,
                runtime_state,
                tool_name,
                arguments,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_output),
                }
            )
            if maybe_result is not None:
                last_result = maybe_result

        if last_result is not None:
            completion = create_chat_completion(
                client,
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            final_text = (completion.choices[0].message.content or "").strip()
            break

    return last_result, final_text, tool_rounds, runtime_state.method_log, runtime_state


async def run_single_task(
    client: OpenAI,
    env: MetricTrackerRlEnv,
    task_id: str,
) -> dict[str, Any]:
    """Run one named benchmark task and return a reproducible summary."""
    task_spec = get_task_spec(task_id)
    log_start(task=task_spec.task_id, env=BENCHMARK, model=MODEL_NAME)
    reset_result = await env.reset(task_id=task_spec.task_id)
    final_result, final_text, tool_rounds, method_log, runtime_state = await run_agent_loop(
        client,
        env,
        reset_result.observation,
    )
    if final_result is None:
        raise RuntimeError(f"The model never submitted a graded action for task `{task_spec.task_id}`.")

    reward = float(final_result.reward or 0.0)
    task_score = bounded_task_score(reward)
    success = bool(final_result.done and reward >= 0.999999)
    return {
        "task_id": task_spec.task_id,
        "difficulty": task_spec.difficulty,
        "objective": task_spec.objective,
        "grader_name": task_spec.grader_name,
        "score": task_score,
        "normalized_score": task_score,
        "done": final_result.done,
        "success": success,
        "final_status": final_result.observation.status,
        "final_message": final_result.observation.message,
        "issue_count": len(final_result.observation.submission_issues),
        "correct_row_count": final_result.observation.correct_row_count,
        "expected_row_count": final_result.observation.expected_row_count,
        "tool_rounds": tool_rounds,
        "assistant_summary": final_text,
        "steps": runtime_state.steps,
        "rewards": runtime_state.rewards,
        "reward_breakdown": (
            final_result.observation.reward_breakdown.model_dump()
            if final_result.observation.reward_breakdown
            else None
        ),
    }


async def run_single_task_with_retries(
    client: OpenAI,
    task_id: str,
) -> dict[str, Any]:
    """Run one task with a fresh env connection and bounded reconnect retries."""
    attempts = TASK_RETRY_COUNT + 1
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        env = None
        success = False
        steps = 0
        score = 0.0
        rewards: list[float] = []
        try:
            env = await connect_env()
            summary = await run_single_task(client, env, task_id)
            success = bool(summary["success"])
            steps = int(summary["steps"])
            score = float(summary["score"])
            rewards = list(summary["rewards"])
            return summary
        except (ConnectionClosedError, ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
            print(
                (
                    f"[WARN] task_id={task_id} attempt={attempt}/{attempts} "
                    f"env_connection_error={type(exc).__name__}: {exc}"
                ),
                flush=True,
                file=sys.stderr,
            )
            if attempt >= attempts:
                raise
        finally:
            try:
                if env is not None:
                    await env.close()
            except Exception:
                pass
            if env is not None:
                log_end(success=success, steps=steps, score=score, rewards=rewards)

    assert last_error is not None
    raise last_error


async def main() -> None:
    if not API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY, HF_TOKEN, or API_KEY.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_summaries: list[dict[str, Any]] = []

    for task_id in DEFAULT_TASK_ORDER:
        task_summaries.append(await run_single_task_with_retries(client, task_id))


if __name__ == "__main__":
    asyncio.run(main())
