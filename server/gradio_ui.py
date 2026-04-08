"""Custom Gradio UI for testing the metric tracker RL environment."""

from __future__ import annotations

import json
import math
from statistics import median

import pandas as pd

try:
    from ..analysis_tools import available_analysis_methods
    from ..tasks import DEFAULT_TASK_ORDER, available_task_specs, get_task_spec
except ImportError:
    from analysis_tools import available_analysis_methods
    from tasks import DEFAULT_TASK_ORDER, available_task_specs, get_task_spec

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None


GENERATOR_METHODS = [
    "get_median_filter_rows",
    "get_rate_drop_from_median_rows",
    "get_rate_spike_from_median_rows",
    "get_absolute_drop_in_event_count_rows",
    "get_absolute_spike_in_event_count_rows",
    "get_funnel_break_rows",
    "get_hourly_traffic_mix_shift_rows",
    "get_instrumentation_data_quality_issue_rows",
]
METHOD_CHOICES = [item.name for item in available_analysis_methods()]
TASK_CHOICES = list(DEFAULT_TASK_ORDER)
TASK_SUMMARIES = {
    item.task_id: item.model_dump()
    for item in available_task_specs()
}
METRIC_CHOICES = [
    "app_opens",
    "menu_opens",
    "product_added_to_cart",
    "orders_placed",
    "payment_successful",
    "app_open_to_menu_open",
    "menu_open_to_product_added_to_cart",
    "product_added_to_cart_to_order_placed",
    "order_placed_to_payment_successful",
    "app_open_to_order_placed",
    "app_open_to_payment_successful",
]
CONVERSION_METRICS = [
    "app_open_to_menu_open",
    "menu_open_to_product_added_to_cart",
    "product_added_to_cart_to_order_placed",
    "order_placed_to_payment_successful",
    "app_open_to_order_placed",
    "app_open_to_payment_successful",
]
THRESHOLD_METHODS = {
    "get_rows_with_abs_diff_from_median_gt",
    "get_median_filter_rows",
    "get_rate_drop_from_median_rows",
    "get_rate_spike_from_median_rows",
    "get_absolute_drop_in_event_count_rows",
    "get_absolute_spike_in_event_count_rows",
}
MEDIAN_METHODS = THRESHOLD_METHODS | {
    "get_metric_median",
    "get_metric_std_dev_from_median",
}


def build_metric_tracker_gradio_app(
    web_manager,
    action_fields,
    metadata,
    is_chat_env,
    title,
    quick_start_md,
):
    """Build a method-driven and generator-driven debugger."""
    del action_fields, metadata, is_chat_env, quick_start_md
    if gr is None:  # pragma: no cover
        raise ImportError("gradio is required to build the custom metric tracker UI.")

    with gr.Blocks() as demo:
        gr.Markdown(
            f"""
            # {title} Generator Debugger

            The UI now supports the same named benchmark tasks used by the agent baseline.
            Pick a task to load its canonical easy, medium, or hard setup, then optionally
            override the reset controls for custom debugging.

            Standard mode exposes method calls only. You inspect data through methods like
            `show_raw_data`, `get_metric_median`, `get_metric_std_dev_from_median`,
            and `get_rows_with_abs_diff_from_median_gt`, then assemble payload generators
            such as `get_rate_spike_from_median_rows(metric_name, threshold_multiplier)`.
            """
        )

        session_state = gr.State(_empty_state())

        with gr.Row():
            task_id = gr.Dropdown(
                label="Named Task",
                choices=TASK_CHOICES,
                value=TASK_CHOICES[0],
            )
            task_details = gr.JSON(
                label="Selected Task Details",
                value=TASK_SUMMARIES[TASK_CHOICES[0]],
            )

        with gr.Row():
            initial_task = get_task_spec(TASK_CHOICES[0])
            seed = gr.Number(label="Seed", value=initial_task.seed, precision=0)
            scenario_family = gr.Dropdown(
                label="Scenario Family",
                choices=[
                    "mixed",
                    "rate_drop_from_median",
                    "rate_spike_from_median",
                    "absolute_drop_in_event_count",
                    "absolute_spike_in_event_count",
                    "funnel_break",
                    "hourly_traffic_mix_shift",
                    "instrumentation_data_quality_issue",
                ],
                value=initial_task.scenario_family,
            )
            difficulty = gr.Dropdown(
                label="Difficulty",
                choices=["easy", "medium", "hard"],
                value=initial_task.difficulty,
            )
            anomaly_density = gr.Dropdown(
                label="Anomaly Density",
                choices=["low", "medium", "high"],
                value=initial_task.anomaly_density,
            )
            anomaly_count = gr.Number(label="Anomaly Count", value=initial_task.anomaly_count, precision=0)
            debug_mode = gr.Checkbox(label="Debug Mode", value=False)

        reset_anomalies = gr.Code(
            label="Reset Anomalies JSON",
            language="json",
            value="[]",
            interactive=True,
        )

        with gr.Row():
            reset_btn = gr.Button("Reset Episode", variant="primary")
            preview_btn = gr.Button("Preview Generator Payload", variant="secondary")
            submit_btn = gr.Button("Submit Generator Payload", variant="secondary")
            get_state_btn = gr.Button("Get State", variant="secondary")

        gr.Markdown("## Methods")
        gr.Markdown(
            "Run a method after reset to inspect daily aggregate data and then start from "
            "rate-spike detection on conversion metrics before broadening the search."
        )
        with gr.Row():
            method_name = gr.Dropdown(
                label="Method",
                choices=METHOD_CHOICES,
                value="get_rate_spike_from_median_rows",
            )
            method_metric = gr.Dropdown(
                label="metrics",
                choices=METRIC_CHOICES,
                value=CONVERSION_METRICS,
                multiselect=True,
            )
            method_threshold = gr.Number(label="threshold / multiplier", value=2.0)
            method_limit = gr.Number(label="limit", value=5, precision=0)
            run_method_btn = gr.Button("Run Method", variant="secondary")
        with gr.Row():
            method_date = gr.Textbox(label="date", placeholder="YYYY-MM-DD")
            method_entity = gr.Textbox(label="entity_name", placeholder="orders_placed or app_open_to_order_placed")
        method_rows_json = gr.Code(
            label="rows JSON for preview_submission",
            language="json",
            value="[]",
            interactive=True,
        )
        plot_metrics = gr.Dropdown(
            label="Plot Metrics",
            choices=METRIC_CHOICES,
            value=CONVERSION_METRICS,
            multiselect=True,
        )
        metric_plot = gr.LinePlot(
            label="Metric Plot",
            x="date",
            y="value",
            color="series",
            tooltip=["date", "series", "value"],
        )
        generated_rows = gr.Dataframe(label="Payload Rows For Current Method", interactive=False)
        analysis_result = gr.JSON(label="Last Method Results")

        with gr.Tab("Method Data"):
            gr.Markdown(
                "This panel shows only method-returned data. The chart already loads all daily "
                "rows on reset, so use this table to inspect the exact rows returned by the current method."
            )
            method_rows = gr.Dataframe(label="Method Rows", interactive=False)

        gr.Markdown("## Payload Generators")
        gr.Markdown(
            "Add generator methods here, then preview or submit using the buttons at the top."
        )
        generator_methods_df = gr.Dataframe(
            headers=["method_name", "metric_name", "metric_names", "threshold_multiplier"],
            datatype=["str", "str", "str", "number"],
            label="Generator Methods",
            interactive=True,
        )
        payload_generator_methods = gr.JSON(label="Methods Passed to Payload Generator")
        with gr.Row():
            generator_method_name = gr.Dropdown(label="method_name", choices=GENERATOR_METHODS, value="get_rate_spike_from_median_rows")
            generator_metric_name = gr.Dropdown(
                label="metrics",
                choices=METRIC_CHOICES,
                value=CONVERSION_METRICS,
                multiselect=True,
            )
            generator_multiplier = gr.Number(label="threshold_multiplier", value=2.0)
        with gr.Row():
            add_generator_btn = gr.Button("Add / Update Generator", variant="secondary")
            remove_generator_btn = gr.Button("Remove Generator", variant="secondary")
            clear_generators_btn = gr.Button("Clear Generators", variant="secondary")

        status = gr.Textbox(label="Status", interactive=False)
        summary = gr.JSON(label="Episode Summary")
        active_task = gr.JSON(label="Active Task", value=TASK_SUMMARIES[TASK_CHOICES[0]])
        task_catalog = gr.JSON(label="Available Tasks", value=list(TASK_SUMMARIES.values()))
        synthetic_methods = gr.JSON(label="Synthetic Generator Methods")
        applied_synthetic_generators = gr.Dataframe(label="Applied Synthetic Generators", interactive=False)
        available_methods = gr.JSON(label="Shared Methods")
        submission_feedback = gr.JSON(label="Submission Feedback")
        reward_breakdown = gr.JSON(label="Reward Breakdown")
        raw_json = gr.Code(label="Latest Environment Response", language="json", interactive=False)
        debug_snapshot = gr.JSON(label="Debug Snapshot")

        def apply_task_defaults(selected_task_id: str):
            task = get_task_spec(selected_task_id)
            return (
                task.seed,
                task.scenario_family,
                task.difficulty,
                task.anomaly_density,
                task.anomaly_count,
                task.to_model().model_dump(),
            )

        async def reset_episode(
            selected_task_id,
            seed_value,
            family,
            level,
            density,
            anomaly_count_value,
            reset_anomalies_json,
            debug_enabled,
            selected_plot_metrics,
        ):
            try:
                parsed_anomalies = json.loads(reset_anomalies_json or "[]")
                if not isinstance(parsed_anomalies, list):
                    raise ValueError("Reset anomalies JSON must be a list.")
            except Exception as exc:
                return (
                    _empty_state(),
                    f"Invalid reset anomalies JSON: {exc}",
                    {"status": "error"},
                    {},
                    list(TASK_SUMMARIES.values()),
                    [],
                    _generator_frame([]),
                    [],
                    {},
                    {},
                    {},
                    _plot_frame([], selected_plot_metrics, None),
                    _generator_frame([]),
                    _generator_frame([]),
                    "",
                    _debug_snapshot(web_manager, debug_enabled),
                    _generator_frame([]),
                    [],
                )
            web_manager.env.set_debug_mode(bool(debug_enabled))
            data = await web_manager.reset_environment(
                {
                    "task_id": selected_task_id,
                    "seed": int(seed_value or 0),
                    "scenario_family": family,
                    "difficulty": level,
                    "anomaly_density": density,
                    "anomaly_count": int(anomaly_count_value or 3),
                    "anomalies": parsed_anomalies,
                }
            )
            method_data = await web_manager.step_environment(
                {
                    "analysis_method": "show_raw_data",
                    "analysis_args": {"limit": 10000},
                    "classifications": [],
                    "payload_generators": [],
                }
            )
            state = _state_from_response(data)
            state["latest_response"] = method_data
            state["last_method_result"] = method_data.get("observation", {}).get("analysis_result")
            state["raw_rows"] = ((state["last_method_result"] or {}).get("result") or {}).get("rows", [])
            state["last_plot_context"] = None
            obs = data.get("observation", {})
            method_result = state["last_method_result"] or {}
            plot_frame = _plot_frame(state["raw_rows"], selected_plot_metrics, state["last_plot_context"])
            available_tasks = obs.get("available_tasks") or list(TASK_SUMMARIES.values())
            active_task_payload = next(
                (item for item in available_tasks if item.get("task_id") == obs.get("task_id")),
                {
                    "task_id": obs.get("task_id"),
                    "instruction": obs.get("instruction"),
                    "objective": obs.get("message"),
                    "difficulty": (obs.get("config") or {}).get("difficulty"),
                    "grader_name": (obs.get("config") or {}).get("grader_name"),
                },
            )
            return (
                state,
                obs.get("message", ""),
                {
                    "task_id": obs.get("task_id"),
                    "status": obs.get("status"),
                    "config": obs.get("config"),
                    "expected_row_count": obs.get("expected_row_count"),
                },
                active_task_payload,
                available_tasks,
                [item for item in obs.get("available_synthetic_generator_methods", [])],
                pd.DataFrame([item for item in obs.get("applied_synthetic_generators", [])]),
                [item for item in obs.get("available_methods", [])],
                method_result,
                obs.get("submission_issues") or [],
                obs.get("reward_breakdown") or {},
                plot_frame,
                _method_frame(method_result),
                pd.DataFrame(),
                json.dumps(method_data, indent=2),
                _debug_snapshot(web_manager, debug_enabled),
                _generator_frame(state["payload_generators"]),
                state["payload_generators"],
            )

        async def run_method(
            payload: dict,
            selected_method: str,
            metric_names: list[str],
            method_date_value: str,
            method_entity_value: str,
            method_rows_value: str,
            threshold: float,
            limit_value: int,
            selected_plot_metrics: list[str],
        ):
            if not payload.get("active"):
                return payload, {"error": "Reset the environment first."}, "", gr.skip(), gr.skip(), gr.skip(), gr.skip()
            args = _method_args(
                selected_method,
                metric_names,
                method_date_value,
                method_entity_value,
                method_rows_value,
                threshold,
                limit_value,
                payload["payload_generators"],
            )
            data = await web_manager.step_environment(
                {
                    "analysis_method": selected_method,
                    "analysis_args": args,
                    "classifications": [],
                    "payload_generators": [],
                }
            )
            payload["latest_response"] = data
            payload["last_method_result"] = data.get("observation", {}).get("analysis_result")
            payload["last_plot_context"] = {
                "method_name": selected_method,
                "metric_names": [item for item in (metric_names or []) if item],
                "threshold": float(threshold),
            }
            method_result = payload["last_method_result"] or {}
            generated = method_result.get("result", {}).get("generated_rows", [])
            method_frame = _method_frame(method_result)
            plot_frame = _plot_frame(payload.get("raw_rows", []), selected_plot_metrics, payload["last_plot_context"])
            return (
                payload,
                method_result,
                data.get("observation", {}).get("message", ""),
                plot_frame,
                method_frame,
                pd.DataFrame(generated),
                json.dumps(data, indent=2),
            )

        def add_or_update_generator(payload: dict, method_name_value: str, metric_names: list[str], threshold_multiplier: float):
            if not payload.get("active"):
                return payload, _generator_frame([]), []
            metric_names = [item for item in (metric_names or []) if item]
            row = {
                "method_name": method_name_value,
                "metric_name": metric_names[0] if len(metric_names) == 1 else None,
                "metric_names": metric_names,
                "threshold_multiplier": float(threshold_multiplier),
            }
            keyed = {
                _generator_row_key(item): item
                for item in payload["payload_generators"]
            }
            keyed[_generator_row_key(row)] = row
            payload["payload_generators"] = list(keyed.values())
            return payload, _generator_frame(payload["payload_generators"]), payload["payload_generators"]

        def remove_generator(payload: dict, method_name_value: str, metric_names: list[str]):
            if not payload.get("active"):
                return payload, _generator_frame([]), []
            metric_names = [item for item in (metric_names or []) if item]
            payload["payload_generators"] = [
                item
                for item in payload["payload_generators"]
                if not (
                    item.get("method_name") == method_name_value
                    and [name for name in item.get("metric_names", []) if name] == metric_names
                )
            ]
            return payload, _generator_frame(payload["payload_generators"]), payload["payload_generators"]

        def clear_generators(payload: dict):
            payload["payload_generators"] = []
            return payload, _generator_frame([]), []

        def sync_generator_rows(payload: dict, generator_rows):
            normalized = _normalize_generator_rows(generator_rows)
            payload["payload_generators"] = normalized
            return payload, _generator_frame(normalized), normalized

        async def preview_payload(payload: dict, generator_rows):
            if not payload.get("active"):
                return payload, {"error": "Reset the environment first."}, _generator_frame([]), []
            payload["payload_generators"] = _normalize_generator_rows(generator_rows)
            if not payload.get("payload_generators"):
                return payload, {"error": "Add at least one payload generator first."}, _generator_frame([]), []
            data = await web_manager.step_environment(
                {
                    "analysis_method": "payload_generator",
                    "analysis_args": {"generator_methods": payload["payload_generators"]},
                    "classifications": [],
                    "payload_generators": [],
                }
            )
            payload["latest_response"] = data
            payload["last_method_result"] = data.get("observation", {}).get("analysis_result")
            result = payload["last_method_result"] or {}
            return payload, result, pd.DataFrame(result.get("result", {}).get("generated_rows", [])), payload["payload_generators"]

        async def submit_payload(payload: dict, debug_enabled: bool, generator_rows):
            if not payload.get("active"):
                return payload, "Reset the environment first.", gr.skip(), gr.skip(), gr.skip(), "", gr.skip(), gr.skip(), gr.skip()
            payload["payload_generators"] = _normalize_generator_rows(generator_rows)
            if not payload.get("payload_generators"):
                return (
                    payload,
                    "Add at least one payload generator before submitting.",
                    {
                        "status": "ready",
                        "generator_count": 0,
                    },
                    {"error": "No payload generators configured."},
                    {},
                    "",
                    _debug_snapshot(web_manager, debug_enabled),
                    _generator_frame([]),
                    [],
                )
            data = await web_manager.step_environment(
                {
                    "payload_generators": payload["payload_generators"],
                    "classifications": [],
                }
            )
            payload["latest_response"] = data
            obs = data.get("observation", {})
            summary = {
                "task_id": obs.get("task_id"),
                "status": obs.get("status"),
                "message": obs.get("message"),
                "config": obs.get("config"),
                "expected_row_count": obs.get("expected_row_count"),
                "correct_row_count": obs.get("correct_row_count"),
                "generated_row_count": len(obs.get("generated_rows") or []),
                "submitted_row_count": len(obs.get("submitted_rows") or []),
                "issue_count": len(obs.get("submission_issues") or []),
                "reward": data.get("reward", 0.0),
                "done": data.get("done", False),
            }
            feedback = {
                "message": obs.get("message", ""),
                "issue_count": len(obs.get("submission_issues") or []),
                "issues": obs.get("submission_issues") or [],
                "generated_row_count": len(obs.get("generated_rows") or []),
                "generator_count": len(payload.get("payload_generators") or []),
            }
            return (
                payload,
                obs.get("message", ""),
                summary,
                feedback,
                obs.get("reward_breakdown") or {},
                json.dumps(data, indent=2),
                _debug_snapshot(web_manager, debug_enabled),
                pd.DataFrame([row for row in obs.get("generated_rows", [])]),
                payload["payload_generators"],
            )

        def get_state_sync():
            return json.dumps(web_manager.get_state(), indent=2)

        def update_plot(payload: dict, selected_plot_metrics: list[str]):
            return _plot_frame(
                payload.get("raw_rows", []),
                selected_plot_metrics,
                payload.get("last_plot_context"),
            )

        reset_btn.click(
            fn=reset_episode,
            inputs=[task_id, seed, scenario_family, difficulty, anomaly_density, anomaly_count, reset_anomalies, debug_mode, plot_metrics],
            outputs=[
                session_state,
                status,
                summary,
                active_task,
                task_catalog,
                synthetic_methods,
                applied_synthetic_generators,
                available_methods,
                analysis_result,
                submission_feedback,
                reward_breakdown,
                metric_plot,
                method_rows,
                generated_rows,
                raw_json,
                debug_snapshot,
                generator_methods_df,
                payload_generator_methods,
            ],
        )
        task_id.change(
            fn=apply_task_defaults,
            inputs=[task_id],
            outputs=[seed, scenario_family, difficulty, anomaly_density, anomaly_count, task_details],
        )
        run_method_btn.click(
            fn=run_method,
            inputs=[
                session_state,
                method_name,
                method_metric,
                method_date,
                method_entity,
                method_rows_json,
                method_threshold,
                method_limit,
                plot_metrics,
            ],
            outputs=[session_state, analysis_result, status, metric_plot, method_rows, generated_rows, raw_json],
        )
        plot_metrics.change(
            fn=update_plot,
            inputs=[session_state, plot_metrics],
            outputs=[metric_plot],
        )
        add_generator_btn.click(
            fn=add_or_update_generator,
            inputs=[session_state, generator_method_name, generator_metric_name, generator_multiplier],
            outputs=[session_state, generator_methods_df, payload_generator_methods],
        )
        remove_generator_btn.click(
            fn=remove_generator,
            inputs=[session_state, generator_method_name, generator_metric_name],
            outputs=[session_state, generator_methods_df, payload_generator_methods],
        )
        clear_generators_btn.click(
            fn=clear_generators,
            inputs=[session_state],
            outputs=[session_state, generator_methods_df, payload_generator_methods],
        )
        generator_methods_df.change(
            fn=sync_generator_rows,
            inputs=[session_state, generator_methods_df],
            outputs=[session_state, generator_methods_df, payload_generator_methods],
        )
        preview_btn.click(
            fn=preview_payload,
            inputs=[session_state, generator_methods_df],
            outputs=[session_state, analysis_result, generated_rows, payload_generator_methods],
        )
        submit_btn.click(
            fn=submit_payload,
            inputs=[session_state, debug_mode, generator_methods_df],
            outputs=[session_state, status, summary, submission_feedback, reward_breakdown, raw_json, debug_snapshot, generated_rows, payload_generator_methods],
        )
        get_state_btn.click(fn=get_state_sync, outputs=[raw_json])

    return demo


def _method_args(
    method_name: str,
    metric_names: list[str],
    method_date: str,
    method_entity: str,
    method_rows_json: str,
    threshold: float,
    limit_value: int,
    payload_generators: list[dict],
) -> dict:
    selected = [item for item in (metric_names or []) if item]
    resolved_date = (method_date or "").strip()
    resolved_entity = (method_entity or "").strip()
    if method_name == "show_raw_data":
        return {"limit": int(limit_value or 5)}
    if method_name in {"rows_for_date", "hourly_rows_for_date", "detect_funnel_break", "check_impossible_counts"}:
        return {"date": resolved_date}
    if method_name in {"compare_rate_to_median", "compare_count_to_median"}:
        return {
            "date": resolved_date,
            "entity_name": resolved_entity,
        }
    if method_name in {"get_metric_median", "get_metric_std_dev_from_median"}:
        return {
            "metric_name": selected[0] if len(selected) == 1 else None,
            "metric_names": selected,
        }
    if method_name == "get_rows_with_abs_diff_from_median_gt":
        return {
            "metric_name": selected[0] if len(selected) == 1 else None,
            "metric_names": selected,
            "threshold": float(threshold),
        }
    if method_name in {
        "get_median_filter_rows",
        "get_rate_drop_from_median_rows",
        "get_rate_spike_from_median_rows",
        "get_absolute_drop_in_event_count_rows",
        "get_absolute_spike_in_event_count_rows",
    }:
        return {
            "metric_name": selected[0] if len(selected) == 1 else None,
            "metric_names": selected,
            "threshold_multiplier": float(threshold),
        }
    if method_name in {
        "get_funnel_break_rows",
        "get_hourly_traffic_mix_shift_rows",
        "get_instrumentation_data_quality_issue_rows",
    }:
        return {"threshold_multiplier": float(threshold)}
    if method_name == "payload_generator":
        return {"generator_methods": payload_generators}
    if method_name == "list_suspicious_dates":
        return {"limit": int(limit_value or 10)}
    if method_name == "preview_submission":
        return {"rows": _parse_rows_json(method_rows_json)}
    return {}


def _parse_rows_json(raw_value: str) -> list[dict]:
    if not raw_value or not raw_value.strip():
        return []
    parsed = json.loads(raw_value)
    if not isinstance(parsed, list):
        raise ValueError("rows JSON must be a list.")
    return [item for item in parsed if isinstance(item, dict)]


def _method_frame(method_result: dict) -> pd.DataFrame:
    result = (method_result or {}).get("result") or {}
    if isinstance(result, dict):
        if isinstance(result.get("results"), list):
            rows = []
            for item in result["results"]:
                if isinstance(item, dict) and isinstance(item.get("rows"), list):
                    for row in item["rows"]:
                        enriched = dict(row)
                        enriched["metric_name"] = item.get("metric_name", enriched.get("metric_name"))
                        rows.append(enriched)
                elif isinstance(item, dict):
                    rows.append(item)
            return pd.DataFrame(rows)
        if isinstance(result.get("rows"), list):
            return pd.DataFrame(result["rows"])
        if isinstance(result.get("dates"), list):
            return pd.DataFrame(result["dates"])
        if isinstance(result.get("generated_rows"), list):
            return pd.DataFrame(result["generated_rows"])
    return pd.DataFrame()


def _state_from_response(data: dict) -> dict:
    return {
        "active": True,
        "payload_generators": [],
        "last_method_result": data.get("observation", {}).get("analysis_result"),
        "latest_response": data,
        "raw_rows": [],
        "last_plot_context": None,
    }


def _normalize_generator_rows(generator_rows) -> list[dict]:
    if generator_rows is None:
        return []
    if isinstance(generator_rows, pd.DataFrame):
        rows = generator_rows.to_dict(orient="records")
    elif isinstance(generator_rows, list):
        rows = generator_rows
    else:
        return []

    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        metric_names = row.get("metric_names", [])
        if isinstance(metric_names, str):
            metric_names = [item for item in metric_names.split(",") if item]
        elif not isinstance(metric_names, list):
            metric_names = []
        normalized.append(
            {
                "method_name": row.get("method_name"),
                "metric_name": row.get("metric_name"),
                "metric_names": metric_names,
                "threshold_multiplier": float(row.get("threshold_multiplier", 0.0)),
            }
        )
    return normalized


def _generator_row_key(row: dict) -> str:
    metric_names = [item for item in (row.get("metric_names") or []) if item]
    return (
        f"{row.get('method_name') or ''}"
        f"|{','.join(metric_names)}"
        f"|{row.get('metric_name') or ''}"
        f"|{float(row.get('threshold_multiplier', 0.0)):.6f}"
    )


def _generator_frame(rows: list[dict]) -> pd.DataFrame:
    normalized = []
    for row in rows or []:
        metric_names = [item for item in (row.get("metric_names") or []) if item]
        normalized.append(
            {
                "method_name": row.get("method_name") or "",
                "metric_name": row.get("metric_name") or "",
                "metric_names": ",".join(metric_names),
                "threshold_multiplier": float(row.get("threshold_multiplier", 0.0)),
            }
        )
    return pd.DataFrame(
        normalized,
        columns=["method_name", "metric_name", "metric_names", "threshold_multiplier"],
    )


def _debug_snapshot(web_manager, debug_enabled: bool) -> dict:
    if not debug_enabled:
        return {}
    try:
        return web_manager.env.export_debug_snapshot()
    except Exception as exc:
        return {"error": str(exc)}


def _empty_state() -> dict:
    return {
        "active": False,
        "payload_generators": [],
        "last_method_result": None,
        "latest_response": None,
        "raw_rows": [],
        "last_plot_context": None,
    }


def _plot_frame(raw_rows: list[dict], selected_metrics: list[str], plot_context: dict | None) -> pd.DataFrame:
    if not raw_rows:
        return pd.DataFrame(columns=["date", "value", "series"])
    frame = pd.DataFrame(raw_rows)
    if "date" not in frame.columns:
        return pd.DataFrame(columns=["date", "value", "series"])
    metrics = [item for item in (selected_metrics or []) if item in frame.columns]
    if not metrics:
        return pd.DataFrame(columns=["date", "value", "series"])

    rows: list[dict] = []
    for metric_name in metrics:
        values = pd.to_numeric(frame[metric_name], errors="coerce")
        for date_value, numeric_value in zip(frame["date"], values):
            if pd.isna(numeric_value):
                continue
            rows.append(
                {
                    "date": date_value,
                    "value": float(numeric_value),
                    "series": metric_name,
                }
            )
        rows.extend(_overlay_rows(frame, metric_name, plot_context))
    return pd.DataFrame(rows, columns=["date", "value", "series"])


def _overlay_rows(frame: pd.DataFrame, metric_name: str, plot_context: dict | None) -> list[dict]:
    if not plot_context:
        return []
    selected_metrics = [item for item in (plot_context.get("metric_names") or []) if item]
    method_name = plot_context.get("method_name")
    threshold = float(plot_context.get("threshold", 0.0))
    if metric_name not in selected_metrics or method_name not in MEDIAN_METHODS:
        return []

    values = [float(item) for item in pd.to_numeric(frame[metric_name], errors="coerce").dropna().tolist()]
    if not values:
        return []
    dates = frame["date"].tolist()
    metric_median = float(median(values))
    rows = [
        {"date": date_value, "value": metric_median, "series": f"{metric_name} median"}
        for date_value in dates
    ]
    threshold_value = None
    if method_name == "get_metric_std_dev_from_median":
        threshold_value = _std_from_median(values)
    elif method_name == "get_rows_with_abs_diff_from_median_gt":
        threshold_value = threshold
    elif method_name in THRESHOLD_METHODS:
        threshold_value = _std_from_median(values) * threshold

    if threshold_value is None:
        return rows
    upper = metric_median + threshold_value
    lower = metric_median - threshold_value
    suffix = (
        f"{threshold:.2f}*std"
        if method_name in THRESHOLD_METHODS and method_name != "get_rows_with_abs_diff_from_median_gt"
        else "threshold"
    )
    rows.extend(
        {"date": date_value, "value": upper, "series": f"{metric_name} + {suffix}"}
        for date_value in dates
    )
    rows.extend(
        {"date": date_value, "value": lower, "series": f"{metric_name} - {suffix}"}
        for date_value in dates
    )
    return rows


def _std_from_median(values: list[float]) -> float:
    if not values:
        return 0.0
    metric_median = median(values)
    return math.sqrt(sum((value - metric_median) ** 2 for value in values) / len(values))
