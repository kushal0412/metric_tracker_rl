"""Client for the metric tracker RL environment."""

import os
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from websockets.asyncio.client import connect as ws_connect

from .models import MetricTrackerRlAction, MetricTrackerRlObservation


class MetricTrackerRlEnv(
    EnvClient[MetricTrackerRlAction, MetricTrackerRlObservation, State]
):
    """Typed client for the metric tracking environment."""

    async def connect(self) -> "MetricTrackerRlEnv":
        """Connect with websocket keepalive disabled for long-running step calls."""
        if self._ws is not None:
            return self

        ws_url_lower = self._ws_url.lower()
        is_localhost = "localhost" in ws_url_lower or "127.0.0.1" in ws_url_lower
        old_no_proxy = os.environ.get("NO_PROXY")
        if is_localhost:
            current_no_proxy = old_no_proxy or ""
            if "localhost" not in current_no_proxy.lower():
                os.environ["NO_PROXY"] = (
                    f"{current_no_proxy},localhost,127.0.0.1"
                    if current_no_proxy
                    else "localhost,127.0.0.1"
                )

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=self._max_message_size,
                ping_interval=None,
                ping_timeout=None,
            )
        except Exception as exc:
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {exc}") from exc
        finally:
            if is_localhost:
                if old_no_proxy is None:
                    os.environ.pop("NO_PROXY", None)
                else:
                    os.environ["NO_PROXY"] = old_no_proxy

        return self

    def _step_payload(self, action: MetricTrackerRlAction) -> Dict:
        """Serialize the action as JSON for the environment server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[MetricTrackerRlObservation]:
        """Parse environment responses into a typed observation."""
        observation = MetricTrackerRlObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse environment state payloads."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
