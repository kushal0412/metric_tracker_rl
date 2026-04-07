"""Client for the metric tracker RL environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MetricTrackerRlAction, MetricTrackerRlObservation


class MetricTrackerRlEnv(
    EnvClient[MetricTrackerRlAction, MetricTrackerRlObservation, State]
):
    """Typed client for the metric tracking environment."""

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
