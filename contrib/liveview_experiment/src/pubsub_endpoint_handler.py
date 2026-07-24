"""HTTP endpoint handler for pub/sub requests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from starlette.requests import Request

    from .state_manager import ExperimentStateManager

logger = logging.getLogger(__name__)


class PubSubEndpointHandler:
    """Handles HTTP pub/sub requests for experiment updates."""

    @staticmethod
    async def handle_request(
        request: Request, state_manager: ExperimentStateManager
    ) -> JSONResponse:
        """Handle HTTP pub/sub request.

        Args:
            request: HTTP request with JSON body containing topic and payload
            state_manager: Experiment state manager to update

        Returns:
            JSONResponse with status
        """
        try:
            body = await request.json()
            topic = body.get("topic")
            payload = body.get("payload")

            if not topic or not payload:
                return JSONResponse(
                    {"status": "error", "message": "Missing topic or payload"},
                    status_code=400,
                )

            PubSubEndpointHandler._route_to_handler(state_manager, topic, payload)
            return JSONResponse({"status": "ok"})
        except (ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.exception("Error handling pub/sub request: %s", e)
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @staticmethod
    def _route_to_handler(
        state_manager: ExperimentStateManager, topic: str, payload: dict
    ) -> None:
        """Route topic to appropriate handler.

        Args:
            state_manager: Experiment state manager
            topic: Message topic
            payload: Message payload
        """
        handlers = {
            state_manager.metrics_topic: state_manager._handle_metric_message,
            state_manager.data_topic: state_manager._handle_data_message,
            state_manager.logs_topic: state_manager._handle_log_message,
        }

        handler = handlers.get(topic)
        if handler:
            handler(topic, payload)
