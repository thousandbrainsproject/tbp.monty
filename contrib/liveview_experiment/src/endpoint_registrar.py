"""Registers HTTP endpoints for the LiveView server."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from .state_manager import ExperimentStateManager

logger = logging.getLogger(__name__)

from contrib.liveview_experiment.src.pubsub_endpoint_handler import (
    PubSubEndpointHandler,
)

try:
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except ImportError:
    Request = None
    JSONResponse = None
    Route = None


class EndpointRegistrar:
    """Registers HTTP endpoints for the LiveView server."""

    @staticmethod
    def register_pubsub_endpoint(
        app: Any, state_manager: ExperimentStateManager  # PyView
    ) -> None:
        """Register HTTP pub/sub endpoint.

        Args:
            app: PyView application
            state_manager: Experiment state manager
        """
        if Request is None or JSONResponse is None or Route is None:
            logger.warning("Starlette components not available, skipping endpoint")
            return

        try:

            async def pubsub_endpoint(request: Request) -> JSONResponse:
                """Handle HTTP pub/sub requests."""
                return await PubSubEndpointHandler.handle_request(
                    request, state_manager
                )

            app.routes.append(Route("/api/pubsub", pubsub_endpoint, methods=["POST"]))
            logger.info("HTTP pub/sub endpoint registered at /api/pubsub")
        except (ImportError, AttributeError) as e:
            logger.warning("Could not register HTTP pub/sub endpoint: %s", e)
