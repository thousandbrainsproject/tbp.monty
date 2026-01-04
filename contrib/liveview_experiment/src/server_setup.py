"""Server setup and configuration for LiveView."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

try:
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route
except ImportError:
    Request = None
    JSONResponse = None
    Route = None

from contrib.liveview_experiment.src.liveview_experiment import (
    ExperimentLiveView,
)
from contrib.liveview_experiment.src.pubsub_endpoint_handler import (
    PubSubEndpointHandler,
)
from contrib.liveview_experiment.src.static_file_server import (
    StaticFileServer,
)
from pyview import PyView

if TYPE_CHECKING:
    from .state_manager import ExperimentStateManager

logger = logging.getLogger(__name__)


class LiveViewServerSetup:
    """Handles setup and configuration of the LiveView server."""

    @staticmethod
    def create_app(
        state_manager: ExperimentStateManager,
    ) -> Any:  # PyView
        """Create and configure PyView application.

        Args:
            state_manager: Experiment state manager

        Returns:
            Configured PyView application
        """
        app = PyView()

        # Configure static file serving
        static_file_server = StaticFileServer()
        static_file_server.register_routes(app)

        # Create LiveView instance factory
        live_view_instance = None

        def create_live_view() -> ExperimentLiveView:
            nonlocal live_view_instance
            live_view_instance = ExperimentLiveView(state_manager)
            state_manager.liveview_instance = live_view_instance
            return live_view_instance

        app.add_live_view("/", create_live_view)
        app.state.state_manager = state_manager

        # Register pub/sub endpoint
        LiveViewServerSetup._register_pubsub_endpoint(app, state_manager)

        return app

    @staticmethod
    def _register_pubsub_endpoint(
        app: Any, state_manager: ExperimentStateManager  # PyView
    ) -> None:
        """Register HTTP pub/sub endpoint.

        Args:
            app: PyView application
            state_manager: Experiment state manager
        """
        try:
            if Request is None or JSONResponse is None or Route is None:
                raise ImportError("Starlette imports not available")

            async def pubsub_endpoint(request: Request) -> JSONResponse:
                """Handle HTTP pub/sub requests."""
                return await PubSubEndpointHandler.handle_request(
                    request, state_manager
                )

            app.routes.append(Route("/api/pubsub", pubsub_endpoint, methods=["POST"]))
            logger.info("HTTP pub/sub endpoint registered at /api/pubsub")
        except (ImportError, AttributeError) as e:
            logger.warning("Could not register HTTP pub/sub endpoint: %s", e)
