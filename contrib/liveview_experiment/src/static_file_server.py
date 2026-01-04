"""Static file server implementation (reused from mvg_departures)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyview
from starlette.responses import FileResponse, Response
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, MutableMapping

    from pyview import PyView

logger = logging.getLogger(__name__)


class StaticFileCacheApp:
    """ASGI app wrapper that adds cache headers to static file responses."""

    def __init__(self, static_files: StaticFiles) -> None:
        """Initialize with a StaticFiles instance."""
        self.static_files = static_files

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        """Handle ASGI request and add cache headers."""
        # Intercept the response to add headers
        original_send = send

        async def send_with_cache_headers(
            message: MutableMapping[str, Any],
        ) -> None:
            """Add cache headers before sending response."""
            if message["type"] == "http.response.start":
                # Headers in ASGI are already a list of (bytes, bytes) tuples
                headers = list(message.get("headers", []))
                # Check if Cache-Control already exists
                has_cache_control = any(
                    header[0].lower() == b"cache-control" for header in headers
                )
                if not has_cache_control:
                    # Add Cache-Control header: cache for 1 minute, allow revalidation
                    headers.append(
                        (b"cache-control", b"public, max-age=60, must-revalidate")
                    )
                    message["headers"] = headers
            await original_send(message)

        await self.static_files(scope, receive, send_with_cache_headers)


class StaticFileServer:
    """Serves static files for the web application (reused from mvg_departures)."""

    def register_routes(self, app: PyView) -> None:
        """Register static file routes with the PyView app.

        Args:
            app: The PyView application instance.
        """
        # IMPORTANT: Register specific routes BEFORE mounting the static directory
        # This ensures specific routes take precedence over the mount
        # Use insert(0, ...) to put them at the beginning so they're checked first
        # Add route to serve pyview's client JavaScript
        # (needed for /static/assets/app.js)
        app.routes.insert(0, Route("/static/assets/app.js", self._serve_app_js))

        # Mount static files directory to serve CSS, JS, and other assets
        # Try to find PyView's static directory
        pyview_path = Path(pyview.__file__).parent
        static_path = pyview_path / "static"

        if static_path.exists():
            # Create StaticFiles instance and wrap with cache app
            static_files = StaticFiles(directory=str(static_path))
            # Wrap with cache app to add cache headers
            cached_static = StaticFileCacheApp(static_files)
            app.mount("/static", cached_static, name="static")
            logger.info(
                "Mounted static files from %s with 1-minute cache headers", static_path
            )
        else:
            logger.warning("PyView static directory not found at: %s", static_path)

    async def _serve_app_js(self, _request: Any) -> Any:
        """Serve pyview's client JavaScript."""
        try:
            # Get pyview package path
            pyview_path = Path(pyview.__file__).parent
            client_js_path = pyview_path / "static" / "assets" / "app.js"

            if client_js_path.exists():
                response = FileResponse(
                    str(client_js_path), media_type="application/javascript"
                )
                # Add cache headers: cache for 1 minute
                response.headers["Cache-Control"] = (
                    "public, max-age=60, must-revalidate"
                )
                return response
            # Fallback: try alternative path
            alt_path = pyview_path / "assets" / "js" / "app.js"
            if alt_path.exists():
                response = FileResponse(
                    str(alt_path), media_type="application/javascript"
                )
                response.headers["Cache-Control"] = (
                    "public, max-age=60, must-revalidate"
                )
                return response
            logger.error(
                "Could not find pyview client JS at %s or %s",
                client_js_path,
                alt_path,
            )
            return Response(
                content="// PyView client not found",
                media_type="application/javascript",
                status_code=404,
            )
        except Exception as e:
            logger.exception("Error serving pyview client JS: %s", e)
            return Response(
                content="// Error loading client",
                media_type="application/javascript",
                status_code=500,
            )
