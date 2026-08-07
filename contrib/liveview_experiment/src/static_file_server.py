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

    _local_static_path: Path | None = None

    def register_routes(self, app: PyView) -> None:
        """Register static file routes with the PyView app.

        Args:
            app: The PyView application instance.
        """
        # Mount our local static files directory FIRST (higher priority)
        # This allows us to override pyview's static files and add custom assets
        local_static_path = Path(__file__).parent.parent / "static"
        if local_static_path.exists():
            # Mount at /static with higher priority by inserting first
            app.routes.insert(0, Route("/static/{path:path}", self._serve_local_static))
            self._local_static_path = local_static_path
            logger.info("Mounted local static files from %s", local_static_path)
        else:
            self._local_static_path = None
            logger.debug("No local static directory at: %s", local_static_path)

        # IMPORTANT: Register specific routes AFTER the generic static route
        # insert(0, ...) puts this at position 0, so it's checked FIRST
        # This ensures /static/assets/app.js is handled before the generic route
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

    async def _serve_local_static(self, request: Any) -> Any:
        """Serve files from local static directory."""
        if not self._local_static_path:
            return Response(status_code=404, content="Not found")

        path = request.path_params.get("path", "")
        file_path = self._local_static_path / path

        # Security check: ensure path doesn't escape static directory
        try:
            file_path = file_path.resolve()
            if not str(file_path).startswith(str(self._local_static_path.resolve())):
                return Response(status_code=403, content="Forbidden")
        except (ValueError, OSError):
            return Response(status_code=400, content="Invalid path")

        if file_path.exists() and file_path.is_file():
            # Determine media type
            suffix = file_path.suffix.lower()
            media_types = {
                ".js": "application/javascript",
                ".css": "text/css",
                ".json": "application/json",
                ".html": "text/html",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".svg": "image/svg+xml",
                ".gif": "image/gif",
            }
            media_type = media_types.get(suffix, "application/octet-stream")

            response = FileResponse(str(file_path), media_type=media_type)
            response.headers["Cache-Control"] = "public, max-age=60, must-revalidate"
            return response

        return Response(status_code=404, content="Not found")

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
