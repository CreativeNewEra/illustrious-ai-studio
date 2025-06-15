"""API server package for Illustrious AI Studio.

This package bundles the components required to run the Model Context Protocol
(MCP) API server. It exposes helpers for constructing the FastAPI application
and launching the server, along with utilities for asynchronous task
processing and structured logging.

Modules
-------
api
    FastAPI application factory and server entry point. Provides
    :func:`create_api_app` and :func:`run_mcp_server`.

tasks
    Celery task definitions used for background image generation.

logging_utils
    Utilities for attaching per-request identifiers to log messages.
"""

from .api import create_api_app, run_mcp_server

__all__ = ["create_api_app", "run_mcp_server"]
