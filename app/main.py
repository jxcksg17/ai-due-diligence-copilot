"""FastAPI application factory and production entrypoint."""

from fastapi import FastAPI

from app.api import health
from app.api.exception_handlers import install_exception_handlers
from app.api.routes import router as api_router
from app.api.services import ProductionAPI, ProductionAPIService
from app.config import Settings, get_settings
from app.db.session import engine
from app.observability import configure_logging, install_request_middleware
from app.readiness import ReadinessChecker


def create_app(
    settings: Settings | None = None,
    production_api: ProductionAPI | None = None,
    readiness_checker: ReadinessChecker | None = None,
) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved)
    app = FastAPI(
        title=resolved.app_name,
        description="Evidence-grounded financial intelligence over company filings.",
        version="1.0.0",
    )
    app.state.settings = resolved
    app.state.production_api = production_api or ProductionAPIService(resolved)
    app.state.readiness_checker = readiness_checker or ReadinessChecker(
        settings=resolved, engine=engine
    )
    install_request_middleware(app, resolved)
    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(api_router, prefix=resolved.api_prefix)
    return app


app = create_app()
