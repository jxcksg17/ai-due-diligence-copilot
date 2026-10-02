"""Liveness and dependency-aware readiness endpoints."""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.api.schemas import LivenessResponse, ReadinessResponse
from app.readiness import ReadinessChecker

router = APIRouter(tags=["operations"])


def get_readiness_checker(request: Request) -> ReadinessChecker:
    return request.app.state.readiness_checker


@router.get("/health/live", response_model=LivenessResponse)
@router.get("/health", response_model=LivenessResponse, include_in_schema=False)
def liveness(request: Request) -> LivenessResponse:
    settings = request.app.state.settings
    return LivenessResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
    )


@router.get("/health/ready", response_model=ReadinessResponse)
@router.get("/health/db", response_model=ReadinessResponse, include_in_schema=False)
def readiness(request: Request) -> ReadinessResponse | JSONResponse:
    result = get_readiness_checker(request).check()
    if result.status == "ready":
        return result
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=result.model_dump(mode="json"),
    )
