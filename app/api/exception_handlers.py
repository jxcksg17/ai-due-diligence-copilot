"""Map domain and operational failures to one safe API error envelope."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api.errors import (
    AIConcurrencyLimitError,
    ModelTimeoutError,
    ModelUnavailableError,
    UnsupportedOperationError,
)
from app.api.schemas import ErrorDetail, ErrorResponse
from app.claim_evidence.extraction import (
    AmbiguousClaimSourceError,
    InvalidClaimSourceError,
    MissingClaimSourceError,
)
from app.financial.revenue_growth import FinancialInputError
from app.generation.service import CitationValidationError, StructuredResponseError
from app.observability import get_request_id
from app.retrieval.metadata import (
    AmbiguousMetadataError,
    MetadataConflictError,
    MissingMetadataScopeError,
    MissingSemanticQueryError,
    UnsupportedMetadataError,
    UnsupportedTemporalRequestError,
)
from app.risk_radar.section import RiskSectionError
from app.risk_radar.service import RiskRadarDocumentError
from app.risk_radar.taxonomy import UnsupportedRiskTopicError
from app.temporal.financial import TemporalFinancialInputError
from app.temporal.resolution import (
    AmbiguousTemporalDocumentError,
    InvalidTemporalRangeError,
    UnavailableTemporalDocumentError,
)


logger = logging.getLogger("app.errors")


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "location": list(item.get("loc", ())),
                "message": item.get("msg", "invalid value"),
                "type": item.get("type", "validation_error"),
            }
            for item in exc.errors()
        ]
        return _error(request, 422, "invalid_request", "request validation failed", details)

    unavailable = (
        UnsupportedMetadataError,
        UnavailableTemporalDocumentError,
        MissingClaimSourceError,
        RiskRadarDocumentError,
    )
    ambiguous = (
        AmbiguousMetadataError,
        AmbiguousTemporalDocumentError,
        AmbiguousClaimSourceError,
    )
    invalid = (
        MetadataConflictError,
        MissingMetadataScopeError,
        MissingSemanticQueryError,
        InvalidTemporalRangeError,
        FinancialInputError,
        TemporalFinancialInputError,
        InvalidClaimSourceError,
        RiskSectionError,
    )
    unsupported = (
        UnsupportedTemporalRequestError,
        UnsupportedRiskTopicError,
        UnsupportedOperationError,
    )

    async def unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error(request, 404, "unavailable_resource", str(exc))

    async def ambiguous_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error(request, 409, "ambiguous_scope", str(exc))

    async def invalid_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error(request, 422, "invalid_domain_input", str(exc))

    async def unsupported_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error(request, 400, "unsupported_operation", str(exc))

    async def capacity_handler(request: Request, exc: Exception) -> JSONResponse:
        response = _error(request, 429, "capacity_exceeded", str(exc))
        response.headers["Retry-After"] = "5"
        return response

    async def timeout_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error(request, 504, "model_timeout", str(exc))

    async def model_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error(request, 503, "model_unavailable", str(exc))

    async def database_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error(request, 503, "database_unavailable", "database is unavailable")

    async def model_response_handler(request: Request, exc: Exception) -> JSONResponse:
        return _error(
            request,
            502,
            "invalid_model_response",
            "model output failed deterministic validation",
        )

    for exception in unavailable:
        app.add_exception_handler(exception, unavailable_handler)
    for exception in ambiguous:
        app.add_exception_handler(exception, ambiguous_handler)
    for exception in invalid:
        app.add_exception_handler(exception, invalid_handler)
    for exception in unsupported:
        app.add_exception_handler(exception, unsupported_handler)
    app.add_exception_handler(AIConcurrencyLimitError, capacity_handler)
    app.add_exception_handler(ModelTimeoutError, timeout_handler)
    app.add_exception_handler(ModelUnavailableError, model_handler)
    app.add_exception_handler(SQLAlchemyError, database_handler)
    app.add_exception_handler(CitationValidationError, model_response_handler)
    app.add_exception_handler(StructuredResponseError, model_response_handler)

    @app.exception_handler(Exception)
    async def application_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unhandled_request_error",
            extra={
                "request_id": get_request_id(),
                "event_fields": {"error_class": type(exc).__name__},
            },
        )
        return _error(request, 500, "internal_error", "an internal error occurred")


def _error(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: list[dict] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=getattr(request.state, "request_id", get_request_id()),
            details=details,
        )
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))
