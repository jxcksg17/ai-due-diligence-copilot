"""Production API routes. Business logic remains in existing services."""

import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.schemas import (
    ClaimEvidenceRequest,
    ClaimEvidenceResponse,
    QueryRequest,
    QueryResponse,
    RiskRadarRequest,
    RiskRadarResponse,
    TemporalRevenueRequest,
    TemporalRevenueResponse,
)
from app.api.services import ProductionAPI
from app.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


def get_production_api(request: Request) -> ProductionAPI:
    return request.app.state.production_api


COMMON_ERRORS = {
    400: {"description": "Invalid or unsupported request"},
    404: {"description": "Requested filing not found"},
    409: {"description": "Ambiguous document scope"},
    422: {"description": "Request validation or missing evidence error"},
    429: {"description": "Local AI capacity exhausted"},
    503: {"description": "Database or local model provider unavailable"},
    504: {"description": "Local model request timed out"},
}


@router.post("/query", response_model=QueryResponse, tags=["grounded Q&A"], responses=COMMON_ERRORS)
def grounded_query(
    payload: QueryRequest,
    db: Session = Depends(get_db),
    service: ProductionAPI = Depends(get_production_api),
) -> QueryResponse:
    logger.info("grounded_query", extra={"company": payload.company, "top_k": payload.top_k})
    return service.query(db, payload)


@router.post(
    "/temporal/revenue",
    response_model=TemporalRevenueResponse,
    tags=["deterministic financial analysis"],
    responses=COMMON_ERRORS,
)
def temporal_revenue(
    payload: TemporalRevenueRequest,
    db: Session = Depends(get_db),
    service: ProductionAPI = Depends(get_production_api),
) -> TemporalRevenueResponse:
    logger.info("temporal_revenue", extra={"company": payload.company})
    return service.compare_revenue(db, payload)


@router.post(
    "/risk-radar/compare", response_model=RiskRadarResponse, tags=["risk analysis"], responses=COMMON_ERRORS
)
def risk_radar(
    payload: RiskRadarRequest,
    db: Session = Depends(get_db),
    service: ProductionAPI = Depends(get_production_api),
) -> RiskRadarResponse:
    logger.info("risk_radar", extra={"company": payload.company})
    return service.compare_risks(db, payload)


@router.post(
    "/claim-evidence/assess",
    response_model=ClaimEvidenceResponse,
    tags=["management claims"],
    responses=COMMON_ERRORS,
)
def claim_evidence(
    payload: ClaimEvidenceRequest,
    db: Session = Depends(get_db),
    service: ProductionAPI = Depends(get_production_api),
) -> ClaimEvidenceResponse:
    logger.info("claim_evidence", extra={"company": payload.company, "fiscal_year": payload.fiscal_year})
    return service.assess_claim(db, payload)
