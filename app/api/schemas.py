"""Stable public HTTP contracts for the production API."""

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ErrorDetail(StrictModel):
    code: str
    message: str
    request_id: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(StrictModel):
    error: ErrorDetail


class EvidenceReference(StrictModel):
    evidence_id: int | None = None
    chunk_id: int
    document_id: int
    company: str
    document_type: str
    fiscal_year: int
    page_number: int
    excerpt: str
    vector_similarity: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


class VerificationClaim(StrictModel):
    claim_text: str
    citation_ids: list[int]
    status: Literal["supported", "unsupported", "ambiguous", "error"]
    entailment: float | None = None
    contradiction: float | None = None
    neutral: float | None = None
    error: str | None = None


class QueryRequest(StrictModel):
    question: str = Field(min_length=3, max_length=2_000)
    company: str = Field(min_length=1, max_length=255)
    document_type: str | None = Field(default=None, min_length=1, max_length=50)
    fiscal_year: int | None = Field(default=None, ge=1900, le=2200)
    document_id: int | None = Field(default=None, ge=1)
    top_k: int = Field(default=5, ge=1, le=10)


class QueryResponse(StrictModel):
    answer: str
    citation_ids: list[int]
    insufficient_evidence: bool
    verification_state: Literal["verified", "flagged", "insufficient_evidence"]
    evidence: list[EvidenceReference]
    claim_verifications: list[VerificationClaim]


class TemporalRevenueRequest(StrictModel):
    company: str = Field(min_length=1, max_length=255)
    older_year: int = Field(ge=1900, le=2200)
    newer_year: int = Field(ge=1900, le=2200)
    document_type: str = Field(default="10-K", min_length=1, max_length=50)
    top_k: int = Field(default=10, ge=2, le=20)
    rounding_places: int = Field(default=1, ge=0, le=4)


class FinancialValueResponse(StrictModel):
    fiscal_year: int
    amount: Decimal
    unit: str
    evidence: EvidenceReference


class TemporalRevenueResponse(StrictModel):
    metric: str
    older: FinancialValueResponse
    newer: FinancialValueResponse
    absolute_change: Decimal
    percentage_change: Decimal | None
    state: Literal["increased", "declined", "unchanged"]
    formula: str
    rounding_places: int


class RiskRadarRequest(StrictModel):
    company: str = Field(min_length=1, max_length=255)
    older_year: int = Field(ge=1900, le=2200)
    newer_year: int = Field(ge=1900, le=2200)
    document_type: str = Field(default="10-K", min_length=1, max_length=50)
    topic_keys: list[str] | None = Field(default=None, min_length=1, max_length=4)
    top_k: int = Field(default=2, ge=1, le=5)


class RiskSignalResponse(StrictModel):
    fiscal_year: int
    presence: Literal["disclosed", "not_found"]
    qualifying_passage_count: int
    complete_item_1a_scan: bool
    evidence: list[EvidenceReference]


class RiskComparisonResponse(StrictModel):
    topic_key: str
    topic_label: str
    temporal_state: str
    older: RiskSignalResponse
    newer: RiskSignalResponse
    error: str | None = None


class RiskRadarResponse(StrictModel):
    company: str
    document_type: str
    older_year: int
    newer_year: int
    comparisons: list[RiskComparisonResponse]


class ClaimEvidenceRequest(StrictModel):
    company: str = Field(min_length=1, max_length=255)
    fiscal_year: int = Field(ge=1900, le=2200)
    claim_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9_]+$")
    older_year: int = Field(default=2024, ge=1900, le=2200)
    newer_year: int = Field(default=2025, ge=1900, le=2200)
    top_k: int = Field(default=10, ge=2, le=20)


class ClaimSourceResponse(StrictModel):
    speaker_name: str
    speaker_role: str
    source_title: str
    source_date: str
    exhibit_label: str
    source_url: str
    quote: str
    evidence: EvidenceReference


class ClaimEvidenceResponse(StrictModel):
    claim_id: str
    normalized_claim: str
    state: Literal[
        "supported",
        "partially_supported",
        "contradicted",
        "ambiguous",
        "insufficient_evidence",
        "not_objectively_verifiable",
    ]
    rationale: str
    claimed_amount_millions: Decimal | None
    filing_amount_millions: Decimal | None
    rounding_tolerance_millions: Decimal | None
    source: ClaimSourceResponse
    filing_evidence: list[EvidenceReference]


class LivenessResponse(StrictModel):
    status: Literal["ok"]
    app_name: str
    environment: str


class DependencyCheck(StrictModel):
    status: Literal["ok", "error"]
    detail: str | None = None


class ReadinessResponse(StrictModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, DependencyCheck]
