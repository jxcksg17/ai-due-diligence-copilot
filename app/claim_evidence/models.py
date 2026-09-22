"""M10 management-claim, evidence, and deterministic assessment models."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from app.generation.service import GroundedGenerationResult, NumberedEvidence
from app.retrieval.vector_store import VectorSearchResult
from app.temporal.models import TemporalNumericChange


class ManagementClaimType(StrEnum):
    REVENUE_AMOUNT = "revenue_amount"
    COMPOUND_PERFORMANCE = "compound_performance"
    AMBIGUOUS_FINANCIAL = "ambiguous_financial"
    OBJECTIVE_UNAVAILABLE = "objective_unavailable"
    QUALITATIVE = "qualitative"


class ClaimEvidenceState(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_OBJECTIVELY_VERIFIABLE = "not_objectively_verifiable"


@dataclass(frozen=True)
class ClaimSource:
    """The attributable issuer statement and complete source provenance."""

    document_id: int
    company: str
    document_type: str
    fiscal_year: int
    source_path: str
    source_url: str
    source_title: str
    source_date: str
    exhibit_label: str
    page_number: int
    chunk_ids: tuple[int, ...]
    speaker_name: str
    speaker_role: str
    quote_text: str
    evidence: VectorSearchResult

    def __post_init__(self) -> None:
        if self.document_type != "earnings_release":
            raise ValueError("management claim source must be an earnings_release")
        if not self.chunk_ids or self.evidence.chunk_id not in self.chunk_ids:
            raise ValueError("claim source must retain its source chunk provenance")
        if self.evidence.document_id != self.document_id:
            raise ValueError("claim source evidence must match its source document")
        if not self.speaker_name.strip() or not self.speaker_role.strip():
            raise ValueError("management claim requires an attributed speaker")
        if not self.source_url.startswith("https://www.sec.gov/Archives/edgar/"):
            raise ValueError("claim source must retain its official SEC exhibit URL")


@dataclass(frozen=True)
class ManagementClaim:
    """One constrained, inspectable claim derived from an attributed quote."""

    claim_id: str
    original_text: str
    normalized_text: str
    claim_type: ManagementClaimType
    source: ClaimSource
    metric: str | None = None
    stated_amount_millions: Decimal | None = None
    display_precision_millions: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.original_text.strip():
            raise ValueError("management claim text and ID must not be empty")
        if self.stated_amount_millions is not None:
            if self.metric is None or self.display_precision_millions is None:
                raise ValueError("numeric claims require a metric and display precision")
            if self.display_precision_millions <= 0:
                raise ValueError("display precision must be positive")


@dataclass(frozen=True)
class ClaimEvidenceAssessment:
    """Authoritative application comparison, separate from LLM interpretation."""

    claim: ManagementClaim
    state: ClaimEvidenceState
    rationale: str
    filing_evidence: tuple[NumberedEvidence, ...]
    temporal_calculation: TemporalNumericChange | None = None
    actual_amount_millions: Decimal | None = None
    rounding_tolerance_millions: Decimal | None = None

    @property
    def claim_source_evidence(self) -> NumberedEvidence:
        return NumberedEvidence(1, self.claim.source.evidence)

    @property
    def all_evidence(self) -> tuple[NumberedEvidence, ...]:
        return (self.claim_source_evidence, *self.filing_evidence)

    @property
    def has_filing_evidence(self) -> bool:
        return bool(self.filing_evidence)

    def as_prompt_block(self) -> str:
        amount_lines = ""
        if self.claim.stated_amount_millions is not None:
            amount_lines += (
                f"\nClaimed amount: {self.claim.stated_amount_millions:,} USD millions"
            )
        if self.actual_amount_millions is not None:
            amount_lines += (
                f"\nFiling amount: {self.actual_amount_millions:,} USD millions"
            )
        if self.rounding_tolerance_millions is not None:
            amount_lines += (
                "\nAllowed rounding tolerance: "
                f"{self.rounding_tolerance_millions:,} USD millions"
            )
        return (
            "Deterministic claim-vs-evidence assessment "
            "(authoritative; do not recompute or override):\n"
            f"Original attributed claim: {self.claim.original_text}\n"
            f"Normalized claim: {self.claim.normalized_text}\n"
            f"Classification: {self.state.value}\n"
            f"Reason: {self.rationale}"
            f"{amount_lines}\n"
            "Evidence ID [1] is the management-claim source. Evidence IDs [2+] "
            "are underlying filing evidence when available. MANDATORY CITATION "
            "CONTRACT: put [1] immediately after the sentence that states or "
            "paraphrases management's claim; when filing evidence exists, put at "
            "least one filing marker immediately after the filing fact it supports; "
            "and make citation_ids exactly equal the unique inline markers. Do not "
            "infer intent, truthfulness, or deception."
        )


@dataclass(frozen=True)
class ClaimEvidenceAnalysis:
    assessment: ClaimEvidenceAssessment
    interpretation: GroundedGenerationResult
