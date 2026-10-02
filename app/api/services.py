"""Production orchestration over the existing M3-M10 business services."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from sqlalchemy.orm import Session

from app.api.concurrency import AIConcurrencyGuard
from app.api.errors import (
    ModelTimeoutError,
    ModelUnavailableError,
    UnsupportedOperationError,
)
from app.api.schemas import (
    ClaimEvidenceRequest,
    ClaimEvidenceResponse,
    ClaimSourceResponse,
    EvidenceReference,
    FinancialValueResponse,
    QueryRequest,
    QueryResponse,
    RiskComparisonResponse,
    RiskRadarRequest,
    RiskRadarResponse,
    RiskSignalResponse,
    TemporalRevenueRequest,
    TemporalRevenueResponse,
    VerificationClaim,
)
from app.claim_evidence.extraction import ManagementClaimExtractor
from app.claim_evidence.models import ManagementClaimType
from app.claim_evidence.service import (
    ClaimEvidenceService,
    assess_claim_against_revenue,
)
from app.config import Settings
from app.embeddings.client import get_embedding_client
from app.generation.llm_client import get_llm_client
from app.generation.service import GroundedGenerationService
from app.retrieval.base import Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical_store import LexicalStore
from app.retrieval.reranker import get_reranker
from app.retrieval.vector_store import VectorSearchResult, VectorStore
from app.risk_radar.discovery import RiskEvidenceDiscovery
from app.risk_radar.service import RiskRadarService, align_risk_signals
from app.risk_radar.taxonomy import resolve_risk_topics
from app.temporal.financial import TemporalFinancialComparisonService
from app.verification.service import CitationVerificationService
from app.verification.verifier import LocalNLIVerifier


class ProductionAPI(Protocol):
    def query(self, db: Session, request: QueryRequest) -> QueryResponse: ...

    def compare_revenue(
        self, db: Session, request: TemporalRevenueRequest
    ) -> TemporalRevenueResponse: ...

    def compare_risks(
        self, db: Session, request: RiskRadarRequest
    ) -> RiskRadarResponse: ...

    def assess_claim(
        self, db: Session, request: ClaimEvidenceRequest
    ) -> ClaimEvidenceResponse: ...


@dataclass(frozen=True)
class _StaticRetriever:
    results: list[VectorSearchResult]

    def search(self, db: Session, query: str, **_: Any) -> list[VectorSearchResult]:
        return self.results


class ProductionAPIService:
    """Build local models only for the stage that needs them, then release them."""

    def __init__(
        self,
        settings: Settings,
        *,
        guard: AIConcurrencyGuard | None = None,
    ) -> None:
        self._settings = settings
        self._guard = guard or AIConcurrencyGuard(
            limit=settings.ai_max_concurrency,
            queue_timeout_seconds=settings.ai_queue_timeout_seconds,
        )

    def query(self, db: Session, request: QueryRequest) -> QueryResponse:
        with self._guard.slot():
            results = self._retrieve(
                db,
                request.question,
                company=request.company,
                document_type=request.document_type,
                fiscal_year=request.fiscal_year,
                document_id=request.document_id,
                top_k=request.top_k,
            )
            try:
                generation = GroundedGenerationService(
                    retriever=_StaticRetriever(results),
                    llm_client=get_llm_client(self._settings),
                ).answer(
                    db,
                    request.question,
                    company=request.company,
                    document_type=request.document_type,
                    fiscal_year=request.fiscal_year,
                    document_id=request.document_id,
                    top_k=request.top_k,
                )
            except Exception as exc:
                self._raise_model_transport(exc)
                raise
            finally:
                _unload_ollama(self._settings)
                _release_local_resources()

            if generation.answer.insufficient_evidence:
                verification_state = "insufficient_evidence"
                verified_claims = ()
            else:
                try:
                    verifier = self._build_verifier()
                    verification = CitationVerificationService(verifier).verify_generation(
                        generation
                    )
                    verification_state = verification.status.value
                    verified_claims = verification.claims
                except Exception as exc:
                    self._raise_model_transport(exc)
                    raise
                finally:
                    _release_local_resources()

        return QueryResponse(
            answer=generation.answer.answer,
            citation_ids=generation.answer.citation_ids,
            insufficient_evidence=generation.answer.insufficient_evidence,
            verification_state=verification_state,
            evidence=[
                _evidence(item.result, evidence_id=item.evidence_id)
                for item in generation.evidence
            ],
            claim_verifications=[
                VerificationClaim(
                    claim_text=claim.claim_text,
                    citation_ids=list(claim.citation_ids),
                    status=claim.status.value,
                    entailment=(
                        claim.probabilities.entailment
                        if claim.probabilities is not None
                        else None
                    ),
                    contradiction=(
                        claim.probabilities.contradiction
                        if claim.probabilities is not None
                        else None
                    ),
                    neutral=(
                        claim.probabilities.neutral
                        if claim.probabilities is not None
                        else None
                    ),
                    error=claim.error,
                )
                for claim in verified_claims
            ],
        )

    def compare_revenue(
        self, db: Session, request: TemporalRevenueRequest
    ) -> TemporalRevenueResponse:
        with self._guard.slot():
            retriever = self._build_hybrid()
            try:
                _, comparison = TemporalFinancialComparisonService(
                    retriever=retriever
                ).compare_total_net_sales(
                    db,
                    company=request.company,
                    older_year=request.older_year,
                    newer_year=request.newer_year,
                    document_type=request.document_type,
                    top_k=request.top_k,
                    rounding_places=request.rounding_places,
                )
            finally:
                del retriever
                _release_local_resources()
        return TemporalRevenueResponse(
            metric=comparison.metric,
            older=_financial_value(comparison.older),
            newer=_financial_value(comparison.newer),
            absolute_change=comparison.absolute_change,
            percentage_change=comparison.percentage_change,
            state=comparison.category.value,
            formula="((newer - older) / older) * 100",
            rounding_places=comparison.rounding_places,
        )

    def compare_risks(
        self, db: Session, request: RiskRadarRequest
    ) -> RiskRadarResponse:
        topics = resolve_risk_topics(
            tuple(request.topic_keys) if request.topic_keys is not None else None
        )
        with self._guard.slot():
            retriever = self._build_hybrid()
            discovery_service = RiskRadarService(
                discovery=RiskEvidenceDiscovery(retriever)
            )
            try:
                pairs = [
                    discovery_service.discover_topic(
                        db,
                        company=request.company,
                        document_type=request.document_type,
                        older_year=request.older_year,
                        newer_year=request.newer_year,
                        topic=topic,
                        top_k=request.top_k,
                    )
                    for topic in topics
                ]
            finally:
                del discovery_service, retriever
                _release_local_resources()

            try:
                verifier = self._build_verifier()
                comparisons = [
                    align_risk_signals(pair, verifier=verifier) for pair in pairs
                ]
            except Exception as exc:
                self._raise_model_transport(exc)
                raise
            finally:
                _release_local_resources()

        return RiskRadarResponse(
            company=request.company,
            document_type=request.document_type,
            older_year=request.older_year,
            newer_year=request.newer_year,
            comparisons=[
                RiskComparisonResponse(
                    topic_key=item.topic.key,
                    topic_label=item.topic.label,
                    temporal_state=item.temporal_state.value,
                    older=_risk_signal(item.older),
                    newer=_risk_signal(item.newer),
                    error=item.error,
                )
                for item in comparisons
            ],
        )

    def assess_claim(
        self, db: Session, request: ClaimEvidenceRequest
    ) -> ClaimEvidenceResponse:
        claims = ManagementClaimExtractor().extract(
            db,
            company=request.company,
            fiscal_year=request.fiscal_year,
        )
        claim = next((item for item in claims if item.claim_id == request.claim_id), None)
        if claim is None:
            supported = ", ".join(sorted(item.claim_id for item in claims))
            raise UnsupportedOperationError(
                f"unsupported claim_id; available claim IDs: {supported}"
            )

        with self._guard.slot():
            if claim.claim_type in {
                ManagementClaimType.REVENUE_AMOUNT,
                ManagementClaimType.COMPOUND_PERFORMANCE,
            }:
                retriever = self._build_hybrid()
                try:
                    assessment = ClaimEvidenceService(retriever=retriever).assess(
                        db,
                        claim=claim,
                        older_year=request.older_year,
                        newer_year=request.newer_year,
                        top_k=request.top_k,
                    )
                finally:
                    del retriever
                    _release_local_resources()
            else:
                assessment = assess_claim_against_revenue(claim)

        source = assessment.claim.source
        return ClaimEvidenceResponse(
            claim_id=claim.claim_id,
            normalized_claim=claim.normalized_text,
            state=assessment.state.value,
            rationale=assessment.rationale,
            claimed_amount_millions=claim.stated_amount_millions,
            filing_amount_millions=assessment.actual_amount_millions,
            rounding_tolerance_millions=assessment.rounding_tolerance_millions,
            source=ClaimSourceResponse(
                speaker_name=source.speaker_name,
                speaker_role=source.speaker_role,
                source_title=source.source_title,
                source_date=source.source_date,
                exhibit_label=source.exhibit_label,
                source_url=source.source_url,
                quote=source.quote_text,
                evidence=_evidence(source.evidence, evidence_id=1),
            ),
            filing_evidence=[
                _evidence(item.result, evidence_id=item.evidence_id)
                for item in assessment.filing_evidence
            ],
        )

    def _retrieve(self, db: Session, query: str, **scope: Any) -> list[VectorSearchResult]:
        retriever = self._build_hybrid()
        try:
            return retriever.search(db, query, **scope)
        finally:
            del retriever
            _release_local_resources()

    def _build_hybrid(self) -> HybridRetriever:
        try:
            embedding = get_embedding_client(self._settings)
            reranker = get_reranker(self._settings)
        except Exception as exc:
            self._raise_model_transport(exc)
            raise
        return HybridRetriever(
            vector_store=VectorStore(embedding),
            lexical_store=LexicalStore(),
            reranker=reranker,
            candidate_k=self._settings.reranker_candidate_k,
            rrf_k=self._settings.hybrid_rrf_k,
        )

    def _build_verifier(self) -> LocalNLIVerifier:
        return LocalNLIVerifier(
            model_name=self._settings.citation_verifier_model,
            batch_size=self._settings.citation_verifier_batch_size,
            device="cpu",
        )

    @staticmethod
    def _raise_model_transport(exc: Exception) -> None:
        if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
            raise ModelTimeoutError("local model request exceeded its timeout") from exc
        if isinstance(exc, (ConnectionError, OSError, httpx.TransportError)):
            raise ModelUnavailableError("required local model service is unavailable") from exc


def _evidence(
    result: VectorSearchResult,
    *,
    evidence_id: int | None,
) -> EvidenceReference:
    excerpt = " ".join(result.text.split())
    if len(excerpt) > 500:
        excerpt = excerpt[:497].rstrip() + "..."
    return EvidenceReference(
        evidence_id=evidence_id,
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        company=result.company,
        document_type=result.document_type,
        fiscal_year=result.fiscal_year,
        page_number=result.page_number,
        excerpt=excerpt,
        vector_similarity=result.similarity,
        lexical_score=result.lexical_score,
        fusion_score=result.fusion_score,
        rerank_score=result.rerank_score,
    )


def _financial_value(value: Any) -> FinancialValueResponse:
    return FinancialValueResponse(
        fiscal_year=value.fiscal_year,
        amount=value.amount,
        unit=value.unit,
        evidence=_evidence(
            value.evidence.result,
            evidence_id=value.evidence.evidence_id,
        ),
    )


def _risk_signal(signal: Any) -> RiskSignalResponse:
    return RiskSignalResponse(
        fiscal_year=signal.document.fiscal_year,
        presence=signal.presence.value,
        qualifying_passage_count=signal.qualifying_passage_count,
        complete_item_1a_scan=signal.complete_item_1a_scan,
        evidence=[
            _evidence(item.result, evidence_id=item.evidence_id)
            for item in signal.evidence
        ],
    )


def _release_local_resources() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _unload_ollama(settings: Settings) -> None:
    try:
        from ollama import Client

        Client(
            host=settings.ollama_host,
            timeout=settings.readiness_timeout_seconds,
        ).generate(model=settings.llm_model, prompt="", keep_alive=0)
    except Exception:
        # Unloading is best-effort cleanup; the original request outcome remains
        # authoritative and readiness will expose a genuinely unavailable server.
        pass
