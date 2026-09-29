"""Deterministic and structured-domain evaluation slices."""

from decimal import Decimal
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.claim_evidence.extraction import ManagementClaimExtractor
from app.claim_evidence.service import ClaimEvidenceService
from app.financial.revenue_growth import SourceFinancialValue, calculate_revenue_growth
from app.retrieval.base import Retriever
from app.retrieval.metadata import (
    AmbiguousMetadataError,
    MetadataCatalog,
    MetadataQueryError,
    UnsupportedMetadataError,
    build_metadata_query_plan,
)
from app.retrieval.vector_store import VectorSearchResult
from app.temporal.financial import calculate_temporal_numeric_change
from app.temporal.models import (
    SourcePeriod,
    TemporalChangeCategory,
    TemporalEvidence,
    TemporalFinancialValue,
    TemporalNumericChange,
)
from app.temporal.resolution import (
    TemporalDocumentResolver,
    TemporalResolutionError,
    UnavailableTemporalDocumentError,
)
from evals.metrics import classification_metrics, mean, numeric_matches
from evals.schema import EvaluationCase


def evaluate_metadata(
    db: Session, cases: list[EvaluationCase]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    documents = MetadataCatalog().list_documents(db)
    expected: list[str] = []
    predicted: list[str] = []
    outcomes: dict[str, str] = {}
    latencies: list[float] = []
    for case in cases:
        if case.question is None or case.expected_state is None:
            raise ValueError(f"metadata case {case.id} is incomplete")
        scope = case.scope
        started = perf_counter()
        try:
            plan = build_metadata_query_plan(
                case.question,
                documents,
                company=scope.company if scope else None,
                document_type=scope.document_type if scope else None,
                fiscal_year=scope.fiscal_year if scope else None,
            )
            state = "resolved"
            if scope is not None:
                matches_scope = (
                    plan.document.company == scope.company
                    and plan.document.document_type == scope.document_type
                    and plan.document.fiscal_year == scope.fiscal_year
                )
                if not matches_scope:
                    state = "wrong_scope"
            expected_semantic = case.expected.get("semantic_query")
            if expected_semantic and plan.semantic_query != expected_semantic:
                state = "wrong_semantic_query"
        except UnsupportedMetadataError:
            state = "unavailable"
        except AmbiguousMetadataError:
            state = "ambiguous"
        except MetadataQueryError:
            state = "error"
        latencies.append(perf_counter() - started)
        expected.append(case.expected_state)
        predicted.append(state)
        outcomes[case.id] = state
    scored = classification_metrics(expected, predicted)
    return (
        {
            "accuracy": round(scored.accuracy, 6),
            "macro_f1": round(scored.macro_f1, 6),
            "case_count": scored.count,
        },
        {"mean": round(mean(latencies), 6), "max": round(max(latencies), 6)},
        outcomes,
    )


def evaluate_financial(
    cases: list[EvaluationCase],
    *,
    real_temporal: TemporalNumericChange | None,
) -> dict[str, Any]:
    numeric: list[bool] = []
    classifications: list[bool] = []
    rounding: list[bool] = []
    provenance: list[bool] = []
    for case in cases:
        values = case.expected
        if values.get("execution") == "real_temporal":
            if real_temporal is None:
                raise ValueError("real financial case requires a temporal result")
            percentage = real_temporal.percentage_change
            actual_direction = _direction(real_temporal.absolute_change)
            has_provenance = (
                real_temporal.older.evidence.result.document_id
                != real_temporal.newer.evidence.result.document_id
                and real_temporal.older.evidence.result.page_number > 0
                and real_temporal.newer.evidence.result.page_number > 0
            )
        else:
            prior = _financial_value(
                amount=Decimal(str(values["prior"])), fiscal_year=2024, evidence_id=2
            )
            current = _financial_value(
                amount=Decimal(str(values["current"])), fiscal_year=2025, evidence_id=1
            )
            result = calculate_revenue_growth(
                current,
                prior,
                rounding_places=int(values["rounding_places"]),
            )
            percentage = result.percentage_change
            actual_direction = _direction(current.amount - prior.amount)
            has_provenance = (
                result.current.evidence_id == 1
                and result.prior.evidence_id == 2
                and result.current.chunk_id != result.prior.chunk_id
            )
        numeric.append(
            percentage is not None
            and numeric_matches(percentage, Decimal(str(values["percentage"])))
        )
        classifications.append(actual_direction == values["direction"])
        rounding.append(
            percentage is not None
            and percentage.as_tuple().exponent == -int(values["rounding_places"])
        )
        provenance.append(has_provenance is bool(values["provenance"]))
    return {
        "numeric_accuracy": _boolean_rate(numeric),
        "classification_accuracy": _boolean_rate(classifications),
        "rounding_accuracy": _boolean_rate(rounding),
        "provenance_accuracy": _boolean_rate(provenance),
        "case_count": len(cases),
    }


def evaluate_temporal_numeric(
    cases: list[EvaluationCase],
    *,
    real_temporal: TemporalNumericChange | None,
    db: Session,
) -> tuple[dict[str, Any], dict[str, str]]:
    value_checks: list[bool] = []
    state_expected: list[str] = []
    state_predicted: list[str] = []
    outcomes: dict[str, str] = {}
    resolver = TemporalDocumentResolver()
    for case in cases:
        execution = case.expected.get("execution")
        if execution == "real_disclosure":
            continue
        if execution == "real_numeric":
            if real_temporal is None:
                raise ValueError("real temporal case requires a temporal result")
            result = real_temporal
            correct_values = all(
                (
                    numeric_matches(result.older.amount, case.expected["older_amount"]),
                    numeric_matches(result.newer.amount, case.expected["newer_amount"]),
                    numeric_matches(
                        result.absolute_change, case.expected["absolute_change"]
                    ),
                    result.percentage_change is not None
                    and numeric_matches(
                        result.percentage_change,
                        case.expected["percentage_change"],
                    ),
                )
            )
            state = result.category.value
        elif execution == "synthetic_numeric":
            result = calculate_temporal_numeric_change(
                _temporal_value(
                    amount=Decimal(str(case.expected["older_amount"])),
                    fiscal_year=2024,
                    period=SourcePeriod.OLDER,
                    evidence_id=1,
                    document_id=1,
                ),
                _temporal_value(
                    amount=Decimal(str(case.expected["newer_amount"])),
                    fiscal_year=2025,
                    period=SourcePeriod.NEWER,
                    evidence_id=2,
                    document_id=2,
                ),
            )
            correct_values = (
                numeric_matches(result.absolute_change, case.expected["absolute_change"])
                and result.percentage_change is not None
                and numeric_matches(
                    result.percentage_change, case.expected["percentage_change"]
                )
            )
            state = result.category.value
        elif execution == "resolution":
            try:
                resolver.resolve(
                    db,
                    company="Apple",
                    document_type="10-K",
                    older_year=int(case.expected["older_year"]),
                    newer_year=int(case.expected["newer_year"]),
                )
                state = "resolved"
            except UnavailableTemporalDocumentError:
                state = "unavailable"
            except TemporalResolutionError:
                state = "error"
            correct_values = True
        else:
            raise ValueError(f"unsupported temporal execution {execution!r}")
        value_checks.append(correct_values)
        state_expected.append(str(case.expected_state))
        state_predicted.append(state)
        outcomes[case.id] = state
    state_scores = classification_metrics(state_expected, state_predicted)
    return (
        {
            "numeric_accuracy": _boolean_rate(value_checks),
            "state_accuracy": round(state_scores.accuracy, 6),
            "numeric_case_count": len(value_checks),
        },
        outcomes,
    )


def evaluate_claim_evidence(
    db: Session,
    cases: list[EvaluationCase],
    *,
    retriever: Retriever,
) -> tuple[dict[str, Any], dict[str, str]]:
    claims = {
        claim.claim_id: claim
        for claim in ManagementClaimExtractor().extract(
            db, company="Apple", fiscal_year=2025
        )
    }
    service = ClaimEvidenceService(retriever=retriever)
    expected_states: list[str] = []
    actual_states: list[str] = []
    numeric_checks: list[bool] = []
    claim_provenance: list[bool] = []
    filing_provenance: list[bool] = []
    outcomes: dict[str, str] = {}
    for case in cases:
        claim_id = str(case.expected["claim_id"])
        assessment = service.assess(db, claim=claims[claim_id])
        state = assessment.state.value
        expected_states.append(str(case.expected_state))
        actual_states.append(state)
        outcomes[case.id] = state
        source = assessment.claim.source
        claim_provenance.append(
            source.document_type == "earnings_release"
            and source.evidence.document_id == source.document_id
            and source.evidence.chunk_id in source.chunk_ids
            and source.source_url.startswith("https://www.sec.gov/Archives/edgar/")
        )
        expects_filing = bool(case.expected["filing_provenance"])
        has_filing = bool(assessment.filing_evidence)
        filing_provenance.append(
            has_filing == expects_filing
            and all(
                item.result.document_type == "10-K"
                and item.result.document_id != source.document_id
                for item in assessment.filing_evidence
            )
        )
        if "actual_amount" in case.expected:
            numeric_checks.append(
                assessment.actual_amount_millions is not None
                and numeric_matches(
                    assessment.actual_amount_millions,
                    case.expected["actual_amount"],
                )
            )
    scored = classification_metrics(expected_states, actual_states)
    return (
        {
            "state_accuracy": round(scored.accuracy, 6),
            "macro_precision": round(scored.macro_precision, 6),
            "macro_recall": round(scored.macro_recall, 6),
            "macro_f1": round(scored.macro_f1, 6),
            "numeric_agreement": _boolean_rate(numeric_checks),
            "claim_source_provenance": _boolean_rate(claim_provenance),
            "filing_evidence_provenance": _boolean_rate(filing_provenance),
            "case_count": len(cases),
        },
        outcomes,
    )


def _financial_value(
    *, amount: Decimal, fiscal_year: int, evidence_id: int
) -> SourceFinancialValue:
    return SourceFinancialValue(
        metric="total_net_sales",
        fiscal_year=fiscal_year,
        amount=amount,
        unit="USD millions",
        evidence_id=evidence_id,
        chunk_id=evidence_id,
        page_number=1,
        document_id=evidence_id,
    )


def _temporal_value(
    *,
    amount: Decimal,
    fiscal_year: int,
    period: SourcePeriod,
    evidence_id: int,
    document_id: int,
) -> TemporalFinancialValue:
    result = VectorSearchResult(
        chunk_id=evidence_id,
        text="synthetic evaluator fixture",
        page_number=1,
        document_id=document_id,
        company="Fixture",
        document_type="10-K",
        fiscal_year=fiscal_year,
        cosine_distance=None,
        similarity=None,
    )
    return TemporalFinancialValue(
        metric="total_net_sales",
        amount=amount,
        unit="USD millions",
        fiscal_year=fiscal_year,
        evidence=TemporalEvidence(evidence_id, period, result),
    )


def _direction(change: Decimal) -> str:
    if change > 0:
        return "positive"
    if change < 0:
        return "decline"
    return "unchanged"


def _boolean_rate(values: list[bool]) -> float:
    if not values:
        return 1.0
    return round(sum(values) / len(values), 6)
