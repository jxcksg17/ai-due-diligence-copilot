"""CLI for quick, retrieval-only, and full local M11 evaluation runs."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import Chunk, Company, Document
from app.db.session import SessionLocal
from app.embeddings.client import get_embedding_client
from app.generation.llm_client import get_llm_client
from app.generation.service import (
    GroundedAnswer,
    GroundedGenerationResult,
    GroundedGenerationService,
    NumberedEvidence,
)
from app.retrieval.base import Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical_store import LexicalStore
from app.retrieval.metadata import (
    AmbiguousMetadataError,
    MetadataCatalog,
    MetadataQueryError,
    build_metadata_query_plan,
)
from app.retrieval.reranker import get_reranker
from app.retrieval.vector_store import VectorSearchResult, VectorStore
from app.risk_radar.discovery import RiskEvidenceDiscovery
from app.risk_radar.models import RiskSignalPair
from app.risk_radar.service import RiskRadarService, align_risk_signals
from app.risk_radar.taxonomy import get_risk_topic
from app.temporal.disclosures import (
    COMPETITION_RISK,
    SUPPLY_CHAIN_RISK,
    DisclosureEvidenceLocator,
    classify_disclosure_change,
)
from app.temporal.financial import TemporalFinancialComparisonService
from app.temporal.models import (
    DisclosurePeriodEvidence,
    SourcePeriod,
    TemporalEvidence,
    TemporalNumericChange,
)
from app.temporal.resolution import (
    TemporalDocumentResolver,
    TemporalResolutionError,
)
from app.verification.service import (
    AnswerVerificationStatus,
    CitationVerificationService,
    VerificationStatus,
)
from app.verification.verifier import LocalNLIVerifier
from evals.domain import (
    evaluate_claim_evidence,
    evaluate_financial,
    evaluate_metadata,
    evaluate_temporal_numeric,
)
from evals.metrics import anchor_coverage, classification_metrics, mean
from evals.regression import compare_reports
from evals.reporting import build_report, load_report, write_report_atomic
from evals.retrieval import RetrievalArtifacts, evaluate_retrieval
from evals.schema import EvaluationCase, EvaluationDataset, load_dataset


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "evals" / "datasets" / "m11_v1.json"
DEFAULT_OUTPUT = ROOT / "evals" / "reports" / "m11_current.json"


@dataclass(frozen=True)
class StaticRetriever:
    results: list[VectorSearchResult]

    def search(self, db: Session, query: str, **_: Any) -> list[VectorSearchResult]:
        return self.results


@dataclass(frozen=True)
class DisclosureInputs:
    case: EvaluationCase
    older: DisclosurePeriodEvidence
    newer: DisclosurePeriodEvidence


@dataclass(frozen=True)
class RetrievalPhase:
    retrieval: RetrievalArtifacts
    generation_evidence: dict[str, list[VectorSearchResult]]
    financial_metrics: dict[str, Any]
    temporal_metrics: dict[str, Any]
    temporal_outcomes: dict[str, str]
    claim_metrics: dict[str, Any]
    claim_outcomes: dict[str, str]
    disclosure_inputs: tuple[DisclosureInputs, ...]
    risk_inputs: tuple[tuple[EvaluationCase, RiskSignalPair], ...]
    phase_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("quick", "retrieval", "full"), default="quick"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--compare", type=Path, help="compare the completed report with a baseline"
    )
    arguments = parser.parse_args()
    dataset = load_dataset(arguments.dataset)
    settings = get_settings()
    started = perf_counter()
    with SessionLocal() as db:
        report = run_evaluation(
            db,
            dataset,
            settings=settings,
            mode=arguments.mode,
        )
    report["latencies_seconds"]["overall"] = round(perf_counter() - started, 6)
    write_report_atomic(report, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if arguments.compare:
        comparison = compare_reports(load_report(arguments.compare), report)
        print("\nRegression comparison:")
        print(json.dumps(comparison, indent=2, sort_keys=True))
        if comparison["overall"] == "regressed":
            raise SystemExit(2)


def run_evaluation(
    db: Session,
    dataset: EvaluationDataset,
    *,
    settings: Settings,
    mode: str,
) -> dict[str, Any]:
    counts = dict(sorted(Counter(case.capability for case in dataset.cases).items()))
    metadata_metrics, metadata_latency, metadata_outcomes = evaluate_metadata(
        db, dataset.by_capability("metadata")
    )
    metrics: dict[str, Any] = {
        "classification": {"metadata": metadata_metrics},
    }
    latencies: dict[str, Any] = {"metadata_resolution": metadata_latency}
    runtime = _runtime_metadata(db)
    case_outcomes = dict(metadata_outcomes)

    if mode == "quick":
        synthetic_financial = [
            case
            for case in dataset.by_capability("financial")
            if case.expected.get("execution") != "real_temporal"
        ]
        metrics["deterministic"] = {
            "financial": evaluate_financial(
                synthetic_financial, real_temporal=None
            )
        }
        metrics["refusal"] = {
            "metadata_safe_failure_rate": _expected_outcome_rate(
                dataset.by_capability("metadata"), metadata_outcomes
            )
        }
    else:
        phase = _run_retrieval_phase(db, dataset, settings=settings)
        metrics["retrieval"] = phase.retrieval.metrics
        metrics["deterministic"] = {
            "financial": phase.financial_metrics,
            "temporal_numeric": phase.temporal_metrics,
        }
        metrics["classification"].update(
            {
                "claim_evidence": phase.claim_metrics,
            }
        )
        case_outcomes.update(phase.temporal_outcomes)
        case_outcomes.update(phase.claim_outcomes)
        latencies.update(phase.retrieval.latencies)
        latencies["retrieval_domain_phase"] = round(phase.phase_seconds, 6)
        if mode == "full":
            _release_local_resources()
            generation, generation_latency = _run_generation_phase(
                db,
                dataset,
                settings=settings,
                evidence=phase.generation_evidence,
            )
            latencies["qwen_generation"] = generation_latency
            _unload_ollama(settings)
            _release_local_resources()
            semantic_metrics, semantic_latency, semantic_outcomes = _run_nli_phase(
                dataset,
                settings=settings,
                generation=generation,
                generation_evidence=phase.generation_evidence,
                disclosure_inputs=phase.disclosure_inputs,
                risk_inputs=phase.risk_inputs,
            )
            case_outcomes.update(semantic_outcomes)
            latencies["citation_and_alignment_verification"] = semantic_latency
            metrics["generation"] = semantic_metrics["generation"]
            metrics["citation"] = semantic_metrics["citation"]
            metrics["classification"]["temporal_disclosure"] = semantic_metrics[
                "temporal_disclosure"
            ]
            metrics["classification"]["risk_radar"] = semantic_metrics[
                "risk_radar"
            ]
            unsupported_outcomes = _unsupported_outcomes(
                db,
                dataset.by_capability("unsupported"),
                metadata_outcomes=metadata_outcomes,
                generation=generation,
                semantic_outcomes=semantic_outcomes,
            )
            metrics["refusal"] = {
                "correct_behavior_rate": _expected_outcome_rate(
                    dataset.by_capability("unsupported"), unsupported_outcomes
                ),
                "case_count": len(unsupported_outcomes),
            }
            case_outcomes.update(unsupported_outcomes)
        else:
            metrics["refusal"] = {
                "metadata_safe_failure_rate": _expected_outcome_rate(
                    dataset.by_capability("metadata"), metadata_outcomes
                )
            }

    return build_report(
        evaluation_version=dataset.evaluation_version,
        git_commit=_git_commit(),
        mode=mode,
        models={
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "generator": settings.llm_model,
            "citation_verifier": settings.citation_verifier_model,
        },
        case_counts=counts,
        metrics=metrics,
        latencies=latencies,
        runtime=runtime,
        case_outcomes=case_outcomes,
    )


def _run_retrieval_phase(
    db: Session, dataset: EvaluationDataset, *, settings: Settings
) -> RetrievalPhase:
    started = perf_counter()
    embedding = get_embedding_client(settings)
    reranker = get_reranker(settings)
    vector_store = VectorStore(embedding)
    hybrid: Retriever = HybridRetriever(
        vector_store=vector_store,
        lexical_store=LexicalStore(),
        reranker=reranker,
        candidate_k=settings.reranker_candidate_k,
        rrf_k=settings.hybrid_rrf_k,
    )
    retrieval_cases = dataset.by_capability("retrieval")
    retrieval = evaluate_retrieval(
        db,
        retrieval_cases,
        embedding_client=embedding,
        reranker=reranker,
        candidate_k=settings.reranker_candidate_k,
        rrf_k=settings.hybrid_rrf_k,
    )

    evidence = {
        case.id: retrieval.final_results[case.id]
        for case in retrieval_cases
        if case.generation is not None
    }
    for case in dataset.by_capability("unsupported"):
        if case.generation is None or case.scope is None or case.question is None:
            continue
        evidence[case.id] = hybrid.search(
            db,
            case.question,
            company=case.scope.company,
            document_type=case.scope.document_type,
            fiscal_year=case.scope.fiscal_year,
            top_k=5,
        )

    temporal_service = TemporalFinancialComparisonService(retriever=hybrid)
    _, temporal_result = temporal_service.compare_total_net_sales(
        db,
        company="Apple",
        older_year=2024,
        newer_year=2025,
        document_type="10-K",
        top_k=10,
    )
    financial_metrics = evaluate_financial(
        dataset.by_capability("financial"), real_temporal=temporal_result
    )
    temporal_metrics, temporal_outcomes = evaluate_temporal_numeric(
        dataset.by_capability("temporal"),
        real_temporal=temporal_result,
        db=db,
    )
    claim_metrics, claim_outcomes = evaluate_claim_evidence(
        db,
        dataset.by_capability("claim_evidence"),
        retriever=hybrid,
    )
    disclosures = _discover_disclosures(
        db, dataset.by_capability("temporal"), retriever=hybrid
    )
    risk_inputs = _discover_risk_inputs(
        db, dataset.by_capability("risk_radar"), retriever=hybrid
    )
    return RetrievalPhase(
        retrieval=retrieval,
        generation_evidence=evidence,
        financial_metrics=financial_metrics,
        temporal_metrics=temporal_metrics,
        temporal_outcomes=temporal_outcomes,
        claim_metrics=claim_metrics,
        claim_outcomes=claim_outcomes,
        disclosure_inputs=tuple(disclosures),
        risk_inputs=tuple(risk_inputs),
        phase_seconds=perf_counter() - started,
    )


def _discover_disclosures(
    db: Session,
    cases: list[EvaluationCase],
    *,
    retriever: Retriever,
) -> list[DisclosureInputs]:
    topics = {
        "supply_chain_risk": SUPPLY_CHAIN_RISK,
        "competition_risk": COMPETITION_RISK,
    }
    resolver = TemporalDocumentResolver()
    locator = DisclosureEvidenceLocator(retriever)
    discovered: list[DisclosureInputs] = []
    for case in cases:
        if case.expected.get("execution") != "real_disclosure":
            continue
        documents = resolver.resolve(
            db,
            company="Apple",
            document_type="10-K",
            older_year=int(case.expected["older_year"]),
            newer_year=int(case.expected["newer_year"]),
        )
        topic = topics[str(case.expected["topic"])]
        older_located = locator.locate(
            db,
            document=documents.older,
            source_period=SourcePeriod.OLDER,
            topic=topic,
            top_k=2,
        )
        newer_located = locator.locate(
            db,
            document=documents.newer,
            source_period=SourcePeriod.NEWER,
            topic=topic,
            top_k=2,
        )
        older = _period_evidence(older_located.results, 2024, SourcePeriod.OLDER, 1)
        newer = _period_evidence(
            newer_located.results,
            2025,
            SourcePeriod.NEWER,
            len(older.evidence) + 1,
        )
        discovered.append(DisclosureInputs(case, older, newer))
    return discovered


def _discover_risk_inputs(
    db: Session,
    cases: list[EvaluationCase],
    *,
    retriever: Retriever,
) -> list[tuple[EvaluationCase, RiskSignalPair]]:
    service = RiskRadarService(discovery=RiskEvidenceDiscovery(retriever))
    inputs: list[tuple[EvaluationCase, RiskSignalPair]] = []
    for case in cases:
        pair = service.discover_topic(
            db,
            company="Apple",
            document_type="10-K",
            older_year=int(case.expected["older_year"]),
            newer_year=int(case.expected["newer_year"]),
            topic=get_risk_topic(str(case.expected["topic"])),
            top_k=2,
        )
        inputs.append((case, pair))
    return inputs


def _run_generation_phase(
    db: Session,
    dataset: EvaluationDataset,
    *,
    settings: Settings,
    evidence: dict[str, list[VectorSearchResult]],
) -> tuple[dict[str, GroundedGenerationResult | None], dict[str, Any]]:
    llm = get_llm_client(settings)
    generated: dict[str, GroundedGenerationResult | None] = {}
    timings: list[float] = []
    cases = [case for case in dataset.cases if case.generation is not None]
    for case in cases:
        if case.question is None:
            raise ValueError(f"generation case {case.id} is missing its question")
        service = GroundedGenerationService(
            retriever=StaticRetriever(evidence[case.id]),
            llm_client=llm,
        )
        started = perf_counter()
        try:
            generated[case.id] = service.answer(
                db,
                case.question,
                company=case.scope.company if case.scope else None,
                document_type=case.scope.document_type if case.scope else None,
                fiscal_year=case.scope.fiscal_year if case.scope else None,
                top_k=5,
            )
        except Exception as exc:
            print(f"generation case {case.id} failed validation: {exc}", file=sys.stderr)
            generated[case.id] = None
        timings.append(perf_counter() - started)
    return generated, _latency_summary(timings)


def _run_nli_phase(
    dataset: EvaluationDataset,
    *,
    settings: Settings,
    generation: dict[str, GroundedGenerationResult | None],
    generation_evidence: dict[str, list[VectorSearchResult]],
    disclosure_inputs: tuple[DisclosureInputs, ...],
    risk_inputs: tuple[tuple[EvaluationCase, RiskSignalPair], ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    verifier = LocalNLIVerifier(
        model_name=settings.citation_verifier_model,
        batch_size=settings.citation_verifier_batch_size,
        device="cpu",
    )
    citation_service = CitationVerificationService(verifier)
    started_all = perf_counter()
    verification_times: list[float] = []
    validity: list[bool] = []
    supported_claims = 0
    evaluated_claims = 0
    answer_relevancy: list[float] = []
    semantic_outcomes: dict[str, str] = {}

    for case in [case for case in dataset.cases if case.generation is not None]:
        result = generation.get(case.id)
        validity.append(result is not None)
        if result is None:
            semantic_outcomes[case.id] = "generation_error"
            continue
        expectation = case.generation
        assert expectation is not None
        if expectation.insufficient_evidence:
            state = (
                "insufficient_evidence"
                if result.answer.insufficient_evidence
                else "answered"
            )
            semantic_outcomes[case.id] = state
            continue
        answer_relevancy.append(
            anchor_coverage(result.answer.answer, expectation.expected_answer_anchors)
        )
        started = perf_counter()
        report = citation_service.verify_generation(result)
        verification_times.append(perf_counter() - started)
        evaluated_claims += len(report.claims)
        supported_claims += sum(
            claim.status == VerificationStatus.SUPPORTED for claim in report.claims
        )
        semantic_outcomes[case.id] = report.status.value
        print(
            "citation audit "
            f"{case.id}: answer={result.answer.answer!r}; "
            f"claims={[(claim.claim_text, claim.status.value) for claim in report.claims]}",
            file=sys.stderr,
        )

    mismatch_case = next(
        case
        for case in dataset.by_capability("unsupported")
        if case.expected.get("execution") == "citation_mismatch"
    )
    source_results = generation_evidence["ret_2025_total_net_sales"]
    mismatch = GroundedGenerationResult(
        answer=GroundedAnswer(
            answer="Apple reported fiscal 2025 total net sales of $999,999 million [1].",
            citation_ids=[1],
            insufficient_evidence=False,
        ),
        evidence=(NumberedEvidence(1, source_results[0]),),
    )
    started = perf_counter()
    mismatch_report = citation_service.verify_generation(mismatch)
    verification_times.append(perf_counter() - started)
    mismatch_detected = mismatch_report.status == AnswerVerificationStatus.FLAGGED
    semantic_outcomes[mismatch_case.id] = (
        "unsupported_claim" if mismatch_detected else "missed_unsupported_claim"
    )

    temporal_expected: list[str] = []
    temporal_actual: list[str] = []
    temporal_times: list[float] = []
    for inputs in disclosure_inputs:
        started = perf_counter()
        category, _, _, _ = classify_disclosure_change(
            older=inputs.older,
            newer=inputs.newer,
            verifier=verifier,
        )
        temporal_times.append(perf_counter() - started)
        temporal_expected.append(str(inputs.case.expected_state))
        temporal_actual.append(category.value)
        semantic_outcomes[inputs.case.id] = category.value
    temporal_scores = classification_metrics(temporal_expected, temporal_actual)

    risk_expected: list[str] = []
    risk_actual: list[str] = []
    risk_times: list[float] = []
    for case, pair in risk_inputs:
        started = perf_counter()
        comparison = align_risk_signals(pair, verifier=verifier)
        risk_times.append(perf_counter() - started)
        risk_expected.append(str(case.expected_state))
        risk_actual.append(comparison.temporal_state.value)
        semantic_outcomes[case.id] = comparison.temporal_state.value
    risk_scores = classification_metrics(risk_expected, risk_actual)

    semantic_support_rate = (
        supported_claims / evaluated_claims if evaluated_claims else 0.0
    )
    metrics = {
        "generation": {
            "faithfulness": round(semantic_support_rate, 6),
            "answer_relevancy": round(mean(answer_relevancy), 6),
            "supported_case_count": len(answer_relevancy),
            "generation_case_count": len(generation),
        },
        "citation": {
            "citation_validity_rate": round(sum(validity) / len(validity), 6),
            "semantic_support_rate": round(semantic_support_rate, 6),
            "unsupported_claim_detection": float(mismatch_detected),
            "verified_claim_count": evaluated_claims,
        },
        "temporal_disclosure": {
            "accuracy": round(temporal_scores.accuracy, 6),
            "macro_f1": round(temporal_scores.macro_f1, 6),
            "case_count": temporal_scores.count,
        },
        "risk_radar": {
            "accuracy": round(risk_scores.accuracy, 6),
            "macro_precision": round(risk_scores.macro_precision, 6),
            "macro_recall": round(risk_scores.macro_recall, 6),
            "macro_f1": round(risk_scores.macro_f1, 6),
            "case_count": risk_scores.count,
        },
    }
    latencies = {
        "overall": round(perf_counter() - started_all, 6),
        "citation": _latency_summary(verification_times),
        "temporal_alignment": _latency_summary(temporal_times),
        "risk_alignment": _latency_summary(risk_times),
    }
    return metrics, latencies, semantic_outcomes


def _unsupported_outcomes(
    db: Session,
    cases: list[EvaluationCase],
    *,
    metadata_outcomes: dict[str, str],
    generation: dict[str, GroundedGenerationResult | None],
    semantic_outcomes: dict[str, str],
) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for case in cases:
        execution = case.expected.get("execution")
        if case.generation is not None:
            outcomes[case.id] = semantic_outcomes.get(case.id, "generation_error")
        elif execution == "citation_mismatch":
            outcomes[case.id] = semantic_outcomes[case.id]
        elif execution == "temporal_resolution":
            try:
                TemporalDocumentResolver().resolve(
                    db,
                    company="Apple",
                    document_type="10-K",
                    older_year=int(case.expected["older_year"]),
                    newer_year=int(case.expected["newer_year"]),
                )
                outcomes[case.id] = "resolved"
            except TemporalResolutionError:
                outcomes[case.id] = "unavailable"
        elif execution == "metadata":
            documents = MetadataCatalog().list_documents(db)
            try:
                build_metadata_query_plan(case.question or "", documents)
                outcomes[case.id] = "resolved"
            except AmbiguousMetadataError:
                outcomes[case.id] = "ambiguous"
            except MetadataQueryError:
                outcomes[case.id] = "error"
        else:
            outcomes[case.id] = metadata_outcomes.get(case.id, "error")
    return outcomes


def _period_evidence(
    results: tuple[VectorSearchResult, ...],
    year: int,
    period: SourcePeriod,
    starting_id: int,
) -> DisclosurePeriodEvidence:
    return DisclosurePeriodEvidence(
        document_year=year,
        source_period=period,
        evidence=tuple(
            TemporalEvidence(starting_id + index, period, result)
            for index, result in enumerate(results)
        ),
        complete_anchor_scan=True,
    )


def _expected_outcome_rate(
    cases: list[EvaluationCase], outcomes: dict[str, str]
) -> float:
    if not cases:
        return 1.0
    return round(
        sum(outcomes.get(case.id) == case.expected_state for case in cases)
        / len(cases),
        6,
    )


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "max": 0.0}
    summary = {
        "count": len(values),
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "max": round(max(values), 6),
        "cold_start": round(values[0], 6),
    }
    if len(values) > 1:
        summary["warm_mean"] = round(mean(values[1:]), 6)
    return summary


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

        Client(host=settings.ollama_host).generate(
            model=settings.llm_model,
            prompt="",
            keep_alive=0,
        )
    except Exception as exc:
        print(f"warning: Ollama model unload request failed: {exc}", file=sys.stderr)


def _runtime_metadata(db: Session) -> dict[str, Any]:
    documents = db.execute(
        select(
            Document.id,
            Company.name,
            Document.document_type,
            Document.fiscal_year,
            func.count(Chunk.id).label("chunks"),
            func.count(Chunk.embedding).label("embedded"),
        )
        .join(Company, Document.company_id == Company.id)
        .join(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id, Company.name)
        .order_by(Document.id)
    ).all()
    extension = db.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar_one_or_none()
    revision = db.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "documents": [
            {
                "id": row.id,
                "company": row.name,
                "document_type": row.document_type,
                "fiscal_year": row.fiscal_year,
                "chunks": row.chunks,
                "embedded": row.embedded,
            }
            for row in documents
        ],
        "chunk_count": sum(row.chunks for row in documents),
        "embedded_count": sum(row.embedded for row in documents),
        "pgvector": extension,
        "alembic_revision": revision,
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    main()
