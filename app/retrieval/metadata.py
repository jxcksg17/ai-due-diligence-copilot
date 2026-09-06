"""Deterministic metadata-aware routing in front of vector retrieval."""

import re
from dataclasses import dataclass
from typing import Final

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Company, Document
from app.retrieval.vector_store import VectorSearchResult, VectorStore


class MetadataQueryError(ValueError):
    """Base error for metadata parsing and catalog resolution."""


class MissingMetadataScopeError(MetadataQueryError):
    """Raised when neither the question nor caller identifies a company."""


class AmbiguousMetadataError(MetadataQueryError):
    """Raised when metadata resolves to more than one supported document."""


class MetadataConflictError(MetadataQueryError):
    """Raised when parsed and caller-supplied metadata disagree."""


class UnsupportedMetadataError(MetadataQueryError):
    """Raised when requested metadata is unavailable in the catalog."""


class UnsupportedTemporalRequestError(MetadataQueryError):
    """Raised for multi-year requests reserved for temporal retrieval."""


class MissingSemanticQueryError(MetadataQueryError):
    """Raised when a question contains metadata but no searchable subject."""


@dataclass(frozen=True)
class DocumentMetadata:
    document_id: int
    company: str
    document_type: str
    fiscal_year: int


@dataclass(frozen=True)
class MetadataQueryPlan:
    """Resolved retrieval scope plus cleaned semantic query."""

    original_query: str
    semantic_query: str
    document: DocumentMetadata


_DOCUMENT_TYPE_PATTERNS: Final = {
    "10-K": re.compile(r"\b(?:form\s+)?10[\s-]?k\b|\bannual\s+report\b", re.I),
    "10-Q": re.compile(r"\b(?:form\s+)?10[\s-]?q\b|\bquarterly\s+report\b", re.I),
    "8-K": re.compile(r"\b(?:form\s+)?8[\s-]?k\b", re.I),
    "20-F": re.compile(r"\b(?:form\s+)?20[\s-]?f\b", re.I),
}
_YEAR_PATTERN: Final = re.compile(r"\b(?:19|20)\d{2}\b")
_DOCUMENT_ID_PATTERN: Final = re.compile(
    r"\bdocument\s+id\s*[:#]?\s*(?P<document_id>\d+)\b", re.I
)
_YEAR_TOKEN: Final = r"(?:(?:fiscal\s+year\s+|FY\s*)?(?:19|20)\d{2})"
_DOCUMENT_TYPE_TOKEN: Final = (
    r"(?:(?:form\s+)?(?:10[\s-]?k|10[\s-]?q|8[\s-]?k|20[\s-]?f)"
    r"|annual\s+report|quarterly\s+report)"
)
_DOCUMENT_SCOPE_PATTERN: Final = re.compile(
    rf"\b(?:in|from|according\s+to)\s+(?:(?:its|the)\s+)?(?:"
    rf"{_YEAR_TOKEN}\s+{_DOCUMENT_TYPE_TOKEN}"
    rf"|{_DOCUMENT_TYPE_TOKEN}\s+(?:for\s+)?{_YEAR_TOKEN})\b",
    re.I,
)
_DOCUMENT_ID_SCOPE_PATTERN: Final = re.compile(
    r"\b(?:in|from|according\s+to)\s+(?:the\s+)?"
    r"document\s+id\s*[:#]?\s*\d+\b",
    re.I,
)


class MetadataCatalog:
    """Read the document scopes that are actually available for retrieval."""

    def list_documents(self, db: Session) -> list[DocumentMetadata]:
        rows = db.execute(
            select(
                Document.id,
                Company.name,
                Document.document_type,
                Document.fiscal_year,
            )
            .join(Company, Document.company_id == Company.id)
            .order_by(Company.name, Document.fiscal_year, Document.document_type)
        ).all()
        return [
            DocumentMetadata(
                document_id=row.id,
                company=row.name,
                document_type=row.document_type,
                fiscal_year=row.fiscal_year,
            )
            for row in rows
        ]


def build_metadata_query_plan(
    query: str,
    documents: list[DocumentMetadata],
    *,
    company: str | None = None,
    document_id: int | None = None,
    document_type: str | None = None,
    fiscal_year: int | None = None,
) -> MetadataQueryPlan:
    """Parse explicit metadata and resolve it to exactly one catalog document."""
    if not query.strip():
        raise ValueError("query must not be empty")

    detected_metadata_spans: list[tuple[int, int]] = []
    company_matches: list[tuple[str, re.Match[str]]] = []
    for known_company in sorted({item.company for item in documents}):
        pattern = re.compile(
            rf"(?<!\w){re.escape(known_company)}(?:['’]s)?(?!\w)", re.I
        )
        match = pattern.search(query)
        if match is not None:
            company_matches.append((known_company, match))
    parsed_companies = {name for name, _ in company_matches}
    if len(parsed_companies) > 1:
        raise AmbiguousMetadataError("question names multiple supported companies")
    parsed_company = next(iter(parsed_companies), None)
    detected_metadata_spans.extend(match.span() for _, match in company_matches)

    parsed_types: list[str] = []
    for canonical_type, pattern in _DOCUMENT_TYPE_PATTERNS.items():
        match = pattern.search(query)
        if match is not None:
            parsed_types.append(canonical_type)
            detected_metadata_spans.append(match.span())
    if len(parsed_types) > 1:
        raise AmbiguousMetadataError("question names multiple document types")
    parsed_type = parsed_types[0] if parsed_types else None

    year_matches = list(_YEAR_PATTERN.finditer(query))
    parsed_years = {int(match.group()) for match in year_matches}
    if len(parsed_years) > 1:
        raise UnsupportedTemporalRequestError(
            "multi-year metadata requests require the later temporal milestone"
        )
    parsed_year = next(iter(parsed_years), None)
    detected_metadata_spans.extend(match.span() for match in year_matches)

    document_id_match = _DOCUMENT_ID_PATTERN.search(query)
    parsed_document_id = (
        int(document_id_match.group("document_id"))
        if document_id_match is not None
        else None
    )
    if document_id_match is not None:
        detected_metadata_spans.append(document_id_match.span())

    resolved_company = _merge_metadata("company", parsed_company, company)
    resolved_type = _merge_metadata("document type", parsed_type, document_type)
    resolved_year = _merge_metadata("fiscal year", parsed_year, fiscal_year)
    resolved_document_id = _merge_metadata(
        "document ID", parsed_document_id, document_id
    )
    if resolved_company is None and resolved_document_id is None:
        raise MissingMetadataScopeError(
            "company must be supplied explicitly or named in the question"
        )

    matches = [
        item
        for item in documents
        if (
            resolved_company is None
            or item.company.casefold() == str(resolved_company).casefold()
        )
        and (resolved_type is None or item.document_type == resolved_type)
        and (resolved_year is None or item.fiscal_year == resolved_year)
        and (
            resolved_document_id is None
            or item.document_id == resolved_document_id
        )
    ]
    if not matches:
        raise UnsupportedMetadataError(
            "requested company/document metadata is not available"
        )
    if len(matches) > 1:
        raise AmbiguousMetadataError(
            "metadata matches multiple documents; specify document type or fiscal year"
        )

    subject_query = _remove_spans(query, detected_metadata_spans)
    if not re.search(r"[A-Za-z0-9]", subject_query):
        raise MissingSemanticQueryError(
            "question must include a subject beyond company/document metadata"
        )
    scope_spans = [
        match.span()
        for pattern in (_DOCUMENT_SCOPE_PATTERN, _DOCUMENT_ID_SCOPE_PATTERN)
        for match in pattern.finditer(query)
    ]
    semantic_query = _remove_spans(query, scope_spans)
    return MetadataQueryPlan(
        original_query=query,
        semantic_query=semantic_query,
        document=matches[0],
    )


def _merge_metadata(label: str, parsed: object, explicit: object) -> object:
    if parsed is not None and explicit is not None:
        if isinstance(parsed, str) and isinstance(explicit, str):
            agrees = parsed.casefold() == explicit.casefold()
        else:
            agrees = parsed == explicit
        if not agrees:
            raise MetadataConflictError(
                f"question {label} {parsed!r} conflicts with explicit value {explicit!r}"
            )
    return explicit if explicit is not None else parsed


def _remove_spans(query: str, spans: list[tuple[int, int]]) -> str:
    characters = list(query)
    for start, end in spans:
        characters[start:end] = " " * (end - start)
    cleaned = "".join(characters)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)
    return cleaned.strip(" ,;:-")


class MetadataAwareRetriever:
    """Resolve metadata against the catalog, then delegate to pgvector."""

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        catalog: MetadataCatalog | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._catalog = catalog or MetadataCatalog()

    def plan(
        self,
        db: Session,
        query: str,
        *,
        company: str | None = None,
        document_id: int | None = None,
        document_type: str | None = None,
        fiscal_year: int | None = None,
    ) -> MetadataQueryPlan:
        return build_metadata_query_plan(
            query,
            self._catalog.list_documents(db),
            company=company,
            document_id=document_id,
            document_type=document_type,
            fiscal_year=fiscal_year,
        )

    def search(
        self,
        db: Session,
        query: str,
        *,
        company: str | None = None,
        top_k: int = 5,
        document_id: int | None = None,
        document_type: str | None = None,
        fiscal_year: int | None = None,
    ) -> list[VectorSearchResult]:
        plan = self.plan(
            db,
            query,
            company=company,
            document_id=document_id,
            document_type=document_type,
            fiscal_year=fiscal_year,
        )
        document = plan.document
        return self._vector_store.search(
            db,
            plan.semantic_query,
            company=document.company,
            top_k=top_k,
            document_id=document.document_id,
            document_type=document.document_type,
            fiscal_year=document.fiscal_year,
        )
