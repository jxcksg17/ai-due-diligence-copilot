"""Versioned, manually inspectable evaluation-dataset contracts."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Capability = Literal[
    "retrieval",
    "metadata",
    "unsupported",
    "financial",
    "temporal",
    "risk_radar",
    "claim_evidence",
]


class DocumentScope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    company: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    fiscal_year: int


class EvidenceTarget(BaseModel):
    """Stable relevance target; no database row ID is required."""

    model_config = ConfigDict(extra="forbid", strict=True)

    page: int = Field(ge=1)
    anchors: list[str] = Field(min_length=1)


class GenerationExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    expected_answer_anchors: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False

    @model_validator(mode="after")
    def validate_supported_answer(self) -> "GenerationExpectation":
        if not self.insufficient_evidence and not self.expected_answer_anchors:
            raise ValueError("supported generation cases require answer anchors")
        if self.insufficient_evidence and self.expected_answer_anchors:
            raise ValueError("insufficient-evidence cases cannot require answer anchors")
        return self


class EvaluationCase(BaseModel):
    """One golden case with only the fields needed by its capability."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_]+$")
    capability: Capability
    question: str | None = None
    scope: DocumentScope | None = None
    expected_evidence: list[EvidenceTarget] = Field(default_factory=list)
    expected_state: str | None = None
    expected: dict[str, Any] = Field(default_factory=dict)
    generation: GenerationExpectation | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def validate_capability_fields(self) -> "EvaluationCase":
        if self.capability in {"retrieval", "metadata", "unsupported"}:
            if self.question is None or not self.question.strip():
                raise ValueError(f"{self.capability} cases require a question")
        if self.capability == "retrieval":
            if self.scope is None or not self.expected_evidence:
                raise ValueError("retrieval cases require scope and expected evidence")
        if self.capability in {
            "metadata",
            "unsupported",
            "temporal",
            "risk_radar",
            "claim_evidence",
        } and self.expected_state is None:
            raise ValueError(f"{self.capability} cases require expected_state")
        if self.capability == "financial" and not self.expected:
            raise ValueError("financial cases require deterministic expected values")
        if self.generation is not None and self.capability not in {
            "retrieval",
            "unsupported",
        }:
            raise ValueError("generation expectations are allowed only for RAG cases")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    evaluation_version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> "EvaluationDataset":
        identifiers = [case.id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evaluation case IDs must be unique")
        return self

    def by_capability(self, capability: Capability) -> list[EvaluationCase]:
        return [case for case in self.cases if case.capability == capability]


def load_dataset(path: str | Path) -> EvaluationDataset:
    source = Path(path)
    return EvaluationDataset.model_validate_json(source.read_text(encoding="utf-8"))
