import json

import pytest
from pydantic import ValidationError

from evals.schema import EvaluationCase, EvaluationDataset, load_dataset


def test_versioned_dataset_loads_from_json(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "evaluation_version": "test-v1",
                "description": "A reviewed test dataset.",
                "cases": [
                    {
                        "id": "fin_case",
                        "capability": "financial",
                        "expected": {"percentage": "10.0"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = load_dataset(path)
    assert dataset.evaluation_version == "test-v1"
    assert dataset.cases[0].id == "fin_case"


def test_dataset_rejects_duplicate_case_ids() -> None:
    case = EvaluationCase(
        id="fin_case",
        capability="financial",
        expected={"percentage": "10.0"},
    )
    with pytest.raises(ValidationError, match="must be unique"):
        EvaluationDataset(
            evaluation_version="test-v1",
            description="Reviewed.",
            cases=[case, case],
        )


def test_retrieval_case_requires_stable_evidence_target() -> None:
    with pytest.raises(ValidationError, match="expected evidence"):
        EvaluationCase(
            id="ret_case",
            capability="retrieval",
            question="What was revenue?",
            scope={"company": "Apple", "document_type": "10-K", "fiscal_year": 2025},
        )


def test_metadata_case_requires_expected_state() -> None:
    with pytest.raises(ValidationError, match="expected_state"):
        EvaluationCase(
            id="meta_case",
            capability="metadata",
            question="What was revenue?",
        )


def test_supported_generation_case_requires_answer_anchors() -> None:
    with pytest.raises(ValidationError, match="require answer anchors"):
        EvaluationCase(
            id="ret_case",
            capability="retrieval",
            question="What was revenue?",
            scope={"company": "Apple", "document_type": "10-K", "fiscal_year": 2025},
            expected_evidence=[{"page": 1, "anchors": ["revenue"]}],
            generation={"insufficient_evidence": False},
        )


def test_insufficient_generation_case_rejects_expected_facts() -> None:
    with pytest.raises(ValidationError, match="cannot require answer anchors"):
        EvaluationCase(
            id="unsupported_case",
            capability="unsupported",
            question="Future price?",
            expected_state="insufficient_evidence",
            generation={
                "insufficient_evidence": True,
                "expected_answer_anchors": ["price"],
            },
        )
