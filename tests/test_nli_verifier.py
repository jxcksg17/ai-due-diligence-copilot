"""Tests for the local NLI adapter without downloading a model."""

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.verification.verifier import (
    LocalNLIVerifier,
    VerifierConfigurationError,
    get_citation_verifier,
)


class FakeEncoder:
    def __init__(self, labels: dict[int, str] | None = None) -> None:
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                id2label=labels
                or {0: "contradiction", 1: "entailment", 2: "neutral"}
            )
        )
        self.calls: list[tuple[object, dict[str, object]]] = []

    def predict(self, pairs: object, **kwargs: object) -> list[list[float]]:
        self.calls.append((pairs, kwargs))
        return [[0.1, 0.8, 0.1]]


def test_nli_adapter_maps_model_labels_and_requests_softmax() -> None:
    encoder = FakeEncoder()
    verifier = LocalNLIVerifier(
        model_name="fake-model",
        batch_size=4,
        device="cpu",
        encoder=encoder,
    )
    result = verifier.verify_many([("source evidence", "generated claim")])
    assert result[0].entailment == pytest.approx(0.8)
    assert result[0].contradiction == pytest.approx(0.1)
    assert encoder.calls == [
        (
            [("source evidence", "generated claim")],
            {
                "batch_size": 4,
                "show_progress_bar": False,
                "apply_softmax": True,
                "convert_to_numpy": True,
            },
        )
    ]


def test_nli_adapter_rejects_model_without_semantic_labels() -> None:
    with pytest.raises(VerifierConfigurationError, match="required labels"):
        LocalNLIVerifier(
            model_name="bad-model",
            batch_size=4,
            device="cpu",
            encoder=FakeEncoder({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}),
        )


def test_verifier_factory_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class FakeLocalVerifier:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)

    monkeypatch.setattr(
        "app.verification.verifier.LocalNLIVerifier", FakeLocalVerifier
    )
    get_citation_verifier(Settings(database_url="sqlite://"))
    assert created == {
        "model_name": "cross-encoder/nli-deberta-v3-small",
        "batch_size": 8,
    }
