"""Local natural-language-inference verifier abstraction."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

from app.config import Settings


CITATION_VERIFIER_MODEL = "cross-encoder/nli-deberta-v3-small"


class VerifierConfigurationError(ValueError):
    """Raised when an NLI model does not expose the required labels."""


@dataclass(frozen=True)
class EntailmentProbabilities:
    """Model softmax probabilities for the three NLI classes."""

    contradiction: float
    entailment: float
    neutral: float

    def __post_init__(self) -> None:
        values = (self.contradiction, self.entailment, self.neutral)
        if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise ValueError("NLI probabilities must be finite values between 0 and 1")
        if not 0.99 <= sum(values) <= 1.01:
            raise ValueError("NLI probabilities must sum to approximately 1")


class EntailmentVerifier(Protocol):
    """Score premise/hypothesis pairs with three-way NLI probabilities."""

    @property
    def model_name(self) -> str:
        """Return the verifier model identifier."""

    def verify_many(
        self, pairs: list[tuple[str, str]]
    ) -> list[EntailmentProbabilities]:
        """Return one probability triple for every input pair."""


def _select_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalNLIVerifier:
    """Sentence Transformers NLI cross-encoder with label introspection."""

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        device: str | None = None,
        encoder: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._model_name = model_name
        self._batch_size = batch_size
        self.device = device or _select_device()
        if encoder is None:
            from sentence_transformers import CrossEncoder

            try:
                encoder = CrossEncoder(model_name, device=self.device, max_length=512)
            except RuntimeError:
                if self.device != "mps":
                    raise
                self.device = "cpu"
                encoder = CrossEncoder(model_name, device="cpu", max_length=512)
        self._encoder = encoder
        self._label_indices = _resolve_label_indices(encoder)

    @property
    def model_name(self) -> str:
        return self._model_name

    def verify_many(
        self, pairs: list[tuple[str, str]]
    ) -> list[EntailmentProbabilities]:
        if not pairs:
            return []
        if any(not premise.strip() or not hypothesis.strip() for premise, hypothesis in pairs):
            raise ValueError("NLI premise and hypothesis must not be empty")
        scores = self._encoder.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            apply_softmax=True,
            convert_to_numpy=True,
        )
        contradiction_index = self._label_indices["contradiction"]
        entailment_index = self._label_indices["entailment"]
        neutral_index = self._label_indices["neutral"]
        return [
            EntailmentProbabilities(
                contradiction=float(row[contradiction_index]),
                entailment=float(row[entailment_index]),
                neutral=float(row[neutral_index]),
            )
            for row in scores
        ]


def _resolve_label_indices(encoder: Any) -> dict[str, int]:
    raw_labels = encoder.model.config.id2label
    normalized = {
        str(label).casefold(): int(index) for index, label in raw_labels.items()
    }
    missing = {
        label for label in ("contradiction", "entailment", "neutral") if label not in normalized
    }
    if missing:
        raise VerifierConfigurationError(
            f"NLI model is missing required labels: {', '.join(sorted(missing))}"
        )
    return {label: normalized[label] for label in normalized if label in {
        "contradiction", "entailment", "neutral"
    }}


def get_citation_verifier(settings: Settings) -> EntailmentVerifier:
    if settings.citation_verifier_model != CITATION_VERIFIER_MODEL:
        raise ValueError(
            "Unsupported CITATION_VERIFIER_MODEL "
            f"{settings.citation_verifier_model!r}; expected {CITATION_VERIFIER_MODEL!r}"
        )
    return LocalNLIVerifier(
        model_name=settings.citation_verifier_model,
        batch_size=settings.citation_verifier_batch_size,
    )
