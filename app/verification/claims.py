"""Deterministic extraction of citation-bearing claims."""

import re
from dataclasses import dataclass


_CITED_CLAIM_PATTERN = re.compile(
    r"(?P<claim>.*?)(?P<citations>(?:\s*\[\d+\])+)(?P<ending>[.!?]?)(?=\s|$)",
    re.DOTALL,
)
_CITATION_ID_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class ExtractedClaim:
    """One generated claim and the evidence IDs cited directly after it."""

    claim_text: str
    citation_ids: tuple[int, ...]


def extract_cited_claims(answer: str) -> list[ExtractedClaim]:
    """Extract claims using adjacent citation groups as inspectable boundaries."""
    normalized = re.sub(r"\s+", " ", answer).strip()
    claims: list[ExtractedClaim] = []
    for match in _CITED_CLAIM_PATTERN.finditer(normalized):
        claim_text = match.group("claim").strip(" .")
        if not claim_text:
            continue
        ending = match.group("ending")
        if ending:
            claim_text += ending
        citation_ids = tuple(
            dict.fromkeys(
                int(value)
                for value in _CITATION_ID_PATTERN.findall(match.group("citations"))
            )
        )
        claims.append(
            ExtractedClaim(claim_text=claim_text, citation_ids=citation_ids)
        )
    return claims
