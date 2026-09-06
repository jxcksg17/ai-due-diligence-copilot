"""Tests for deterministic citation-anchored claim extraction."""

from app.verification.claims import extract_cited_claims


def test_extracts_multiple_citation_bearing_claims() -> None:
    claims = extract_cited_claims(
        "Net sales were $416 billion [1]. Supply shortages are a risk [2]."
    )
    assert [(claim.claim_text, claim.citation_ids) for claim in claims] == [
        ("Net sales were $416 billion.", (1,)),
        ("Supply shortages are a risk.", (2,)),
    ]


def test_adjacent_citations_remain_one_multi_evidence_claim() -> None:
    claims = extract_cited_claims(
        "Manufacturing concentration and shortages create supply risk [1] [2]."
    )
    assert len(claims) == 1
    assert claims[0].citation_ids == (1, 2)


def test_repeated_citation_id_is_deduplicated_within_claim() -> None:
    claims = extract_cited_claims("The filing reports the value [1] [1].")
    assert claims[0].citation_ids == (1,)
