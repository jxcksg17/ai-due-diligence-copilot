"""M7 verification adapter for generated Risk Radar interpretations."""

from app.risk_radar.generation import RiskRadarAnalysis
from app.verification.service import (
    CitationVerificationReport,
    CitationVerificationService,
)


class RiskRadarVerificationService:
    """Verify generated risk claims without changing evidence or signals."""

    def __init__(self, citation_verifier: CitationVerificationService) -> None:
        self._citation_verifier = citation_verifier

    def verify(self, analysis: RiskRadarAnalysis) -> CitationVerificationReport:
        return self._citation_verifier.verify_generation(analysis.interpretation)
