"""Small, explicit filing-risk taxonomy for the M9 Risk Radar."""

from dataclasses import dataclass


class UnsupportedRiskTopicError(ValueError):
    """Raised when a caller requests a topic outside the M9 taxonomy."""


@dataclass(frozen=True)
class RiskTopic:
    """One supported risk identity and its deterministic discovery anchors."""

    key: str
    label: str
    query: str
    anchor_groups: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.label.strip() or not self.query.strip():
            raise ValueError("risk topic fields must not be empty")
        if not self.anchor_groups or any(not group for group in self.anchor_groups):
            raise ValueError("risk topic anchor groups must not be empty")


SUPPLY_CHAIN_MANUFACTURING = RiskTopic(
    key="supply_chain_manufacturing",
    label="Supply chain and manufacturing",
    query="supplier concentration manufacturing components supply shortages",
    anchor_groups=(
        ("supply", "supplier", "suppliers", "source", "sources", "outsourcing"),
        (
            "component",
            "components",
            "manufacturing",
            "manufacturer",
            "manufacturers",
            "shortage",
            "shortages",
        ),
    ),
)

COMPETITION = RiskTopic(
    key="competition",
    label="Competition and innovation",
    query="competitive markets products services pricing margins innovation",
    anchor_groups=(
        ("competitive", "competition", "competitor", "competitors"),
        ("product", "products", "service", "services", "market", "markets"),
        (
            "price",
            "pricing",
            "margin",
            "margins",
            "innovation",
            "innovative",
            "technology",
            "technological",
        ),
    ),
)

CYBERSECURITY_INFORMATION_SYSTEMS = RiskTopic(
    key="cybersecurity_information_systems",
    label="Cybersecurity and information systems",
    query="cybersecurity attacks information systems network disruption data security",
    anchor_groups=(
        ("cybersecurity", "cyber", "ransomware", "malicious", "attack", "attacks"),
        ("system", "systems", "network", "networks", "data", "information", "security"),
    ),
)

LEGAL_REGULATORY = RiskTopic(
    key="legal_regulatory",
    label="Legal and regulatory",
    query="legal regulatory compliance investigations litigation penalties laws",
    anchor_groups=(
        ("legal", "regulatory", "regulation", "regulations", "litigation", "antitrust"),
        (
            "compliance",
            "investigation",
            "investigations",
            "proceeding",
            "proceedings",
            "penalty",
            "penalties",
            "law",
            "laws",
        ),
    ),
)

RISK_TOPICS = (
    SUPPLY_CHAIN_MANUFACTURING,
    COMPETITION,
    CYBERSECURITY_INFORMATION_SYSTEMS,
    LEGAL_REGULATORY,
)
_RISK_TOPICS_BY_KEY = {topic.key: topic for topic in RISK_TOPICS}


def get_risk_topic(key: str) -> RiskTopic:
    """Resolve only explicitly supported topic keys; never guess aliases."""
    try:
        return _RISK_TOPICS_BY_KEY[key]
    except KeyError as exc:
        supported = ", ".join(topic.key for topic in RISK_TOPICS)
        raise UnsupportedRiskTopicError(
            f"unsupported risk topic {key!r}; supported topics: {supported}"
        ) from exc


def resolve_risk_topics(keys: tuple[str, ...] | None = None) -> tuple[RiskTopic, ...]:
    """Return the stable default taxonomy or an explicit supported subset."""
    if keys is None:
        return RISK_TOPICS
    if not keys:
        raise UnsupportedRiskTopicError("at least one risk topic is required")
    if len(keys) != len(set(keys)):
        raise UnsupportedRiskTopicError("risk topic keys must not contain duplicates")
    return tuple(get_risk_topic(key) for key in keys)
