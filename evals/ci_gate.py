"""Fast, model-free integrity gate for the approved M11 evaluation baseline."""

from pathlib import Path

from evals.domain import evaluate_financial
from evals.reporting import load_report
from evals.schema import load_dataset


ROOT = Path(__file__).resolve().parent.parent


def run() -> None:
    dataset = load_dataset(ROOT / "evals/datasets/m11_v1.json")
    baseline = load_report(ROOT / "evals/baselines/m11_v1.json")
    expected_counts = {key: len(dataset.by_capability(key)) for key in baseline["case_counts"]}
    if expected_counts != baseline["case_counts"]:
        raise SystemExit("golden dataset counts no longer match the approved baseline")
    if dataset.evaluation_version != baseline["evaluation_version"]:
        raise SystemExit("evaluation version no longer matches the approved baseline")

    synthetic = [
        case
        for case in dataset.by_capability("financial")
        if case.expected.get("execution") != "real_temporal"
    ]
    current = evaluate_financial(synthetic, real_temporal=None)
    required = ("numeric_accuracy", "classification_accuracy", "rounding_accuracy", "provenance_accuracy")
    for metric in required:
        if current[metric] != 1.0:
            raise SystemExit(f"deterministic financial regression: {metric}={current[metric]}")
    print(f"M11 CI gate passed: {len(dataset.cases)} cases; deterministic finance remains exact")


if __name__ == "__main__":
    run()
