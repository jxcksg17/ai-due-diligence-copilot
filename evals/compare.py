"""Compare a current M11 report against an approved baseline."""

import argparse
import json
from pathlib import Path

from evals.regression import compare_reports
from evals.reporting import load_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    arguments = parser.parse_args()
    result = compare_reports(
        load_report(arguments.baseline), load_report(arguments.current)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["overall"] == "regressed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
