from evals.reporting import build_report, load_report, write_report_atomic


def test_report_round_trip_records_models_and_configuration(tmp_path) -> None:
    report = build_report(
        evaluation_version="m11-v1",
        git_commit="abc123",
        mode="quick",
        models={"embedding": "bge", "generator": "qwen"},
        case_counts={"financial": 4},
        metrics={"deterministic": {"accuracy": 1.0}},
        latencies={"overall": 0.1},
        runtime={"python": "3.14.7"},
        case_outcomes={"case_b": "supported", "case_a": "ambiguous"},
    )
    path = tmp_path / "baseline.json"
    write_report_atomic(report, path)
    loaded = load_report(path)
    assert loaded["models"] == {"embedding": "bge", "generator": "qwen"}
    assert loaded["mandatory_external_api_cost_usd"] == 0
    assert list(loaded["case_outcomes"]) == ["case_a", "case_b"]
    assert not (tmp_path / "baseline.json.tmp").exists()


def test_invalid_report_is_rejected(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"metrics": {}}', encoding="utf-8")
    try:
        load_report(path)
    except ValueError as exc:
        assert "missing fields" in str(exc)
    else:
        raise AssertionError("invalid report was accepted")
